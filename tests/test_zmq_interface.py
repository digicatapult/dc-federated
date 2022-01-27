from dc_federated.backend.zmq_interface import ZMQInterfaceModel, ZMQInterfaceServer
import zmq
from unittest.mock import Mock, patch
import threading
import time
import pytest

# Initalise some mocks that can be used in place of the callbacks to check that
# they are called. Along with some other constants that are used to check for
# equality in the tests.
register_worker_callback = Mock()
unregister_worker_callback = Mock()
GLOBAL_MODEL = {"test": "test_val"}
return_global_model_callback = Mock(return_value=GLOBAL_MODEL)
IS_MOST_RECENT = True
is_global_model_most_recent = Mock(return_value=IS_MOST_RECENT)
WORKER_UPDATE = "test_string"
receive_worker_update_callback = Mock(return_value=WORKER_UPDATE)
server_subprocess_args = ["test", "subprocess", "args", 1, True, None]
port = 5556


def mock_recv(context, socket, fn_name, close=True):
    """
    A function that takes a zmq recv function and mocks it from being a
    blocking call to a polled call. This means that it can be threaded using
    the same interpreter without preventing the test from continuing/finishing.
    """

    original_recv = getattr(socket, fn_name)  # Hold original recv function

    def recv_py(flags=0):
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        while True:
            evts = poller.poll(500)
            if len(evts) > 0:
                # If we get a message call the original blocking recv
                result = original_recv(flags=flags)
                if close:
                    socket.close()
                    context.term()
                return result
            # Release the GIL to send/receive messages on the other socket.
            time.sleep(0.5)

    setattr(socket, fn_name, Mock(side_effect=recv_py))


def mock_new_socket():
    """
    A function for mocking the ZMQInterfaceServer `_new_socket` function that
    converts the relevant "recv" functions from blocking to non-blocking.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")

    mock_recv(context, socket, "recv_pyobj")
    mock_recv(context, socket, "recv")
    mock_recv(context, socket, "recv_string")
    return socket


@pytest.fixture(autouse=True)
def run_model_interface():
    """
    A function for initialising the REP socket that is initialised by the
    model-process. This is done in its own thread with a non-blocking
    `recv_multipart` so that the server-process can send/receive messages too.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")

    mock_recv(context, socket, "recv_multipart", close=False)

    zmqM = ZMQInterfaceModel(
        socket=socket,
        register_worker_callback=register_worker_callback,
        unregister_worker_callback=unregister_worker_callback,
        return_global_model_callback=return_global_model_callback,
        is_global_model_most_recent=is_global_model_most_recent,
        receive_worker_update_callback=receive_worker_update_callback,
        server_subprocess_args=server_subprocess_args,
    )
    thread = threading.Thread(target=zmqM.receive, daemon=True)
    thread.start()
    yield thread
    socket.close()
    context.term()


zmqS = ZMQInterfaceServer(port=port)

# Test each of the interface's functions
@patch.object(zmqS, "_new_socket", mock_new_socket)
def test_server_args_request():
    result = zmqS.server_args_request_send()
    assert result == server_subprocess_args


@patch.object(zmqS, "_new_socket", mock_new_socket)
def test_register_worker():
    result = zmqS.register_worker_send("test123")
    assert result == b"1"
    register_worker_callback.assert_called_once_with("test123")


@patch.object(zmqS, "_new_socket", mock_new_socket)
def test_unregister_worker():
    result = zmqS.unregister_worker_send("test123")
    assert result == b"1"
    unregister_worker_callback.assert_called_once_with("test123")


@patch.object(zmqS, "_new_socket", mock_new_socket)
def test_return_global_model():
    result = zmqS.return_global_model_send()
    assert result == GLOBAL_MODEL
    return_global_model_callback.assert_called_once_with()


@patch.object(zmqS, "_new_socket", mock_new_socket)
def test_is_global_model_most_recent():
    result = zmqS.is_global_model_most_recent_send(123)
    assert result == IS_MOST_RECENT
    is_global_model_most_recent.assert_called_once_with(123)


@patch.object(zmqS, "_new_socket", mock_new_socket)
def test_receive_worker_update():
    result = zmqS.receive_worker_update_send("test123", b"model_update")
    assert result == WORKER_UPDATE
    receive_worker_update_callback.assert_called_once_with("test123", b"model_update")
