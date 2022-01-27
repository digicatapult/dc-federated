from unittest.mock import patch, ANY
from dc_federated.backend.zmq_interface import ZMQInterfaceServer
from dc_federated.backend.dcf_server import DCFServer
from dc_federated.backend.subprocess_dcf_server import run as run_subprocess

server_subprocess_args = {"arg1": "test1", "arg2": "test2"}


@patch.object(
    ZMQInterfaceServer, "server_args_request_send", return_value=server_subprocess_args
)
@patch.object(ZMQInterfaceServer, "__init__", return_value=None)
@patch.object(DCFServer, "start_server")
@patch.object(DCFServer, "__init__", return_value=None)
def test_requests_args(
    DCFServer_init_mock, DCFServer_start_mock, ZQMIS_init_mock, server_args_request_mock
):
    """
    Tests that the script initialises the ZMQ interface correctly and starts
    the server with the correct args.
    """
    run_subprocess("test_port")
    ZQMIS_init_mock.assert_called_once_with("test_port")
    server_args_request_mock.assert_called_once_with()
    DCFServer_init_mock.assert_called_once_with(
        register_worker_callback=ANY,
        unregister_worker_callback=ANY,
        return_global_model_callback=ANY,
        is_global_model_most_recent=ANY,
        receive_worker_update_callback=ANY,
        **server_subprocess_args,
    )
    DCFServer_start_mock.assert_called_once_with()
