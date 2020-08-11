"""
Test authentication related facilities.
"""
import os

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
from nacl.exceptions import BadSignatureError

from dc_federated.backend import DCFServer, DCFWorker
from dc_federated.backend._constants import *
from dc_federated.backend.worker_key_pair_tool import gen_pair, verify_pair


def test_worker_key_pair():
    key_file = "gen_pair_test"
    private_key, public_key = gen_pair(key_file)

    test_phrase = "Test phrase"
    try:
        public_key.verify(private_key.sign(test_phrase))
    except BadSignatureError as bse:
        assert False

    # load the generated keys from the file
    with open(key_file, 'r') as f:
        loaded_private_key = SigningKey(f.read().encode(), encoder=HexEncoder)
    with open(key_file + '.pub', 'r') as f:
        loaded_public_key = VerifyKey(f.read().encode(), encoder=HexEncoder)

    # test the
    try:
        loaded_public_key.verify(loaded_public_key.sign(test_phrase))
    except BadSignatureError as bse:
        assert False
    assert verify_pair(key_file)

    # test that a bad signature is detected
    with open(key_file, 'w') as f:
        f.write(SigningKey.generate().encode(encoder=HexEncoder).decode('utf-8'))
    assert not verify_pair(key_file)

    # clean up
    os.remove(key_file)
    os.remove(key_file + '.pub')


def test_worker_authentication():
    # Create a set of keys to be supplied to the server
    num_workers = 10
    private_keys = [SigningKey.generate() for n in range(num_workers)]
    public_keys = [private_key.verify_key for private_key in private_keys]

    worker_ids = []
    worker_updates = {}
    status = 'Status is good!!'

    def test_register_func_cb(id):
        worker_ids.append(id)

    def test_ret_global_model_cb():
        return pickle.dumps("Pickle dump of a string")

    def test_query_status_cb():
        return status

    def test_rec_server_update_cb(worker_id, update):
        if worker_id in worker_ids:
            worker_updates[worker_id] = update
            return f"Update received for worker {worker_id}."
        else:
            return f"Unregistered worker {worker_id} tried to send an update."

    def test_glob_mod_chng_cb():
        pass

    worker_key_file = 'worker_public_keys.txt'

    with open(worker_key_file, 'w') as f:
        for public_key in public_keys:
            f.write(public_key)

    dcf_server = DCFServer(
        test_register_func_cb,
        test_ret_global_model_cb,
        test_query_status_cb,
        test_rec_server_update_cb,
        key_file=worker_key_file
    )
    server_thread = Thread(target=dcf_server.start_server)
    server_thread.start()
    time.sleep(2)


if __name__ == '__main__':
    test_worker_key_pair()
