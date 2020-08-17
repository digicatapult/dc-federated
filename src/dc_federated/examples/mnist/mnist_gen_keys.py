"""
Simple file to generate keys for the mnist experiment.
"""
import os
from nacl.encoding import HexEncoder


from dc_federated.backend.worker_key_pair_tool import gen_pair, verify_pair


def gen_keys_for_mnist(num_workers=3):
    """
    Generate keys for use with the mnist example.

    Parameters
    ----------

    num_workers: int
        Number of workers.
    """
    private_keys = []
    public_keys = []

    for n in range(num_workers):
        private_key, public_key = gen_pair('mnist_worker' + f'_{n}_' + 'key')
        private_keys.append(private_key)
        public_keys.append(public_key)

    worker_key_file = 'mnist_key_list_file.txt'
    with open(worker_key_file, 'w') as f:
        for public_key in public_keys[:-1]:
            f.write(public_key.encode(encoder=HexEncoder).decode('utf-8') + os.linesep)
        f.write(public_keys[-1].encode(encoder=HexEncoder).decode('utf-8') + os.linesep)


if __name__ == '__main__':
    gen_keys_for_mnist()