import os
import sys
import argparse
from nacl.encoding import HexEncoder

from dc_federated.backend.worker_key_pair_tool import gen_pair


def gen_stress_key_pairs(keys_folder, num_workers):
    """
    Generate the keys for the stress test. The list of public keys
    is put in keys_folder/'stress_worker_public_keys.txt' while
    each worker private key is put in 'stress_worker_key_file_{n}'
    (+ '.pub' for the public key).

    Parameters
    -----------

    keys_folder: str
        The name and location of the key folder.

    num_workers: int
        The number of workers to generate the keys for.
    """
    private_keys = []
    public_keys = []

    if not os.path.exists(keys_folder):
        os.mkdir(keys_folder)
    worker_key_file_prefix = 'stress_worker_key_file'

    for n in range(num_workers):
        private_key, public_key = gen_pair(os.path.join(keys_folder, worker_key_file_prefix + f'_{n}'))
        private_keys.append(private_key)
        public_keys.append(public_key)

    worker_key_file = os.path.join(keys_folder, 'stress_worker_public_keys.txt')
    with open(worker_key_file, 'w') as f:
        for public_key in public_keys[:-1]:
            f.write(public_key.encode(encoder=HexEncoder).decode('utf-8') + os.linesep)
        f.write(public_keys[-1].encode(encoder=HexEncoder).decode('utf-8') + os.linesep)


def get_args():
    """
    Parse the argument for generating the keys.
    """
    # Make parser object
    p = argparse.ArgumentParser(
        description="Generate the keys for the stress-test\n")

    p.add_argument("--keys-folder",
                   help="The location where keys will be put.",
                   type=str,
                   default=None,
                   required=False)
    p.add_argument("--num-workers",
                   help="The number of workers to generate the keys for.",
                   type=int,
                   required=True)

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    gen_stress_key_pairs(args.keys_folder, args.num_workers)
