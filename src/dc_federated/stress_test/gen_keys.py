import os
import sys
from nacl.encoding import HexEncoder

from dc_federated.backend.worker_key_pair_tool import gen_pair

# Create a set of keys to be supplied to the server
num_workers = int(sys.argv[1])
private_keys = []
public_keys = []

keys_folder = 'stress_keys_folder'
if not os.path.exists(keys_folder):
    os.mkdir(keys_folder)
worker_key_file_prefix = 'worker_key_file'

for n in range(num_workers):
    private_key, public_key = gen_pair(os.path.join(keys_folder, worker_key_file_prefix + f'_{n}'))
    private_keys.append(private_key)
    public_keys.append(public_key)

worker_key_file = os.path.join(keys_folder, 'stress_worker_public_keys.txt')
with open(worker_key_file, 'w') as f:
    for public_key in public_keys[:-1]:
        f.write(public_key.encode(encoder=HexEncoder).decode('utf-8') + os.linesep)
    f.write(public_keys[-1].encode(encoder=HexEncoder).decode('utf-8') + os.linesep)
