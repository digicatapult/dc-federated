# Worker authentication

The `dc_federated` library backend supports worker authentication via `Ed25519` based public/private key cryptography algorithm using the `libsodium` library. The scheme works as follows. 

- Before starting the training, a public/private key pair is generated for each worker using the `dc_federated.backend.worker_key_pair` tool. 
- The public keys for all the workers is sent to the server and is be put in a plain text file, one key per worker.
    - If such a file is not given, the server starts in an `Usafe Mode` and accepts any nodes that tries to register.
- When the server is started the key list file is the passed to the server as an argument. At this point only workers only workers with a public key in that file will be allowed to participate in the training or request models from the server.
- When starting the worker, the name of the private key file is passed on as an argument. The client also assumes that the corresponding public key is available in the same folder as the private key but with a `.pub` extension (this is the format in which the tool generates the keys).
- When the worker communicates with the server, it uses the public key to authenticate itself.

For details on how this may be used in an algorithm or an application please see the documentation for the mnist example in `docs/mnist.md`.

The key pairs for a worker may be generated from the command line as follows. Go to the folder `src/dc_federated/backend`. The run
```bash
> python worker_key_pair_tool.py <filename>
``` 
The private key will be in `<filename>` and the corresponding public key will be in `<filename>.pub`. As usual, the private key file needs to be kept secret. 
