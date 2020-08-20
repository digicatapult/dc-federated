## The MNIST Federated Learning Example

This document will show you how to run the MNIST federated learning example included in the repo in a fully distributed fashion. In particular, it will show you:

- how to start the server
- how to start three different workers in one or more machines that are connected via the internet
- run the `FedAvg` algorithm to solve the MNIST problem
- use the authentication facility in `dc_federated` to authenticate workers.

### The problem

The MNIST dataset is, at this point, one of the simplest machine vision benchmark problems for deep neural networks. It consists of images of handwritten digits (0 to 9) and the task is to recognize which digit each image corresponds to. In the federated setting we describe below, there are three workers and each worker will only have images for a particular subset of the digits - worker `0` will have images for digits `0 - 3`, worker `1` for `4 - 6` and worker `2` for `7 - 9`. During federated learning the goal will be to create a model at the server that is equally good on the combined dataset but without looking at all three datasets.

### Running the Example

We assume that you are running on Linux or Mac OS. You will need to translate the instructions for Windows, which should be quite similar. Follow the steps below to run the example:

#### Setting up

Clone and install the repo in the machines where you want to deploy the servers and the workers (as described in `docs/getting_started.md`). _You can skip this step if you are running everything on the same machine._ Open a terminal for the server, and one terminal for each of the three workers.

#### Start The Server

> NOTE: The machine that the server is running on must be enabled to host a webserver accessible to the machines running the workers.

First, go to the terminal for the server and activate the environment:

```bash
> source /path/to/venv_dc_federated
```

Now `cd` into the `dc_federated` package root folder and then to the mnist example folder

```bash
(venv_dc_federated)> cd src/dc_federated/examples/mnist
```

now start the server:

```bash
(venv_dc_federated)> python mnist_fed_avg_server.py
```

You should see an output of the following form:

```bash
IINFO:dc_federated.algorithms.fed_avg.fed_avg_server:Initializing FedAvg server for model class MNISTNet
WARNING:dc_federated.backend.dcf_server:No key list file provided - no worker authentication will be used!!!.
WARNING:dc_federated.backend.dcf_server:Server is running in ****UNSAFE MODE.****

************
Starting an Federated Average Server at
	server-host-ip: 192.124.1.177
	server-port: 8080

************
```

But with different `server-host-ip` but the same `server-port`. You may see additional output if you are downloading the mnist dataset. Note the warning that the server is starting in `unsafe mode` - this is because that the worker authentication is not being used. See below for how to run this example with the workers being authenticated.

#### Start the Workers

Now move to a terminal for a worker and `cd` into the location where the library is installed and, then to the mnist folder (same as above), activate the virtual enironment and run:

```bash
(venv_dc_federated)> python mnist_fed_avg_worker.py --server-host-ip 192.124.1.177  --server-port 8080 --digit-class 0
```

The `--digit-class 0` argument means that this worker only only train on digits `0-3`. Using arguments `1` and `2` correspond to training only on digits `4-6` and `7-9` respectively. The `192.124.1.177` and `8080` should be replaced by the values obtained when the server was run.

Once you run the worker, you should see an output of the form

```bash
INFO:dc_federated.algorithms.fed_avg.fed_avg_worker:Registered with FedAvg Server with worker id 7aa788c425ae1132672327085065281c79e017dd4821118b314a01f627348530
Train Epoch: 0 [0/24754(0%)]	Loss: 2.341936
Train Epoch: 0 [640/24754(3%)]	Loss: 0.288457

Test set: Average loss: 0.2058, Accuracy: 3885/4157(93%)

INFO:dc_federated.algorithms.fed_avg.fed_avg_worker:Finished training of local model for worker 7aa788c425ae1132672327085065281c79e017dd4821118b314a01f627348530
INFO:dc_federated.algorithms.fed_avg.fed_avg_worker:Sent model update from worker 7aa788c425ae1132672327085065281c79e017dd4821118b314a01f627348530 to the server.
Train Epoch: 0 [1280/24754(5%)]	Loss: 0.591940
```

This means that the first worker has trainer and sent its update to the server and now the server is waiting for the other workers to send their updates to the server. Indeed, in the server terminal you should see additional lines of the following form:

```bash
WARNING:dc_federated.backend.dcf_server:Accepting worker as valid without authentication.
WARNING:dc_federated.backend.dcf_server:Server was likely started without a list of valid public keys from workers.
INFO:dc_federated.backend.dcf_server:Successfully registered worker with public key: No public key was provided when worker was started.
INFO:dc_federated.algorithms.fed_avg.fed_avg_server:Registered worker f3bdfed678321f9d22aa742b579edda9d6ec84f5cb03fe259a5a70ac_unauthenticated
INFO:dc_federated.algorithms.fed_avg.fed_avg_server: Model update received from worker f3bdfed678321f9d22aa742b579edda9d6ec84f5cb03fe259a5a70ac_unauthenticated
```

---

Move to the terminal of the next 2 workers and run

```
(venv_dc_federated)> python mnist_fed_avg_worker.py --server-host-ip 192.124.1.177 --server-port 8080 --digit-class 1
```

and

```
(venv_dc_federated)> python mnist_fed_avg_worker.py --server-host-ip 192.124.1.177 --server-port 8080 --digit-class 2
```

Note that only the `--digit-class` argument changes.

#### Federated Learning In Action

Once you have started the third worker, the `FedAvg` federated learning iterations will start and you should see scrolling output in all four terminals, with the output performance in the server improving over time. For instance on the server side you will see outputs of the form.

```bash
.
.
.
Test set: Average loss: 2.2841, Accuracy: 2019/10000(20%)

INFO:dc_federated.algorithms.fed_avg.fed_avg_server: Model update received from worker 6a324b496d449b6103450f8d5e4a188c5467ab963bc97926baad3c09_unauthenticated
INFO:dc_federated.algorithms.fed_avg.fed_avg_server: Model update received from worker 39c87ce30a29c2e0a27fd5506c3a27c6ca1b18d854abc0a0737e71bc_unauthenticated
INFO:dc_federated.algorithms.fed_avg.fed_avg_server: Model update received from worker f3bdfed678321f9d22aa742b579edda9d6ec84f5cb03fe259a5a70ac_unauthenticated
INFO:dc_federated.algorithms.fed_avg.fed_avg_server: Updating the global model.


Test set: Average loss: 1.6747, Accuracy: 4613/10000(46%)

INFO:dc_federated.algorithms.fed_avg.fed_avg_server: Model update received from worker 6a324b496d449b6103450f8d5e4a188c5467ab963bc97926baad3c09_unauthenticated
INFO:dc_federated.algorithms.fed_avg.fed_avg_server: Model update received from worker 39c87ce30a29c2e0a27fd5506c3a27c6ca1b18d854abc0a0737e71bc_unauthenticated
INFO:dc_federated.algorithms.fed_avg.fed_avg_server: Model update received from worker f3bdfed678321f9d22aa742b579edda9d6ec84f5cb03fe259a5a70ac_unauthenticated
INFO:dc_federated.algorithms.fed_avg.fed_avg_server: Updating the global model.


Test set: Average loss: 1.2767, Accuracy: 6377/10000(64%)

INFO:dc_federated.algorithms.fed_avg.fed_avg_server: Model update received from worker 6a324b496d449b6103450f8d5e4a188c5467ab963bc97926baad3c09_unauthenticated
INFO:dc_federated.algorithms.fed_avg.fed_avg_server: Model update received from worker 39c87ce30a29c2e0a27fd5506c3a27c6ca1b18d854abc0a0737e71bc_unauthenticated
INFO:dc_federated.algorithms.fed_avg.fed_avg_server: Model update received from worker f3bdfed678321f9d22aa742b579edda9d6ec84f5cb03fe259a5a70ac_unauthenticated
INFO:dc_federated.algorithms.fed_avg.fed_avg_server: Updating the global model..
.
.
```

You should see similar output in the worker terminals:

```bash
.
.
.
INFO:dc_federated.algorithms.fed_avg.fed_avg_worker:Finished training of local model for worker 7aa788c425ae1132672327085065281c79e017dd4821118b314a01f627348530
INFO:dc_federated.algorithms.fed_avg.fed_avg_worker:Sent model update from worker 7aa788c425ae1132672327085065281c79e017dd4821118b314a01f627348530 to the server.
Train Epoch: 0 [1280/24754(5%)]	Loss: 0.591940

Test set: Average loss: 0.1943, Accuracy: 3920/4157(94%)

INFO:dc_federated.algorithms.fed_avg.fed_avg_worker:Finished training of local model for worker 7aa788c425ae1132672327085065281c79e017dd4821118b314a01f627348530
INFO:dc_federated.algorithms.fed_avg.fed_avg_worker:Sent model update from worker 7aa788c425ae1132672327085065281c79e017dd4821118b314a01f627348530 to the server.
Train Epoch: 0 [1920/24754(8%)]	Loss: 0.672269

Test set: Average loss: 0.2386, Accuracy: 3772/4157(91%)

INFO:dc_federated.algorithms.fed_avg.fed_avg_worker:Finished training of local model for worker 7aa788c425ae1132672327085065281c79e017dd4821118b314a01f627348530
INFO:dc_federated.algorithms.fed_avg.fed_avg_worker:Sent model update from worker 7aa788c425ae1132672327085065281c79e017dd4821118b314a01f627348530 to the server.
Train Epoch: 0 [2560/24754(10%)]	Loss: 0.233826

Test set: Average loss: 0.2814, Accuracy: 3707/4157(89%)

INFO:dc_federated.algorithms.fed_avg.fed_avg_worker:Finished training of local model for worker 7aa788c425ae1132672327085065281c79e017dd4821118b314a01f627348530
INFO:dc_federated.algorithms.fed_avg.fed_avg_worker:Sent model update from worker 7aa788c425ae1132672327085065281c79e017dd4821118b314a01f627348530 to the server.
Train Epoch: 0 [3200/24754(13%)]	Loss: 0.200743

Test set: Average loss: 0.1206, Accuracy: 4001/4157(96%)
.
.
.
```

Note that for the server, the test set is based on the combined data set of all the digits, whereas for the workers the test set is only its local dataset on the subset of digits. Of course, in a real application, the server-side test results would not be available because the combined data is never available at the server!

You can stop by pressing `Ctrl+C` on the server terminal.

### Worker Authentication

We now show how worker authentication may be incoporated into mnist example, so that only valid workers are allowed to join. To get a general introduction to the `dc_federated` authentication scheme please see `docs/worker_authentication.md` and then come back here.

The first task is to generate the key files for the workers and the servers. You can do that by running:

```bash
(venv_dc_federated)> python mnist_gen_keys.py
```

This script is simply a helper wrapper around the `dc_federated.backend.worker_key_pair` tool. Running this script will generate 9 files - the `mnist_key_list_file.txt` will contain the list of public keys of the three workers, the `mnist_worker_<i>` files will contain the private key and `mnist_worker_<i>.pub` will contain the public key for worker `i`.

Now follow the exact same steps as above, but start the server with:

```bash
(venv_dc_federated)> python mnist_fed_avg_server --key-list-file mnist_key_list_file.txt
```

and start the workers with, for example:

```bash
(venv_dc_federated)> python mnist_fed_avg_worker.py --server-host-ip 192.168.1.155 --server-port 8080 --private-key-file mnist_worker_0_key --digit-class 0
```

You should see the output at the both the server and worker side changed such that all the authentication related warnings are gone, and also the worker id is now the public key of the worker, with no `_unauthenticated` postfix. Try running

```bash
(venv_dc_federated)> python mnist_fed_avg_worker.py --server-host-ip 192.168.1.155 --server-port 8080  --digit-class 0
```

to confirm that no unauthenticated workers are allowed.
