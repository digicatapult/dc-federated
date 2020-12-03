# The MNIST federated learning Example

This document will show you how to run the MNIST federated learning example included in the repo in a fully distributed fashion. In particular, it will show you:

- how to start the server
- how to start three different workers in one or more machines that are connected via the internet
- run the `FedAvg` algorithm to solve the MNIST problem
- use the authentication facility in `dc_federated` to authenticate workers.

## The problem

The MNIST dataset is, at this point, one of the simplest machine vision benchmark problems for deep neural networks. It consists of images of handwritten digits (0 to 9) and the task is to recognize which digit each image corresponds to. In the federated setting we describe below, there are three workers and each worker will only have images for a particular subset of the digits - worker `0` will have images for digits `0 - 3`, worker `1` for `4 - 6` and worker `2` for `7 - 9`. During federated learning the goal will be to create a model at the server that is equally good on the combined dataset but without looking at all three datasets.

## Running the example

We assume that you are running on Linux or Mac OS. You will need to translate the instructions for Windows, which should be quite similar. Follow the steps below to run the example:

### Setting up

Clone and install the repo in the machines where you want to deploy the servers and the workers (as described in `docs/getting_started.md`). _You can skip this step if you are running everything on the same machine._ Open a terminal for the server, and one terminal for each of the three workers.

### Start The Server

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

By default the server will use http for transport.
Alternatively the server can be use https:

```sh
(venv_dc_federated)> python mnist_fed_avg_server.py \
  --server-host-ip localhost \
  --ssl-certfile localhost.crt \
  --ssl-keyfile localhoat.key \
  --ssl-enabled
```

Please refer to [the ssl document for more details](../library/enabling_ssl.md)

You should see an output of the following form:

```bash
2020-12-02 18:31:53 INFO     Initializing FedAvg server for model class MNISTNet
2020-12-02 18:31:53 INFO     Server started is running in **** UNSAFE MODE **** - all workers will be accepted.
[2020-12-02 18:31:53 +0000] [9547] [INFO] Starting gunicorn 20.0.4
[2020-12-02 18:31:53 +0000] [9547] [INFO] Listening at: http://192.124.1.177:8080 (9547)
[2020-12-02 18:31:53 +0000] [9547] [INFO] Using worker: gevent
[2020-12-02 18:31:53 +0000] [9550] [INFO] Booting worker with pid: 9550
```

But with different `server-host-ip` but the same `server-port`. You may see additional output if you are downloading the mnist dataset. Note the warning that the server is starting in `unsafe mode` - this is because that the worker authentication is not being used. See below for how to run this example with the workers being authenticated.

### Start the workers

Now move to a terminal for a worker and `cd` into the location where the library is installed and, then to the mnist folder (same as above), activate the virtual enironment and run:

```bash
(venv_dc_federated)> python mnist_fed_avg_worker.py \
	--server-host-ip 192.124.1.177 \
	--server-port 8080 \
	--digit-class 0
```

Or, if the server use https:

```bash
(venv_dc_federated)> REQUESTS_CA_BUNDLE=localhost.crt python mnist_fed_avg_worker.py \
    --server-protocol https \
    --server-host-ip localhost \
    --server-port 8080 \
    --digit-class 0
```

Please refer to [the ssl document for more details](../library/enabling_ssl.md)

The `--digit-class 0` argument means that this worker only only train on digits `0-3`. Using arguments `1` and `2` correspond to training only on digits `4-6` and `7-9` respectively. The `192.124.1.177` and `8080` should be replaced by the values obtained when the server was run.
The `--digit-class 0` argument means that this worker only only train on digits `0-3`. Using arguments `1` and `2` correspond to training only on digits `4-6` and `7-9` respectively. The `192.124.1.177` and `8080` should be replaced by the values obtained when the server was run.

Once you run the worker, you should see an output of the form

```bash
2020-12-02 19:51:47 WARNING  Security alert: https is not enabled!
2020-12-02 19:51:47 WARNING  No public key file provided - server side authentication will not succeed.
2020-12-02 19:51:47 WARNING  Unable to sign message - no private key file provided.
2020-12-02 19:51:47 INFO     Registering public key (short) No publi with server...
2020-12-02 19:51:47 INFO     Registration for public key (short) No publi done.
2020-12-02 19:51:47 INFO     Registered with FedAvg Server with worker id 844b8060
Train Epoch: 0 [0/17181(0%)]	Loss: 2.307410
Train Epoch: 0 [640/17181(4%)]	Loss: 0.235539

Test set: Average loss: 0.1330, Accuracy: 2720/2832(96%)

2020-12-02 19:51:51 INFO     Finished training of local model for worker 844b8060
2020-12-02 19:51:51 WARNING  Unable to sign message - no private key file provided.
2020-12-02 19:51:51 INFO     Sent model update from worker 844b8060 to the server.
```

This means that the first worker has trained and sent its update to the server and now the server is waiting for the other workers to send their updates to the server. Indeed, in the server terminal you should see additional lines of the following form:

```bash
2020-12-02 19:51:47 WARNING  Accepting worker as valid without authentication.
2020-12-02 19:51:47 INFO     Successfully added worker with public key (short) No publi
2020-12-02 19:51:47 INFO     Set registration status of worker 844b8060 from False to True.
2020-12-02 19:51:47 INFO     Registered worker 844b8060
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

### Federated Learning In Action

Once you have started the third worker, the `FedAvg` federated learning iterations will start and you should see scrolling output in all four terminals, with the output performance in the server improving over time. For instance on the server side you will see outputs of the form.

```bash
.
.
.
2020-12-02 19:52:12 INFO     Received model update from worker b3f286d0.
2020-12-02 19:52:12 INFO     Model update from worker b3f286d0 accepted.
2020-12-02 19:52:12 INFO     Updating the global model.


Test set: Average loss: 2.2816, Accuracy: 1677/10000(17%)

2020-12-02 19:52:19 INFO     Notified global model version changed to d8278edd.
2020-12-02 19:52:19 INFO     Notified global model version changed to 844b8060.
2020-12-02 19:52:19 INFO     Received request for global model version change notification from b3f286d0.
2020-12-02 19:52:19 INFO     Notified global model version changed to b3f286d0.
2020-12-02 19:52:19 INFO     Returned global model to 844b8060.
2020-12-02 19:52:19 INFO     Returned global model to d8278edd.
2020-12-02 19:52:19 INFO     Returned global model to b3f286d0.
2020-12-02 19:52:25 WARNING  Accepting worker as valid without authentication.
2020-12-02 19:52:25 INFO     Received model update from worker 844b8060.
2020-12-02 19:52:25 INFO     Model update from worker 844b8060 accepted.
2020-12-02 19:52:25 WARNING  Accepting worker as valid without authentication.
2020-12-02 19:52:25 INFO     Received model update from worker d8278edd.
2020-12-02 19:52:25 INFO     Model update from worker d8278edd accepted.
2020-12-02 19:52:25 INFO     Received request for global model version change notification from 844b8060.
2020-12-02 19:52:25 INFO     Received request for global model version change notification from d8278edd.
2020-12-02 19:52:26 WARNING  Accepting worker as valid without authentication.
2020-12-02 19:52:26 INFO     Received model update from worker b3f286d0.
2020-12-02 19:52:26 INFO     Model update from worker b3f286d0 accepted.
2020-12-02 19:52:26 INFO     Updating the global model.


Test set: Average loss: 1.6433, Accuracy: 6209/10000(62%)

2020-12-02 19:52:34 INFO     Notified global model version changed to 844b8060.
2020-12-02 19:52:34 INFO     Notified global model version changed to d8278edd.
2020-12-02 19:52:34 INFO     Received request for global model version change notification from b3f286d0.
2020-12-02 19:52:34 INFO     Notified global model version changed to b3f286d0.
2020-12-02 19:52:34 INFO     Returned global model to 844b8060.
2020-12-02 19:52:34 INFO     Returned global model to d8278edd.
2020-12-02 19:52:34 INFO     Returned global model to b3f286d0..
.
```

You should see similar output in the worker terminals:

```bash
.
.
.
2020-12-02 19:52:20 INFO     Received global model for worker 844b8060
Train Epoch: 0 [1280/17181(7%)]	Loss: 0.900819

Test set: Average loss: 0.1904, Accuracy: 2671/2832(94%)

2020-12-02 19:52:25 INFO     Finished training of local model for worker 844b8060
2020-12-02 19:52:25 WARNING  Unable to sign message - no private key file provided.
2020-12-02 19:52:25 INFO     Sent model update from worker 844b8060 to the server.
2020-12-02 19:52:25 WARNING  Unable to sign message - no private key file provided.
2020-12-02 19:52:34 WARNING  Unable to sign message - no private key file provided.
2020-12-02 19:52:35 INFO     Received global model for worker 844b8060
Train Epoch: 0 [1920/17181(11%)]	Loss: 0.321993

Test set: Average loss: 0.1811, Accuracy: 2659/2832(94%)

2020-12-02 19:52:40 INFO     Finished training of local model for worker 844b8060
.
.
.
```

Note that for the server, the test set is based on the combined data set of all the digits, whereas for the workers the test set is only its local dataset on the subset of digits. Of course, in a real application, the server-side test results would not be available because the combined data is never available at the server!

You can stop by pressing `Ctrl+C` on the server terminal.

## Docker


Run `docker-compose -f docker_compose_mnist.yml up`

This will build the relevant images and bring up the example. This example has been tested and works using only 8GB of host memory.

## Worker Authentication

We now show how worker authentication may be incoporated into mnist example, so that only valid workers are allowed to join. To get a general introduction to the `dc_federated` authentication scheme please see `docs/worker_authentication.md` and then come back here.

The first task is to generate the key files for the workers and the servers. You can do that by running:

```bash
(venv_dc_federated)> python mnist_gen_keys.py
```

This script is simply a helper wrapper around the `dc_federated.backend.worker_key_pair` tool. Running this script will generate 9 files - the `mnist_key_list_file.txt` will contain the list of public keys of the three workers, the `mnist_worker_<i>` files will contain the private key and `mnist_worker_<i>.pub` will contain the public key for worker `i`.

Now follow the exact same steps as above, but start the server with:

```bash
(venv_dc_federated)> python mnist_fed_avg_server.py --key-list-file mnist_key_list_file.txt
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
