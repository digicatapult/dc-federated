## The MNIST Federated Learning Example

This document will show you how to run the MNIST federated learning example included in the repo in a fully distributed fashion. In particular, it will show you:
- how to start the server
- how to start three different workers in one or more machines that are connected via the internet
- run the `FedAvg` algorithm to solve the MNIST problem.

### The problem

The MNIST dataset is, at this point, one of the simplest machine vision benchmark problems for deep neural networks. It consists of images of handwritten digits (0 to 9) and the task is to recognize which digit each image corresponds to. In the federated setting we describe below, there are three workers and each worker will only have images for a particular subset of the digits - worker `0` will have images for digits `0 - 3`, worker `1` for `4 - 6` and worker `2` for `7 - 9`. During federated learning the goal will be to create a model at the server that is equally good on the combined dataset but without looking at all three datasets.

### Running the Example

We assume that you are running on Linux or Mac OS. You will need to translate the instructions for Windows, which should be quite similar. Follow the steps below to run the example:

#### Setting up

Clone and install the repo in the machines where you want to deploy the servers and the workers (as described in `docs/getting_started.md`). *You can skip this step if you are running everything on the same machine.* Open a terminal for the server, and one terminal for each of the three workers.

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
(venv_dc_federated)> python mnist_fed_avg_server 
```
You should see an output of the following form:

```bash
INFO:dc_federated.fed_avg.fed_avg_server:Initializing FedAvg server for model class MNISTNet

************
Starting an Federated Average Server at
	server-host-ip: 192.124.1.177 
	server-port: 8080

************
```
But with different `server-host-ip` but the same `server-port`. You may see additional output if you are downloading the mnist dataset. 

#### Start the Workers

Now move to a terminal for a worker and `cd` into the location where the library is installed and, then to the mnist folder (same as above), activate the virtual enironment  and run:
```bash
(venv_dc_federated)> python mnist_fed_avg_worker.py --server-host-ip 192.124.1.177  --server-port 8080 --digit-class 0 
```
The `--digit-class 0` argument means that this worker only only train on digits `0-3`. Using arguments `1` and `2` correspond to training only on digits `4-6` and `7-9` respectively. The `192.124.1.177` and `8080` should be replaced by the values obtained when the server was run. 

Once you run the client, you should see an output of the form
```bash
INFO:dc_federated.fed_avg.fed_avg_worker:Registered with FedAvg Server with worker id 0
Train Epoch: 0 [0/24754(0%)]	Loss: 2.281546
Train Epoch: 0 [640/24754(3%)]	Loss: 0.645392

Test set: Average loss: 0.3791, Accuracy: 3584/4157(86%)

INFO:dc_federated.fed_avg.fed_avg_worker:Finished training of local model for worker 0
INFO:dc_federated.fed_avg.fed_avg_worker:Sent model update from worker 0 to the server.
```
This means that the first worker has trainer and sent its update to the server and now the server is waiting for the other workers to send their updates to the server. Indeed, in the server terminal you should see two additional lines of the following form:
```bash
INFO:dc_federated.fed_avg.fed_avg_server:Registered worker 0
INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 0
```
---

Move to the terminal of the next 2 workers and run 
```
python mnist_fed_avg_worker.py --server-host-ip 192.124.1.177 --server-port 8080 --digit-class 1
```
and 
```
python mnist_fed_avg_worker.py --server-host-ip 192.124.1.177 --server-port 8080 --digit-class 2
```
Note that only the `--digit-class` argument changes. 

#### Federated Learning In Action

Once you have started the third worker, the `FedAvg` federated learning iterations will start and you should see scrolling output in all four terminals, with the output performance in the server improving over time. For instance on the server side you will see outputs of the form.

```bash
.
.
.
Test set: Average loss: 2.2801, Accuracy: 2125/10000(21%)

INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 1
INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 2
INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 0
INFO:dc_federated.fed_avg.fed_avg_server: Updating the global model.


Test set: Average loss: 1.6727, Accuracy: 6048/10000(60%)

INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 1
INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 2
INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 0
INFO:dc_federated.fed_avg.fed_avg_server: Updating the global model.


Test set: Average loss: 1.1575, Accuracy: 7225/10000(72%)

INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 1
INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 2
INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 0
INFO:dc_federated.fed_avg.fed_avg_server: Updating the global model.


Test set: Average loss: 0.8983, Accuracy: 7982/10000(80%)
.
.
.
```
You should see similar output in the worker terminals:
```bash
.
.
.
Test set: Average loss: 2.2801, Accuracy: 2125/10000(21%)

INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 1
INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 2
INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 0
INFO:dc_federated.fed_avg.fed_avg_server: Updating the global model.


Test set: Average loss: 1.6727, Accuracy: 6048/10000(60%)

INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 1
INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 2
INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 0
INFO:dc_federated.fed_avg.fed_avg_server: Updating the global model.


Test set: Average loss: 1.1575, Accuracy: 7225/10000(72%)

INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 1
INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 2
INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 0
INFO:dc_federated.fed_avg.fed_avg_server: Updating the global model.


Test set: Average loss: 0.8983, Accuracy: 7982/10000(80%)

INFO:dc_federated.fed_avg.fed_avg_server: Model update received from worker 1
INFO:dc_federated.fed_avg.fed_avg_server: Model update received 
.
.
.
```

Note that for the server, the test set is based on the combined data set of all the digits, whereas for the workers the test set is only its local dataset on the subset of digits. Of course, in a real application, the server-side test results would not be available because the combined data is never available at the server! 


You can stop by pressing `Ctrl+C` on the server terminal.
