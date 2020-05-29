## The MNIST Federated Learning Example

This document will show you how to run the MNIST federated learning example included in the repo in a fully distributed fashion. In particular, it will show you:
- how to start the server
- how to start two three different workers in one or more machines that are on the same network as the server
- run the FedAvg algorithm to solve the MNIST problem.

### The problem

The MNIST dataset is, at this point, one of the simplest machine vision benchmark problems for deep neural networks. It consists of images of handwritten digits (0 to 9) and the task is to recognize which digit each image corresponds to. In the federated setting we describe below, there are three workers and each worker will only have images for a particular subset of the images - worker 0 will have images for digits 0 - 3, worker 1 for 4 - 6 and worker 2 for 7 - 9. During federated learning the goal will be to create a model at the server that is equally good on the combined dataset but without looking at all three datasets.

### Running the Example

We assume that you are running on Linux or Mac OS. You will need to translate the instructions for Windows, which should be quite similar. Follow the steps below to run the example:

**Step 1.** Clone and install the repo in the machines where you want to deploy the servers and the workers (as described in `docs/getting_started.md`). *You can skip this step if you are running everything on the same machine.*

**Step 2.** Open a terminal for the server, and one terminal for each of the three workers.

**Step 3.** Go to the terminal for the server and activate the environment:

```bash
> source /path/to/venv_dc_federated
```

`cd` to the mnist example folder

```bash
(venv_dc_federated)> cd /path/to/dc_federated/src/examples/mnist
```

now start the server:


```bash
(venv_dc_federated)> python mnist_fed_avg_server 
```

You should see an output of the following form:

```bash
(venv_dc_federated)> python mnist_fed_avg_server 
```

But with different server ip address but the same - you can specifuy the same thing, 

With each of the worker do the following run the command.


You should see runnning stuff as followins






