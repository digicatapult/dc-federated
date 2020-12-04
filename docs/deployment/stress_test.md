# Stress Test Tool  

The library also comes with a stress test tool to help simulate federated learning session over a network without actually doing any training to check how robust your network setup is. This test tool can be found in `src/dc_federated/stress_test`. The stress test consists of setting up a server, and then repeatedly sending model updates from a number of different workers from a collection of nodes. Each node consist of `1/k th` fraction of the total number of workers. The basic iteration is as follows:

- Workers send the updates to the model.
- Workers then requests a long poll waiting for the global model update. 
- A 'global model update' happens after all the workers have sent the updates. 
- The global model is sent back to each of the workers that had sent a request. 

By default, the 'global model' and the worker-models are just short strings, but can be changed to a randomly initialised `MobileNetV2` model by setting boolean flag . The cycle repeats for a fixed number of runs specified by the command line. 

The above can be run with a variable number of workers and runs. In the following we show a instance of this for a 1000 workers sending update from 4 different AWS EC2 nodes to a single server also in an AWS EC2 node. The process we outline is manual and a bit tedious - for more elaborate test setup, you would likely want an orchestration tool like [ansible](https://www.ansible.com/) to run your tests in a repeatable way. However this is out of the scope of this document and do not discuss this further. 


The running time of a single iteration of the test depends on your network connection and whether you use real models (each of which are around 13Mb). If you do not use real models, it should be done in < 30 seconds. With real models, it can take 10 minutes to an hour depending on how fast your network connections are. The reason for this is that, with 1000 workers and real models, each of which are 13Mb, you are exchanging ~2.6GB of models between different workers and servers. 


## Setting Up

To run the the test as describe five machines will be required, one for the server, and the remaining four to run the workers (each machine running 250 workers in separate Greenlet threads). Set up the server machine by following the instructions in the `Scalability` section in [Deployment Notes](deployment_notes.md). For the worker side, follow in the section `Scalability >  Open Files` section. Once you have completed the setup,  install the library in both the server machine and worker machines as described in [geting started](../library/getting_started.md).

## Generating Keys

We will run the test in [safe mode](../library/worker_authentication.md) and so the first step will be to generate the private/public keys for the workers and distribute them to each of the five machines. To generate the keys `cd` into `src/dc_federated/stress_test` (in any machine - but probably best in your main development machine) and run  
```bash
python stress_keys_gen.py --num-workers 1000
```
This will create a folder `src/dc_federated/stress_test/stress_keys_folder` with a 1000 private and public keys + a `stress_keys_list.txt` file containing a list of all the public keys. 

Copy the folder `stress_keys_folder`  and its contents to `src/dc_federated/stress_test` in each of the machines. Technically, the server machine only needs the `stress_keys_list.txt` file - but each of the worker machines needs all the files. 

## Running the Server

Log in to the server machine. First make sure that your reverse proxy is running and is accessible over the web by going to `http://<server-host-ip>`. You should see a nginx `503` error - this is because the upstream federated learning server has not yet been started. If you see "Unable to find host" or similar error, this means that you nginx installation was not successful - you need to fix that. Also make sure that the nginx proxy is configured to connect to the upstream federated learning server at `http://127.0.0.1:5000`. 

Once the reverse-proxy is running, activate the virtual environment, go to `src/dc_federated/stress_test` and run
```bash
python stress_server.py
```
This will start the server and will be ready to accept worker requests. In particular you should see something like the following output.
```
2020-11-30 18:31:54 INFO     Server started is running in **** SAFE MODE **** - workers will need to use public key authentication.
2020-11-30 18:31:54 INFO     Successfully added worker with public key (short) d74f7cfc
2020-11-30 18:31:54 INFO     Successfully added worker with public key (short) 9aca0c32
2020-11-30 18:31:54 INFO     Successfully added worker with public key (short) 76274d19
2020-11-30 18:31:54 INFO     Successfully added worker with public key (short) ba1df73b
2020-11-30 18:31:54 INFO     Successfully added worker with public key (short) 122d16e5
2020-11-30 18:31:54 INFO     Successfully added worker with public key (short) 7e9883bc
2020-11-30 18:31:54 INFO     Successfully added worker with public key (short) e954c930
2020-11-30 18:31:54 INFO     Successfully added worker with public key (short) f3c89f37
2020-11-30 18:31:54 INFO     Successfully added worker with public key (short) 2effb2d1
2020-11-30 18:31:54 INFO     Successfully added worker with public key (short) 406f49f8
.
.
.
[2020-11-30 18:31:54 +0000] [8481] [INFO] Starting gunicorn 20.0.4
[2020-11-30 18:31:54 +0000] [8481] [INFO] Listening at: http://127.0.0.1:5000 (8481)
[2020-11-30 18:31:54 +0000] [8481] [INFO] Using worker: gevent
[2020-11-30 18:31:54 +0000] [8484] [INFO] Booting worker with pid: 8484

```  

The `stress_server.py` script supports the following option:

- `--global-model-real`: if this boolean flag is set, then the server  will send randomly initialized  `MobileNetV2` model as a global model update instead of just a string. 


## Running the Workers

Log in to a worker machine, activate the virtual environment and `cd` into `src/dc_federated/stress_test`. Now run the following command:

```bash
> python stress_worker.py --server-host-ip <server-host-name> --server-port <port> --chunk "<n> of 4"
```
The command parameters mean the following:

- `server-host-name`: name or ip address of the machine on which server is running (e.g. for an AWS EC2 instance, it may be something like `ec2-11-111-11-1111.eu-west-2.compute.amazonaws.com`,
- `--server-port`: this will typically be 80, where nginx listens for incoming connections
 - `--chunk`: in this argument, `<n>` will be either, 1, 2, 3 and 4 in each the 4 different worker machines. Running with `--chunk "<n>of 4"` will run the node with workers with keys `(n-1) * 250 - n*250` - so for instance `--chunk "2 of 4"` will run the node with workers with keys `250` to `500`
 
 When you run this, you should see output of the following form:
```bash
n = 1 , k = 1
2020-12-01 14:18:55 WARNING  Security alert: https is not enabled!
.
.
.
2020-12-01 14:18:55 INFO     Registering 0 th worker
2020-12-01 14:18:55 INFO     Registering public key (short) 9aca0c32 with server...
2020-12-01 14:18:55 INFO     Registration for public key (short) 9aca0c32 done.
.
.
.
Requesting global model for 9aca0c3201a9c9bdde4bf09d99926803624d4aea614b1e5226983f44247a25fb (no. 0) 
2020-12-01 14:18:55 INFO     Received global model for worker 9aca0c32
.
.
.
2020-12-01 14:18:55 INFO     ********************** STARTING RUN 1:
2020-12-01 14:19:00 INFO     Response from server sending model update: b'Update received for worker 9aca0c32.'
2020-12-01 14:19:00 INFO     Spawning for worker 0
2020-12-01 14:19:00 INFO     Starting long poll for 9aca0c3201a9c9bdde4bf09d99926803624d4aea614b1e5226983f44247a25fb (no. 0)
.
.
.
020-12-01 14:19:10 INFO     Received global model for worker ba1df73b
Received global model for ba1df73bef23823ae42716d741f1b76913456f1c339e7db9111c1a708ba7f7c4
2020-12-01 14:19:10 INFO     Long poll for ba1df73bef23823ae42716d741f1b76913456f1c339e7db9111c1a708ba7f7c4 finished
020-12-01 14:19:10 INFO     Received global model for worker 7e9883bc
.
.
.
2020-12-01 14:19:12 INFO     250 workers have received the global model update - need to get to 250...
```
The `stress_worker.py` script supports two additional options:

- `--num_runs <k>`: this defines the number of rounds of global model updates to send.

- `--worker-model-real`: if this boolean flag is set, then the workers will send randomly initialized  `MobileNetV2` model instead of just a string as is done by default. 







