# Stress Test Tool  

The library also comes with a stress test tool to help simulate federated learning session over a network without actually doing any training to check how robust your network setup is. This test tool can be found in `src/dc_federated/stress_test`. The stress test consists of setting up a server, and then repeatedly sending model updates from a number of different workers from a collection of nodes. Each node consist of `1/k th` fraction of the total number of workers. Updates sent results in long poll waiting for the global model update. A 'global model update' happens after all the workers have sent the updates, at which point the global model is sent back to each of the workers that had sent a request. By default, the 'global model' is just a short string, but can be changed to a randomly initilized `MobileNetV2` model by setting boolean flag. The cycle repeats for a fixed number of runs. 

The above can be run with a variable number of runs and workers. In the following we show a instance of this for a 1000 workers sending update from 4 different nodes. The process we outline is manual and a bit tedious - for more elaborate test setup, you would likely want an orchestration tool like [ansible](https://www.ansible.com/) to run your tests in a repeatable way. However this is out of the scope of this document and do not discuss this further. 


## Setting Up

To run the the test as describe five machines will be requried, one for the server, and the remaining four to run the workers (with 250 workers running in separate Greenlet threads). Set up the server machine by following the instructions in the `Scalability` section in [deployment_notes](deployment_notes.md). For the worker side, follow in the section `Scalability >  Open Files` section. Once you have completed the setup,  install the library in both the server machine and worker machines as described in [geting started](../library/getting_started.md).

## Generating Keys

We will run the test in [safe mode](../library/worker_authentication.md) and so the first step will be to generate the private/public keys for the workers and distribute them to each of the five machines. To generate the keys `cd` into `src/dc_federated/stress_test` (in any machine - but probably best in your main development machine) and run  
```bash
python stress_keys_gen.py --num-workers 1000
```
This will create a folder `src/dc_federated/stress_test/stress_keys_folder` with a 1000 private and public keys + a `stress_keys_list.txt` file containing a list of all the public keys. 

Copy the folder `stress_keys_folder`  and its contents to `src/dc_federated/stress_test` in each of the machines. Technically, the server machine only needs the `stress_keys_list.txt` file - but each of the worker machines needs all the files. 

## Running the Server

Log in to the server machine. First make sure that your reverse proxy is running and is accessible over the web by going to `http://<server-host-ip>`. You should see a nginx `503` error - this is because the upstream federated learning server has not yet been started. If you see unable to find host or similar error, this means that you nginx installation was not successful - you need to fix that. Also make sure that the nginx proxy is configured to connect to the upstream federated learning server at `http://127.0.0.1:5000`. 

Once the reverse-proxy is running, activate the virtual environment, go to `src/dc_federated/stress_test` and run
```bash
python stress_server.py
```
This will start the server and ready to accept worker requests. In particular you should see something like the following output.
```
<insert output>
```  

The `stress_server.py` script supports the following option:

- `--global-model-real`: if this boolean flag is set, then the server  will send randomly initialized  `MobileNetV2` model as a global model update instead of just a string. 


## Running the Workers

Log in to a worker machine, activate the virtual environment and `cd` into `src/dc_federated/stress_test`. Now run the following command:

```bash
> python stress_worker.py --server-host-ip <server-host-name> --server-port <port> --chunk "<n> of 4"
```
the `server-host-name` parameter is the name or ip address of the machine on which server is runnnig, the port will typically be 80, where nginx listens for incoming connections and the chunk argument `<n>` will be either, 1, 2, 3 and 4 in each the 4 different worker machines. Running with `--chunk "<n>of 4"` will run the node with workers with keys `(n-1) * 250 - n*250` - so for instance `--chunk "2 of 4"` will run the node with workers with keys `250` to `500`. When you run this, you should see output of the following form:
```bash
<insert output>
```

The `stress_worker.py` script supports two additional options:

- `--num_runs <k>`: this defines the number of rounds of global model updates to send.

- `--global-model-real`: if this boolean flag is set, then the workers will send randomly initialized  `MobileNetV2` model instead of just a string as is done by default. 





