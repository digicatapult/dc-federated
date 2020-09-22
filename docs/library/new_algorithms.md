# Implementing new algorithms

To implement a new algorithm using the `dc_federated` package it is necessary for the ML engineer to implement a set of callback functions implementing the algorithm logic. In the following, we will begin by describing the classes in the library backend in more detail and then desribe how to use them.  

## The dc_federated.backend module

The `dc_federated.backend` package for the backend for federated learning consists of two classes:

- `DCFServer`
- `DCFWorker`

These classes abstract away the lower level server/worker communication logic away from the machine learning logic. For a quick example of how to use these, please refer to the example in the package `example_dcf_model` and the corresponding integration test `test_backend_model_integration.py`. For a detailed example, please see the implementation of the `FedAvg` algorithm in `dc_federated.algorithms.fed_avg`

In the following we briefly discuss the requirements for using these two classes.

### DCFServer

The DCFServer class takes care of the lower level communication logic on the server side and it communicates with the workers using long-polling. This is expected to be used by a class implementing the server side of a federated learning algorithm.  A class using it should create an instance of the object with the following four functions.  

- `register_worker_callback`: This function is expected to take the id of a newly registered worker and should contain the application specific logic for dealing with a new worker joining the federated learning pool.

- `unregister_worker_callback`: This function is expected to take the id of a newly unregistered worker and should contain the application specific logic for dealing with a worker leaving the federated learning pool.

- `return_global_model_callback`: This function is expected to return the current global model in a dictionary with two keys, giving the current global model in an application dependent binary serialized form and an federated learning algorithm dependent model version. See the `DCFServer` doc-string for details.

- `is_global_model_most_recent`: This function is expected return true if the
model version supplied as an argument is the most recent global model. The model versioning logic is left up to implementation of the algorithm.

- `receive_worker_update_callback`: This function should receive a worker-id and an application dependent binary serialized update from the worker. The server code ensures that the worker-id was previously registered.

- `start`: Starts the federated learning server.

### DCFWorker

The DCFWorker class takes care of the lower level communication logic with a corresponding running DCFServer object. This is expected to be used by a class implementing the worker side of a federated learning algorithm. The constructor expects the host-ip-address of the server and the port of the server. A class using it is required to supply the following two functions.

- `global_model_version_changed_callback`: This callback is the algorithm dependent logic for what the worker should do when it receives a model from the server.

- `get_worker_version_of_global_model`: This function returns the latest version of the global model that the worker has seen. This will typically be used by the server to decide which model to return to the worker. The versioning logic and the logic is algorithm dependent.

The `DCFWorker` class also provides the following functions for implementing the worker side logic of the federated learning algorithm.

- `register_worker`: Registers the DCFWorker instance with the server and returns the worker-id created.

- `get_global_model`: Gets the global model from the server in application dependent binary serialized form.

- `send_model_update`: Send the model update to the server in an application dependent binary string form.

- `run`: Implements the main worker side loop that gets a model from the server, and then sends it to the worker side logic of the federated learning algorithm.     

## Using the backend classes

However before we describe those, we  and then start off  the callback functions required by. However before we describe these functions, it will be beneficial to understand the control flow in a typical distributed federated learning  application using the `dc_federated` package.  

As a first step in the control flow, after some necessary initialization,  the server is started by calling the `DCFServer.start()` function which starts an http service. This service provides the following end-points for a remote worker:
  
 - registering *itself* (i.e. the worker) so that the worker can send and receive updates
 - sending a model update  
 - requesting the latest global model
 
 The server also provides end-points for the admin on the server side, but they are not relevant for implementing algorithms.
  
Once the server is up and running, each worker registers itself with the server and starts the main loop by calling `DCFWorker.run()`. This loop essentially waits until a new global model is available and then calls the callback function for when a new global model is created.

A typical federated learning algorithm consists of a server side logic and worker side logic. The functions within `DCFServer` requires the following callbacks to be implemented.

- `register_worker_callback`: This function handles the initliazation logic necessary when a new worker joins the federated learning system.

- `unregister_worker_callback`: This function handles the clean up logic necessary when an existing worker stops taking part in the federated learning system strategy.

- `return_global_model_callback`: This callback returns the global model in a specific format.

- `is_global_model_most_recent`: Given a model version number returns true if the model is the most recent version. This logic is specific to the algorithm implementation.

- `receive_worker_update_callback`: This callback handles the logic that should be done when a new model update is recevied. In particular, this function should handle the **logic of performing model aggregation** when sufficient number of model updates have been received. 

The `DCFWorker` class expects to be supplied the following callback functions;

- `global_model_version_changed_callback`: This callback is executed when the server returns a new global model. So this function should contain the the logic necessary to retrain the local model once a new global model has been received. 

- `get_worker_version_of_global_model`: This is a simple callback that is called to get the version of the global model that was last received by the worker. 

