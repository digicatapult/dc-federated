# Implementing new algorithms

To implement a new algorithm using the `dc_federated` package it is necessary for the ML engineer to implement a set of callback functions implementing the algorithm logic. The callback functions are called by the core of the library, which is contained in the`dc_federated.backend` module (everything else outside of it are reference implementaitons and the users of the library are free to decide for themselves how they want to use the facilities provided). This module consists of two public classes:

- `DCFServer`
- `DCFWorker`

These classes abstract away implementation details of various lower level services (see [library core features](library_core_features.md)) from the machine learning logic. To describe how these classes use the callback funcitions, we start by describing the control flow in a typical application using the `dc_federated` library.

## Control flow in a `dc_federated` application

The control flow can be understood in terms of the server side and worker side control flow. 

### Server side control flow
As can be seen in the [MNIST example](../examples/mnist.md), the typical starting point in an application is the server script for the specific application. This script will, after some application specific initialization start the algorithm server ( `dc_federated.algorithsms.fed_avg.FedAvgServer` in the case of MNIST). This algorithm server will perform initialization and create a `DCFServer` object and start it by calling `DCFServer.start()` (the output seen on the console when the MNIST example server is started are as a result of calling this function). This will start the http service that the worker will use to communicate with the server. This service provides the following end-points for a remote worker:
  
 - Registering *itself* (i.e. the worker) so that the worker can send and receive updates via the endpoint `/register_worker`.
 - Sending a model update via the end-point `/receive_worker_update`
 - Requesting the latest global model via the end-point `/return_global_model`. 
   - When the server is running in the safe mode, this is preceeded by a call to the `/challenge_phrase` route to get a challenge phrase necessary for authentication.

The above represents the sum total of the communication protocol supported between the worker and the server.These are all internal to the backend and not referred to in the algorithm implementation. The server also provides end-points for the admin on the server side, but they are not relevant for implementing algorithms.
  
### Worker side control flow

On the worker side, as can again be seen from the [MNIST example](../examples/mnist.md), the typical starting point in an application is the worker for that application. This worker will, after some application specific initialization, start the algorithm worker 
 (the `dc_federated.algorithsms.fed_avg.FedAvgWorker.start()` in the case of MNIST), which in turn will start the `DCFWorker.run()` loop. In this loop the `DCFWorker` will check if a global model is ready and if so call the appropriate callback within the algorithm worker side logic. This worker side logic will typically perform the necessary updates and send the model to the server, using the `DCFWorker` instance, and then go back to the loop.


## Implementing the algorithm

Given the above context, a federated learning algorithm can be implemented using `dc_federated` as follows (please following along to the implementation in `dc_federated.algorithms.fed_avg` for a concrete example).  A typical federated learning algorithm consists of a server side logic and worker side logic. The functions within `DCFServer` requires the following callbacks to be implemented to implement th server side logic.

- `register_worker_callback`: This function handles the initialization logic necessary when a new worker joins the federated learning system. It is expected to take the id of a newly registered worker and should contain the algorithm specific logic for dealing with a new worker joining the federated learning pool. The algorithm implementation is free to expose the application to worker registration events via its application callbacks that are invoked when a worker registers.

- `unregister_worker_callback`: This function handles the clean up logic necessary when an existing worker stops taking part in the federated learning system strategy.

- `return_global_model_callback`: This function is expected to return the current global model in a dictionary with two keys, giving the current global model in an application dependent binary serialized form and an federated learning algorithm dependent model version. See the `DCFServer` doc-string for details.

- `is_global_model_most_recent`: Given a model version number returns true if the model is the most recent version. The versioning logic is specific to the algorithm implementation.

- `receive_worker_update_callback`: This callback handles the logic that should be done when a new model update is recevied. In particular, this function should handle the **logic of performing model aggregation** when sufficient number of model updates have been received. 

The `DCFWorker` class expects to be supplied the following callback functions;

- `global_model_version_changed_callback`: This callback is executed when the server returns a new global model. So this function should contain the the logic necessary to
   - incorporate the new global model into the local model.
   - retrain the local model once a new global model has been received. 

- `get_worker_version_of_global_model`: This is a simple callback that is called to get the version of the global model that was last received by the worker. 

