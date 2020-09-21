# Implementing new algorithms

To implement a new algorithm using the `dc_federated` package it is necessary for the ML engineer to implement all the callback functions required by. However before we describe these functions, it will be beneficial to understand the control flow in a typical distributed federated learning  application using the `dc_federated` package.  

As a first step in the control flow, after some necessary initializaion,  the server is started by calling the `DCFServer.start()` function which starts an http service. This service provides the following end-points for a remote worker:
  
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

