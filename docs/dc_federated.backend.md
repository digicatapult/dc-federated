### Documentation for the classes in dc_federated.backend

The `dc_federated.backend` package for the backend for federated learning consists of two classes:

- `DCFServer`
- `DCFWorker`

These classes abstract away the lower level server/worker communication logic away from the machine learning logic. For a quick example of how to use these, please refer to the example in the package `example_dcf_model` and the corresponding integration test `test_backend_model_integration.py`.

In the following we briefly discuss the requirements for using these two classes.

#### DCFServer

The DCFServer class takes care of the lower level communication logic. A class using it should create an instance of the object with the following 4 callback functions.

- `register_worker_callback`: This function is expected to take the id of a newly registered worker and should contain the application specific logic for dealing with a new worker joining the federated learning pool.

- `unregister_worker_callback`: This function is expected to take the id of a newly unregistered worker and should contain the application specific logic for dealing with a worker leaving the federated learning pool.

- `return_global_model_callback`: This function is expected to return the current global model in some application dependent binary serialized form.

- `query_global_model_status_callback`: This function is expected to return a string giving the application dependent current status of the global model.

- `receive_worker_update_callback`: This function should receive a worker-id and an application dependent binary serialized update from the worker. The server code ensures that the worker-id was previously registered.

See the examples and tests mentioned above for details on how to use this class.

#### DCFWorker

The DCFWorker class takes care of the lower level communication logic with a corresponding running DCFServer object. The constructor expects the host-ip-address of the server and the port of the server. It provides the following functions that may be used to communicate to the server.

- `register_worker`: Registers the DCFWorker instance with the server and returns the worker-id created.

- `get_global_model`: Gets the global model from the server in application dependent binary serialized form.

- `get_global_model_status`: Returns an application dependent string indicating the status of the global model.

- `send_model_update`: Send the model update to the server in an application dependent binary string form.
