### Documentation for the classes in dc_federated.backend

The `dc_federated.backend` package for the backend for federated learning consists of two classes:

- `DCFServer`
- `DCFWorker`

These classes abstract away the lower level server/worker communication logic away from the machine learning logic. For a quick example of how to use these, please refer to the example in the package `example_dcf_model` and the corresponding integration test `test_backend_model_integration.py`. For a detailed example, please see the implementation of the `FedAvg` algorithm in `dc_federated.algorithms.fed_avg`

In the following we briefly discuss the requirements for using these two classes.

#### DCFServer

The DCFServer class takes care of the lower level communication logic on the server side and it communicates with the workers using long-polling. This is expected to be used by a class implementing the server side of a federated learning algorithm.  A class using it should create an instance of the object with the following four functions.  

- `register_worker_callback`: This function is expected to take the id of a newly registered worker and should contain the application specific logic for dealing with a new worker joining the federated learning pool.

- `unregister_worker_callback`: This function is expected to take the id of a newly unregistered worker and should contain the application specific logic for dealing with a worker leaving the federated learning pool.

- `return_global_model_callback`: This function is expected to return the current global model in a dictionary with two keys, giving the current global model in an application dependent binary serialized form and an federated learning algorithm dependent model version. See the `DCFServer` doc-string for details.

- `is_global_model_most_recent`: This function is expected return true if the
model version supplied as an argument is the most recent global model. The model versioning logic is left up to implementation of the algorithm.

- `receive_worker_update_callback`: This function should receive a worker-id and an application dependent binary serialized update from the worker. The server code ensures that the worker-id was previously registered.

- `start`: Starts the federated learning server.

#### DCFWorker

The DCFWorker class takes care of the lower level communication logic with a corresponding running DCFServer object. This is expected to be used by a class implementing the worker side of a federated learning algorithm. The constructor expects the host-ip-address of the server and the port of the server. A class using it is required to supply the following two functions.

- `global_model_version_changed_callback`: This callback is the algorithm dependent logic for what the worker should do when it receives a model from the server.

- `get_worker_version_of_global_model`: This function returns the latest version of the global model that the worker has seen. This will typically be used by the server to decide which model to return to the worker. The versioning logic and the logic is algorithm dependent.

The `DCFWorker` class also provides the following functions for implementing the worker side logic of the federated learning algorithm.

- `register_worker`: Registers the DCFWorker instance with the server and returns the worker-id created.

- `get_global_model`: Gets the global model from the server in application dependent binary serialized form.

- `send_model_update`: Send the model update to the server in an application dependent binary string form.

- `run`: Implements the main worker side loop that gets a model from the server, and then sends it to the worker side logic of the federated learning algorithm.     
