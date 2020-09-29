# Using the FedAvg algorithm

The `dc_federated` package contains an implementation of the FedAvg algorithm in `pytorch`. This can be found in `dc_federated.algorithms.fed_avg` and is implemented via two classes: 

- `FedAvgServer` which implements the server side logic of FedAvg by using `dc_federated.backend.DCFServer`. This entails implementing certain callback functions required by `DCFServer`. 
 
- `FedAvgWorker` which implements the worker side logic of FedAvg by using `dc_federated.backend.DCFWorker`. This entails implementing certain callback functions required by `DCFWorker`.

Applying this algorithm to a specific problem (such as mnist) requires implementing the `dc_federated.algorithms.fed_avg.FedAvgModelTrainer` abstract class. This abstract class expects the following functions to be implemented.

- `train`: Implement application specific model training logic. In particular, this should similar or identical to a normal training loop in a non-federated learning algorithm.

- `test`: Implement application specific model training logic. In particular, this should similar or identical to a normal training loop in a non-federated learning algorithm.
 
- `get_model`: This should return a binary serialized version of the model in whatever form chosen by the application user.. 

- `load_model`: This should be able to load a binary serialized version of the model returned by `get_model`.

- `load_model_from_state_dict`: This function should be able to load the model from a pytorch `state_dict`.

As part of implementing the abstract class, the engineer will likely need to implement additional classes. For instance for the MNIST application we have implemented a class for the model, and for generating the dataset.    

Once this class has been implemented, using FedAvg is as simple as creating instances of the `FedAvgServer` and `FedAvgWorker` and, passing instances of this class as an argument. Please see the modules

- `dc_federated.examples.mnist.mnist_fed_model` for an example of implmenation of the `FedAvgModelTrainer` class
- `dc_federated.examples.mnist.mnist_fed_avg_server` for an example of the implementation starting an application side server. 
- `dc_federated.examples.mnist.mnist_fed_avg_worker` for an example of the implementation starting an application side worker.



