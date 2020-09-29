# The dc_federated package

The `dc_federated` package is a [fedeareted learning](https://en.wikipedia.org/wiki/Federated_learning) library that has been developed at Digital Catapult (UK), by the AI/ML team in London. It has been designed to research, experiment with and demo federated learning, and to deploy consortium  scale (< 1000 workers) federated learning applications. Please start at `docs/index.md` for the full documentation.


## Some quick facts about the library

- *Which machine learning platforms are supported?* The core of the library is *platform independent* and any machine learning platforms (tensorflow, pytorch, sklearn) or combination of platforms may be used with it.
  - The examples currently included in the library are based on pytorch.

- *Which Federated Learning algorthims are supported:* The library is designed to be used with any algorithm. The current version includes the FedAvg algorithm implemented with pytorch and instructions on how to implement your own algorithms.

- *Is the library ready for deployment?* The library has been designed to support consortium level (< 1000 workers) deployment.

- *Does the library support worker authentication?* The library supports public key authentication.

- *Does the library support secure communication?* The library supports secure communication via SSL certificates. 

## Additional Information

You can find additional information in the following locations.

  - The Library:
    - [getting started](library/getting_started.md)
    - [library architecture](library/architecture.md)
    - [library core features](library/library_core_features.md)
    - [implementing new algorithms](library/new_algorithms.md)
    - [worker authentication](library/worker_authentication.md)
    - [secure communication](library/enabling_ssl.md)
    - [admin worker management](library/worker_management.md)
  - Examples:
    - [MNIST](examples/mnist.md)
    - [PlantVillage](examples/plantvillage.md)
    - [Using the FedAvg algorithm](examples/using_fed_avg.md)
    - [Multi-Device Test](examples/multi_device_test.md)
