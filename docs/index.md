# The dc_federated package

The `dc_federated` package is a [fedeareted learning](https://en.wikipedia.org/wiki/Federated_learning) library that has been developed at Digital Catapult (UK), by the AI/ML team and the engineering team in London. It has been designed to research, experiment with and demo federated learning, and to deploy consortium  scale (< 1000 workers) federated learning applications. Please start at `docs/index.md` for the full documentation.


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
    - [Getting Started](library/getting_started.md)
    - [Library Architecture](library/architecture.md)
    - [Library Core Features](library/library_core_features.md)
    - [Implementing New Algorithms](library/new_algorithms.md)
    - [Worker Authentication](library/worker_authentication.md)
    - [Secure Communication](library/enabling_ssl.md)
    - [Admin Worker Management](library/worker_management.md)
  - Examples:
    - [MNIST Dataset](examples/mnist.md)
    - [PlantVillage Dataset](examples/plantvillage.md)
    - [Using the FedAvg algorithm](examples/using_fed_avg.md)
    - [Multi-Device Test](examples/multi_device_test.md)
  - Deployment:
    - [Deployment Notes](deployment/deployment_notes.md)
    - [Stress Test](deployment/stress_test.md)
  
