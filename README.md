# The dc_federated Package

This repo contains the python [fedeareted learning](https://en.wikipedia.org/wiki/Federated_learning) library that was developed at Digital Catapult (UK), by the AI/ML team in London. It was designed to research, experiment with and demo federated learning on a small scale to DC clients - but was also designed in a way so that it can be extended to a full scale library. 

## Some quick facts about the library

- *Which machine learning platforms are supported?* The core of the library is *platform independent* and any machine learning platforms (tensorflow, pytorch, sklearn) or combination of platforms may be used with it.
  - The examples currently included in the library are based on pytorch.

- *Which Federated Learning algorthims are supported:* The library is designed to be used with any algorithm. The current version includes the FedAvg algorithm implemented with pytorch.

- *Is the library ready for deployment?* No - the library is currently meant only for small scale, but truly distributed, experimentation. This means you can have, for instance, 10 workers running in nodes distributed across different machines and experiment with federeated learning algorithms on them. We very much welcome contributions to scale up this library.

- *Does the library support worker authentication?* The library currently does not support any worker authentication. But we very much welcome contributions to improve this situation.

## Additional Information

For additional information, please look at the following documents.
- For installation instructions, please see `docs/getting_started.md`
- To run the basic federated MNIST example, please see `docs/mnist.md`
- To run the more complex PlantVillage dataset examples, please see `docs/plant_village.md`
- For descripition of architecture and how to use the library to implement your own algorithms and problems, please look at `docs/architecture.md`. 
