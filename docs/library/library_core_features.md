# Library Core Features

As described in the [architecture](architecture.md) page, the core of the library provides the API needed for the communication between the server and the workers. In this section we explain what that means in detail and also list the additional functionalities provided that are necessary to use this package for deployment. 

## Communications 

To concretely implement any federated learning algorithm, we need to describe how the server side logic and the worker side logic are expressed in a running or deployed system. In the context of `dc_federated`, it is expected that the server side logic is will be encapsulated within an http service. The library provides the necessary machinery to start and run the service, and the server side logic is invoked by callback functions. 

For example in the [FedAvg](../examples/using_fed_avg.md) algorithm the server side logic consists of gathering the worker updates - and once sufficient number of worker updates are available, aggregating them and send them back to the workers. This is implemented as http service as follows. The library core provides end-points for workers to send updates to the server, and the end-point invokes a callback supplied by the FedAvg server side implementation (see `dc_federated.algorithms.fed_avg.FedAvgServer.receive_worker_update`). This callback saves the update in memory , and then calculates the new model by aggregating updates if  sufficient number of updates has been received at that point. 

Similarly, it is expected that the the client side logic will be implemented as a user of the above http service. The library core provides a machinery for the worker side that runs a loop that queries the server for the next version of the global model, and once that's available, calls a callback function which implements the client side logic. For instance, the implementation of the client side logic of FedAvg  consists of the callback `dc_federated.algorithms.fed_avg.FedAvgWorker.global_model_version_changed_callback` which is invoked once the library core returns a global model. 

 ## Scalability 
 
The current version of the library supports scaling to large number of workers (consortium level, < 1000). The main scalability concern comes from multiple workers requesting the global model at the same time. This is currently handled via via long-polling on the server http service. Long polling is a standard technique where once a client opens a connection to the server it is kept open for a long time. Within the above architecture this is implemented for the server end-point to getting the new model  
 
In future if greater level of scalability is desired this may be implmented using more advanced techniques such as pushing the models to shared storage. However this should not change the server API and have no impact on the algorithm implementations.
 
 ## Authentication

The library core supports worker authentication  using public key digital signatures using Ed25991 in the lib-sodium library. In general, the server backend can be started in an unsafe mode or safe mode. In the unsafe mode workers are not authenticated and anyone with the location of the server can send an update. When run in the safe mode each message from the worker is to be authenticated via a public key digital signature. The list of  public keys that the server accepts are supplied to the server at startup or added by the server admin during operation of the server (see below). Please see the [worker authentication](worker_authentication.md) for details.    

 ## Security
 
The library core also supports encrypted communication via SSL certifcates (i.e. an https service). Please see the [enabling SSL](enabling_ssl.md) for details.
 
 ## Worker management
 
In addition to the API for the algorithms, the library core also provides end points to allow server admins to list, remove, add and register workers. Please see [worker management](worker_management.md) for details.



 
 