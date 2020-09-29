# Library core features

As described in the [architecture](architecture.md) page, the core of the library provides the API needed for the communication between the server and the workers. In this section we explain what that means in detail and also list the additional functionalities provided that are necessary to use this package for deployment. 

## Communications 

To concretely implement any federated learning algorithm, we need to describe how the server side logic and the worker side logic are expressed in a running or deployed system. In the context of `dc_federated`, it is expected that the server side logic is will be encapsulated within an http service. The library provides the necessary machinery to start and run the service, and the server side logic is invoked by callback functions. The http service is implemented using the [bottle micro web-framework](https://bottlepy.org/docs/dev/) using the [gunicorn](https://gunicorn.org/) server adapter.

For example in the [FedAvg](../examples/using_fed_avg.md) algorithm the server side logic consists of accepting worker updates, and once sufficient number of worker updates are available, aggregating the updates into a global model and then sending them back to the workers. This is implemented as http service as follows. The library core provides end-points for workers to send updates to the server, and the end-point invokes a callback supplied by the FedAvg server side implementation (see `dc_federated.algorithms.fed_avg.FedAvgServer.receive_worker_update`). This callback saves the update in memory , and then calculates the new model by aggregating updates if  sufficient number of updates has been received at that point. 

Similarly, it is expected that the the client side logic will be implemented as a user of the above http service. The library core provides a machinery for the worker side that runs a loop that queries the server for the next version of the global model, and once that's available, calls a callback function which implements the client side logic. For instance, the implementation of the client side logic of FedAvg  consists of the callback `dc_federated.algorithms.fed_avg.FedAvgWorker.global_model_version_changed_callback` which is invoked once the library core returns a global model. 

## Scalability 
 
The current version of the library supports scaling to large number of workers (consortium level, < 1000). There are two main messages that are exchanged in federated learning - the worker sending an update to the server and the server sending a global model to a worker upon request. Since the communication is one way (worker --> server)  implementing the second half requires the worker to query the server to find out if a new global model is ready. This requires some form of polling strategy on the worker side and is the main barrier to the library being scalable. In the current version of the library this handled via long-polling with the use of pseudo-threads provided by the [gevent library](https://pypi.org/project/gevent/). [Long polling](https://bottlepy.org/docs/dev/async.html) is a standard technique where once a client opens a connection to the server and this is kept open in the server-side in a non-blocking way until the server is ready to it is kept open until it is ready to respond to the request.
 
Greater level of scalability may be implemented using more advanced techniques such as pushing the models to shared storage etc. or using a P2P framework. However this should not change the server API and have no impact on the algorithm implementations.
 
## Authentication

The library core supports worker authentication  using public key digital signatures using Ed25991 in the [salt cryptography library](https://nacl.cr.yp.to/). In general, the server backend can be started in an unsafe mode or safe mode. In the unsafe mode workers are not authenticated and anyone with the location of the server can send an update and query and receive the global model. When run in the safe mode each message from the worker send the update or receive the global model is authenticated via the ED25991 private/public key digital signature. Each worker is associated with its public key, and the list of  public keys that the server accepts as valid workers are either supplied to the server at startup, or added by the server admin during operation of the server (see below). Please see the [worker authentication](worker_authentication.md) for details.    

## Security
 
The library core also supports encrypted communication via SSL certifcates (i.e. an https service). This is implemented in straightforward manner by using the SSL support provided by gunicorn. Please see the [enabling SSL](enabling_ssl.md) for details on how to use it.
 
## Worker management
 
In addition to the API for the algorithms, the library core also provides end points to allow server admins to list, remove, add and register workers. Please see [worker management](worker_management.md) for details.



 
 