## The Core Principle: 

The core principles behind the Federated Learning library developed at DC are the classic software engineering ideas of layering, modularity and separation of concerns. In the following we describe how these are manifested in the design and implementation of the library.

In particular, an important decision we made was to make the library independent of any specific machine learning or deep learning platform. This ensures that the library has wide applicability and can also be used with any future platforms that are developed or come into prominence. Given, the above, we divided the library design into three layers.

**Backend Layer:** The Backend Layer provides a platform independent API which enables worker nodes and the central server to exchange messages regarding worker/server status and model updates. This is the core of the library and independent of the other two layers. 

**Algorithm Layer:** Specific federated learning algorithms are implemented in the Algorithm Layer, which uses the he API of the Backend Layer to implement the communication protocol necessary for specific algorithm. The main repo comes with an example algorithm, FedAvg, implemented. Implementers of other algorithms will need add their own modules at this layer. The FedAvg implementation can serve as a reference for these new implementations.

**Domain Layer:** Finally, specific domains or applications are implemented at the domain layer which uses the API provided by the Algorithm Layer to implement domain/application specific training and testing logic. 

The figure below illustrates how these layers maps into different classes in the library. Please see the code in `src/algorithms/fed_avg` to see how the library may be used to implement specific federated learning algorithms (in this case `FedAvg`), and see `src/examples/mnist` to see how to apply the algorithm for specific datasets.

