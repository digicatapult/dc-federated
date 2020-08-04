### Running the Multi-Device Test

To run the multi-device test, to check that the backend works when the server and worker are running in different devices, please follow the following steps.

First run the server in the server host device by running the following from the `example_dcf_model` folder:
```bash
> python federated_global_model.py
```
You should see something the following output (with the `server-host-ip` being different, as as the ip address on the second to last line):
```
************
Starting an Example Global Model Server to test backend + model integration
in a distributed setting. Once the server has started, run the corresponding
local-model runner(s) in other devices. Run `python federated_local_model.py -h' for
instructions and use the server parameters provided below.

Server starting at 
	server-host-ip: 10.132.8.109 
	server-port: 8080

************

Bottle v0.12.18 server starting up (using WSGIRefServer())...
Listening on http://10.132.8.109:8080/
Hit Ctrl-C to quit.
```

Now switch to a client device, where this package has been installed, and run the following, again from the example_dcf_model` folder:
```bash
python federated_local_model.py --server-host-ip 10.132.8.109 --server-port 8080
```
Note how the arguments for the local model are run using the parameters returned by running the server. 

Once this is run, on the server side you should see the following output (with the timestamps adjusted appropriately):
```bash
INFO:dc_federated.example_dcf_model.global_model:Example Global Model: Registering worker 0
10.132.8.109 - - [12/Mar/2020 10:36:49] "GET /register_worker HTTP/1.1" 200 1
INFO:dc_federated.example_dcf_model.global_model:Model update received from worker 0
INFO:dc_federated.example_dcf_model.global_model:ExampleModelClass(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
10.132.8.109 - - [12/Mar/2020 10:36:49] "POST /receive_worker_update HTTP/1.1" 200 28
INFO:dc_federated.example_dcf_model.global_model:Example Global Model: returning global model status
10.132.8.109 - - [12/Mar/2020 10:36:49] "GET /query_global_model_status HTTP/1.1" 200 19
INFO:dc_federated.example_dcf_model.global_model:Example Global Model: returning global model
10.132.8.109 - - [12/Mar/2020 10:36:49] "GET /return_global_model HTTP/1.1" 200 264720 
```
 
 and on the client side, you should see the following output.
 
 ```bash
INFO:dc_federated.example_dcf_model.local_model:I got the global model!! -- transforming...
INFO:dc_federated.example_dcf_model.local_model:ExampleModelClass(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)

```

If you see this, then the backend is working in a multi-device federated learning scenario.
