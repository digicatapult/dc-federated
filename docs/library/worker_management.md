# Admin Worker Management

The core library provides some end-points for the server-side admin to manage workers. To be able to use the end-point, the server shell environment has to include the admin credentials via the following two environment variables.
```
DCF_SERVER_ADMIN_USERNAME 
DCF_SERVER_ADMIN_PASSWORD
```
For instance, within linux like systems using bash, this can be done by  
```
> export DCF_SERVER_ADMIN_USERNAME=dcf_server_admin
> export DCF_SERVER_ADMIN_PASSWORD=str0ng_pass_word
```
Once the server is running, it provides end-points for the following functionalities.

 
## Adding a new worker

A new worker can be added by sending a POST request to the `workers` end-point with a json file containing the public-key of the new worker and the desired registration status. For example, the json file will be:
```json
{
  "public_key_str": "46e046cdc32ccae16accc44344703c09c2de19b6c4d9efeeeb28575efd47e7d2",
  "registered": true
}
```
And the corresponding curl command would be:
```bash
curl --user dcf_server_admin:str0ng_pass_word \
        --header "Content-Type: application/json" \
	--request POST http://188.121.1.122:8080/workers \
        --data "@worker_2_info.json"
```
This will return a json containing the  `worker-id` and its registration status. 
```json
{
  "worker_id": "46e046cdc32ccae16accc44344703c09c2de19b6c4d9efeeeb28575efd47e7d2",
  "registered": true
}
```
The id should be used to identify the worker in listings of workers, change its status or delete it. In the current version of the library this id is the same as its public key.  Please note that adding a worker via the admin API is not meaningful when the server is running in the unsafe mode because a worker is assigned a unique-id every time it registers. 

## Listing workers

The current list of workers that have been added and their registration status may be obtained by sending a GET request to the workers end-point. This looks as follows when using curl: 
```bash
curl --user dcf_server_admin:str0ng_pass_word \
        --header "Content-Type: application/json" \
	--request GET http://192.168.1.155:8080/workers 
```
This will return a json list consisting of the id of the worker (which is currently the same as the public-key for the worker) and its registration status.
```json
[{
  "worker_id": "46e046cdc32ccae16accc44344703c09c2de19b6c4d9efeeeb28575efd47e7d2",
  "registered": true
},
{
  "worker_id": "cd549b4f88ec1673a1177c973017fb2a85c1764ce1de564c935c9f28462700c4",
  "registered": false
}
]
```


## Setting worker status

The status of an existing worker can be set by sending a PUT request to the `workers/<worker_id>` end-point with a json file containing the worker id and the new status:
```json
{
  "worker_id": "46e046cdc32ccae16accc44344703c09c2de19b6c4d9efeeeb28575efd47e7d2",
  "registered": false
}
```
And the corresponding curl command would be:
```bash
curl --user dcf_server_admin:str0ng_pass_word \
        --header "Content-Type: application/json" \
	--request PUT http://188.121.1.122:8080/workers/6e046cdc32ccae16accc44344703c09c2de19b6c4d9efeeeb28575efd47e7d2 \
        --data "@worker_status_info.json"
```
Please note this action is not meaningful when the server is running in the unsafe mode because a worker registering for the first time is assigned its own unique id. If successful, the server will return a json message saying that operation was successful, the worker id, and its new status.

```json
{
  "success": "Successfully changed status for worker 46e046cdc32ccae16accc44344703c09c2de19b6c4d9efeeeb28575efd47e7d2.",
  "worker_id": "46e046cdc32ccae16accc44344703c09c2de19b6c4d9efeeeb28575efd47e7d2",
  "registered": false
}
```
Otherwise there will be a json file with an error message instead along with a description of the error.


## Deleting an existing worker

An existing worker can be deleted by sending a DELETE request to an end-point `workers/<worker_id>`. A curl request for this would like as follows:

the corresponding curl command would be:
```bash
curl --user dcf_server_admin:str0ng_pass_word \
	--request DELETE http://188.121.1.122:8080/workers/6e046cdc32ccae16accc44344703c09c2de19b6c4d9efeeeb28575efd47e7d2
```
If successful, the server will return a json message saying that operation was successful, the worker id, and its new status.

```json
{
  "success": "Successfully removed worker 46e046cdc32ccae16accc44344703c09c2de19b6c4d9efeeeb28575efd47e7d2.",
  "worker_id": "46e046cdc32ccae16accc44344703c09c2de19b6c4d9efeeeb28575efd47e7d2",
}
```
Otherwise there will be a json file with an error message instead along with a description of the error.

 





 