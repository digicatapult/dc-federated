# Deployment Notes

The first point we make here is that deployment of an `dc_federated` application in a POC or production setting  should always be done in consultation with the dev-ops team. This document, therefore, should serve as a starting point, where we highlight the main issues that need to be considered for deployment. The solutions we provide are by no means exhaustive - we have found them to work for the applications we have tried, but they are likely not the complete answer for your use-case.  

At a minimum, one needs to consider two different issues during deployment: scalability and security. In the following we address each in turn.


## Scalability

We start by recalling that the server side is implemented as a http service and workers make REST API calls to the server during federated learning iterations. During federated learning iterations, the main source of resource drain is multiple workers long-polling to get the current global model. During a long poll the http request from a worker does not return until a new global model is available and the connection is kept open. This reduces the network traffic, but also means that some basic configuration needs to be done on the server side to ensure that the requisite number of open connections are supported. We now describe the such a configuration for the specific setup of server running on Ubuntu 20.0.4 system. This should help guide as to what is needed for your use case.


### Open Files

Concurrency is handled within `dc_federated` using the standard [`gevent`](http://www.gevent.org/) pseudo-thread library, which requires about 5 open files per connection request. This means that the server should be set up to support at least 5 * number-of-workers open files. Assuming that you have 1000 workers, you will need to support about 5000 files being open simultaneously. In the following we will go for an overkill and support 65535 open files.  

As a first step, check your current limit by running  the following on the shell. 
```bash
> ulimit -n 
```
If the number returned is small you will need to change some configurations. Open the file `/etc/security/limits.conf` and add or modify the following lines 
```
*  soft  nproc  65535
*  hard  nproc  65535
*  soft  nofile 65535
*  hard  nofile 65535
``` 
Now open the files (if any) in `/etc/security/limits.d/` and update the entries for `nofile` to 65535 as well. Once you are done, reboot the system and run `ulimit -n` again to confirm the changes has happened. If this does not work for you, or any of the files are missing, contact your local friendly sys-admin or dev-ops to help you. 


### Reverse Proxy
  
For a deployment it is best to set up a reverse proxy which forwards all requests to the federated learning server and sends the responses back to the clients. In the following we describe a sample configuration assume that the reverse proxy is a [nginx](https://www.nginx.com/) server. First of all, if nginx is not installed, please follow the instructions at the site to install and start an nginx web-server and run the test to make sure it is running properly. 

Once the server is up and running, we will configure it to support the federated learning server. Open up the configuration file in `/etc/var/nginx/nginx.conf`. On the top add the following lines (or modify them if they are already there):
```
worker_rlimit_nofile 10000;

events {
	worker_connections 10000;
}
```  
The first option determines the number of simultaneous open files for the server and the second helps set the number of open connections. 

Now open the file `/etc/var/nginx/sites_available/default` (create a different file if you want to put the configuration for the FL server in a different location). Now add the following block to enable reverse-proxy into the FL server.

```
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    
    # This determines the size of the model you can 
    # send to the server - set it to large values for 
    # large models.
    client_max_body_size 20M; 
    
    root /var/www/html;
    index index.html index.htm index.nginx-debian.html;

    server_name _;

    location / {
    
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 1d;
        proxy_send_timeout 1d;
        proxy_read_timeout 1d;
        send_timeout 1d;


        }
}
```
This sets up the server to as a reverse proxy for the upstream server (i.e. the federated learning server) listening at `http://127.0.0.1:5000`, which in turn means that the `DCFServer` instance should started with parameters `server_host_ip= 127.0.0.1` and `server_port=5000`. 

Of course, this is just a starting point - there are likely other configurations you would need to perform for your setup and usecase.

## Security

You have taken the first step in securing your federated learning setup by using a reverse proxy. As a second step you should probably enable certification via SSL (i.e. https communication) which will likely require the following steps:

- registering a domain for your reverse proxy 
- acquiring a certificate for your domain from a service like [certbot](https://certbot.eff.org/) 
- then setting the protocol to `https` when starting workers.

In addition, the `DCFServer` also supports [SSL communication natively](../library/enabling_ssl.md) so that if the communication between the reverse-proxy and the fedearted learning server happens via the internet, this last leg can also be secure. As before, please consult your local sys-admin or dev-ops guru to figure out the details.

There are many other aspects of security that needs to be taken care of based on the specific problem (for instance handling potenial denial-of-service attacks etc.) These should considered and handled by your dev-ops team.
