# Enabling https

The communication between the server and workers can be optionally encrypted using https.

Following is a example using the MNIST dataset and a self-signed certificate for localhost.

## Generate the certificate (development only)

We can use libressl to generate the certificate and key file locally.
In a production setup these files would be provided by a system administrator.

```sh
cd <path to >/src/dc_federated/examples/mnist/

# This will create the key file
openssl genrsa -out ./localhost.key 2048

# This will create a certificate from the key file, valid 10 years and with localhost as Common Name
openssl req -new -x509 \
  -key ./localhost.key \
  -out ./localhost.crt \
  -days 3650 \
  -subj /CN=localhost \
  -extensions SAN \
  -config <(cat /etc/ssl/openssl.cnf \
    <(printf "\n[SAN]\nsubjectAltName=DNS:localhost"))
```

## Run the server:

```sh
python mnist_fed_avg_server.py \
  --server-host-ip localhost \
  --ssl-certfile localhost.crt \
  --ssl-keyfile localhoat.key \
  --ssl-enabled
```

## Run the Worker(s)

The easiest way to work with self-signed certificate during development is to
add the environment variable `REQUESTS_CA_BUNDLE` set to the root CA.

```sh
  REQUESTS_CA_BUNDLE=localhost.crt python mnist_fed_avg_worker.py \
    --server-protocol https \
    --server-host-ip localhost \
    --server-port 8080 \
    --digit-class 0
```

## Production setup

This can be achieved in production in a similar way by replacing the server-host-ip
by the actual server hostname or ip and replacing the certificated and key files by real ones.
The REQUESTS_CA_BUNDLE should not be set and the program instead rely on the system root certificates
to verify the validity of incoming requests.
