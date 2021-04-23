FROM python:3.9.4-slim-buster AS base

WORKDIR /root/
COPY requirements.txt .
RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir torchvision
RUN pip install -r requirements.txt

COPY setup.py .
COPY setup.cfg .
COPY src src
RUN pip install .

FROM base AS mnist_backend
WORKDIR /root/src/dc_federated/examples/mnist
CMD [ "./mnist_fed_avg_server.py"]

FROM base as mnist_worker
WORKDIR /root/src/dc_federated/examples/mnist
CMD [ "./mnist_fed_avg_worker.py" ]

FROM base as plantvillage_base
WORKDIR /root/src/
RUN apt update && apt install -y unzip && apt clean
COPY master.zip .
RUN unzip master.zip -d PlantVillage-Dataset && rm -rf master.zip
RUN mkdir /root/src/PlantVillage-Dataset/checkpoints/
RUN mkdir /root/src/PlantVillage-Dataset/PlantVillage-Dataset-master/processed/
WORKDIR /root/src/dc_federated/examples/plantvillage/
RUN mv PlantVillage_docker_cfg.yaml PlantVillage_cfg.yaml
RUN python dataset_prep.py

FROM plantvillage_base AS plantvillage_backend
ENV UPDATE_LIM=$UPDATE_LIM
WORKDIR /root/src/dc_federated/examples/plantvillage/
CMD [ "sh", "-c", "python plant_fed_avg_server.py --update-lim ${UPDATE_LIM}" ]

FROM plantvillage_base AS plantvillage_worker
ENV WORKER_ID=$WORKER_ID
WORKDIR /root/src/dc_federated/examples/plantvillage/
CMD [ "sh", "-c", "sleep 5 && python plant_fed_avg_worker.py --server-port 8080 --server-host-ip plantvillage_backend --worker-id ${WORKER_ID}  --train-data-path /root/src/PlantVillage-Dataset/PlantVillage-Dataset-master/processed/train${WORKER_ID}/" ]
