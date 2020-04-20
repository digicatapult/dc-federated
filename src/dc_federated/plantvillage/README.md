## dc_federated with PlantVillage dataset

Project for DC AIML team federated learning demo.

### Preparing the datasets

In the following we assume you have cloned the repo in the folder `FederatedLearningDemo`.
The original PlantVillage dataset shall be in the folder `PlantVillage-Dataset`, and can
be obtained from:
```bash
> git clone https://github.com/spMohanty/PlantVillage-Dataset
```

Use the configuration file `PlantVillage_cfg.yaml` to define the split of  the dataset
in multiple subsets and customized distributions.

Apply the parameters from PlantVillage_cfg.yaml to create data subests using:
```bash
> python dataset_prep.py
```

### Running the demo

In the following we assume you have cloned the repo in the folder `FederatedLearningDemo`.
The demo is using MobileNetv2 as base models.

Start by running a server using:
```bash
> python plant_fed_avg_server.py
```
The server ip and port are printed as output for running the workers.

By default, the projects expect 4 workers.
Start as many workers as required using:
```bash
> python plant_fed_avg_worker.py --server-host-ip [server ip] --server-port 8080 --worker-id [worker id]
```
