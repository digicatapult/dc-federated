## Federated Learning with PlantVillage dataset

Project for DC AIML team federated learning demo.
Training a classifier on images of crop disease from the PlantVillage dataset using
federated learning.

### Preparing the datasets

In the following we assume you have cloned the repo in the folder `FederatedLearningDemo`.
The original PlantVillage dataset shall be in the folder `PlantVillage-Dataset`, and can
be obtained from:
```bash
> git clone https://github.com/spMohanty/PlantVillage-Dataset
```

Use the configuration file `PlantVillage_cfg.yaml` to define how to split of the dataset
in multiple subsets and with customized distributions:
* Target directory for the train, validation and test data,
* PlantVillage dataset categories to use as a list,
* PlantVillage dataset distributions per worker,
* Number of subsets,

Apply the parameters from PlantVillage_cfg.yaml to create data subsets using:
```bash
> python dataset_prep.py
```

### Running the demo

In the following we assume you have cloned the repo in the folder `FederatedLearningDemo`.
We assume you have installed the package.
The demo is using MobileNetv2 as base models.

#### Server

Start by running a server using:
```bash
> python plant_fed_avg_server.py
```
Use the configuration file `PlantVillage_cfg.yaml` to define datasets location, where to save model checkpoints and the expected number of update per iteration.
Optionally, the training and validation data can be supplied as argument using:
```bash
> python plant_fed_avg_server.py --validation-data-path [path] --checkpoint-path [path] --update-lim [int]
```

#### Workers

By default, the projects expect 4 workers.
Start as many workers as required using:
```bash
> python plant_fed_avg_worker.py --server-host-ip [server ip] --server-port 8080 --worker-id [int]
```

Use the configuration file `PlantVillage_cfg.yaml` to define datasets location and the hyperparameters for training the models.
Optionally, the training and validation data can be supplied as argument using:
```bash
> python plant_fed_avg_worker.py --server-host-ip [server ip] --server-port 8080 --worker-id [worker id] --train-data-path [path] --validation-data-path [path]
```