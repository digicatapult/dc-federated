## dc_federated

Project for DC AIML team federated learning demo.

### Installing the package from the repo using conda
 
In the following we assume you have cloned the repo in the folder `FederatedLearningDemo`.
 
Start by creating the environment using:

```bash
> conda create --name dc_federated
```

Now `cd` into the folder `FederatedLearningDemo`, activate the environment and install the necessary packages:

```bash
> conda activate dc_federated
> conda install pip
> pip install -r requirements.txt
```

Finally, install this package itself by running: 

```bash
python setup.py install
```

Alternatively, to install in developer mode (so that any code change you make in the local repo is available immediately after), run 

```bash
python setup.py develop
```

To ensure it has installed properly, `cd` to `FederatedLearningDemo/tests` and run the tests and follow the instructions that appear. Both tests should complete successfully.

```bash
> python test_backend.py
> python test_backend_model_integration.py
```
