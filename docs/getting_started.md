## Installing the package from the repo
 
In the following we assume you have cloned the repo in the folder `dc_fed_lib` and that you are using a Unix based operating system such as Linux or MacOS. You will need to translate the instructions appropriately for Windows. 

First, make sure that the `virtualenv` package is installed within your python environment. Then `cd`  into a folder where you want your virtual environments to be and run:
```bash
> virtualenv venv_dc_federated
```
You can also use other virtual environemnt packages as well. Now `cd` into the folder `dc_fed_lib`, activate the environment and install the necessary packages:

```bash
> source /path/to/venv_dc_federated/bin/activate
(venv_dc_federated)> pip install -r requirements.txt
```
Finally, install this package itself by running: 

```bash
(venv_dc_federated)> python setup.py install
```

Alternatively, to install in developer mode (so that any code change you make in the local repo is available immediately after), run 

```bash
(venv_dc_federated)> python setup.py develop
```
To ensure it has installed properly, `cd` to `dc_fed_lib/tests` and run the tests and follow the instructions that appear. Both tests should complete successfully.

```bash
(venv_dc_federated)> python test_backend.py
(venv_dc_federated)> python test_backend_model_integration.py
```
