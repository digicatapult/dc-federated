# Installing the package from the repo

In the following we assume you have cloned the repo in the folder `dc_federated` and that you are using a Unix based operating system such as Linux or MacOS. You will need to translate the instructions appropriately for Windows.

## Python and virtualenv

Make sure that you are using python 3.7 by running `python --version`. If not, the recommended way to manage python versions is to use [`pyenv`](https://github.com/pyenv/pyenv):

```bash
pyenv install 3.7.8
pyenv local 3.7.8
```

Then check the python versions are correctly setup:

```bash
> python -V
Python 3.7.8
> pip -V
pip 20.2.1 from ~/.pyenv/versions/3.7.8/lib/python3.7/site-packages/pip (python 3.7)
```

Then make sure that the `virtualenv` package is installed within your python environment. Then `cd` into a folder where you want your virtual environments to be and run:

```bash
> virtualenv venv_dc_federated
```

You can also use other virtual environemnt packages as well. Now `cd` into the folder `dc_federated`, activate the environment and install the necessary packages:

```bash
> source ./venv_dc_federated/bin/activate
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

To ensure it has installed properly, `cd` to `./tests` and run the tests and follow the instructions that appear. Both tests should complete successfully.

```bash
(venv_dc_federated)> cd ./tests
(venv_dc_federated)> python test_backend.py
(venv_dc_federated)> python test_backend_model_integration.py
```
