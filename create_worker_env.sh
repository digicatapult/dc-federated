pip install virtualenv
mkdir ../venvs
virtualenv ../venvs/dc_fed_venv
source ../venvs/dc_fed_venv/bin/activate
pip install -r requirements.txt
python setup.py develop
cd src/dc_federated/stress_test
tar -xvf gz_stress_keys_folder.tar.gz
