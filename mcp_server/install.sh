#!/bin/bash

python3 -m venv mcp_env
source mcp_env/bin/activate

pip install -r requirements.txt

git clone git@github.com:cucinellclark/bvbrc-python-api.git
cd bvbrc-python-api
pip install -U pip
pip install -e .
cd ..

