#!/bin/bash

source activate cmr

cd /cmr
pip install --no-cache-dir -r requirements.txt
bash external/install_external.sh
