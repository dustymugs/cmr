#!/bin/bash

source activate cmr

cd /projects/cmr
pip install --no-cache-dir -r requirements.txt
cd external
rm -rf PerceptualSimilarity/ neural_renderer/
bash install_external.sh
