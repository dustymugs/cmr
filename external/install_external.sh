#!/bin/bash

cd /cmr/external

rm -rf PerceptualSimilarity/ neural_renderer/

git clone https://github.com/shubhtuls/PerceptualSimilarity
touch PerceptualSimilarity/__init__.py
#git clone https://github.com/dustymugs/PerceptualSimilarity

# not using neural-renderer-pytorch from pypi
git clone https://github.com/hiroharu-kato/neural_renderer --branch v1.1.0
cd neural_renderer
python setup.py install
