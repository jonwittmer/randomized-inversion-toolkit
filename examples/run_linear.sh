#!/bin/bash

cd 1d_deconvolution
python 1d_deconvolution.py

cd ../regularization_toolbox_examples
python deriv2.py
python foxgood.py
python gravity.py
python heat.py
python phillips.py
python shaw.py

cd ../xray_tomography
python xray_tomography.py
