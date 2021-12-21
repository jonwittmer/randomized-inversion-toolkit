#!/bin/bash

# run all randomization schemes with various number of random vectors
python nonlinear_AD_script.py --strategy "rma" --n_random_vectors 10
python nonlinear_AD_script.py --strategy "rma" --n_random_vectors 20
python nonlinear_AD_script.py --strategy "rma" --n_random_vectors 50
python nonlinear_AD_script.py --strategy "rma" --n_random_vectors 100
python nonlinear_AD_script.py --strategy "no_randomization" --n_random_vectors 0
