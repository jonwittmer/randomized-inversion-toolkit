#!/bin/bash
python nonlinear_AD_script.py --strategy no_randomization --n_random_vectors 0 | tee output.o

python nonlinear_AD_script.py --strategy rma --n_random_vectors 10 | tee -a output.o
python nonlinear_AD_script.py --strategy rma --n_random_vectors 100 | tee -a output.o
python nonlinear_AD_script.py --strategy rma --n_random_vectors 1000 | tee -a output.o
python nonlinear_AD_script.py --strategy rma --n_random_vectors 10000 | tee -a output.o

python nonlinear_AD_script.py --strategy rmap --n_random_vectors 10 | tee -a output.o
python nonlinear_AD_script.py --strategy rmap --n_random_vectors 100 | tee -a output.o
python nonlinear_AD_script.py --strategy rmap --n_random_vectors 1000 | tee -a output.o
python nonlinear_AD_script.py --strategy rmap --n_random_vectors 10000 | tee -a output.o

python nonlinear_AD_script.py --strategy rma_rmap --n_random_vectors 10 | tee -a output.o
python nonlinear_AD_script.py --strategy rma_rmap --n_random_vectors 100 | tee -a output.o
python nonlinear_AD_script.py --strategy rma_rmap --n_random_vectors 1000 | tee -a output.o
python nonlinear_AD_script.py --strategy rma_rmap --n_random_vectors 10000 | tee -a output.o

python nonlinear_AD_script.py --strategy rs --n_random_vectors 10 | tee -a output.o
python nonlinear_AD_script.py --strategy rs --n_random_vectors 100 | tee -a output.o
python nonlinear_AD_script.py --strategy rs --n_random_vectors 1000 | tee -a output.o
python nonlinear_AD_script.py --strategy rs --n_random_vectors 10000 | tee -a output.o
