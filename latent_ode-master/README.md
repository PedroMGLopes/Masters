# Sparse Medical Time Series Modeling with Neural Differential Equations


# Code built upon the work of Julia Rubanova

https://github.com/YuliaRubanova/latent_ode

## Prerequisites

Install `torchdiffeq` from https://github.com/rtqichen/torchdiffeq.

## Experiments on different datasets

By default, the dataset are downloaded and processed when script is run for the first time. 


* Physiosepsis training
```
run_models.py --niters 10 -n 20000 -b 100 -l 20 --dataset physiosepsis --classic-rnn --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.1 --regress

run_models.py --niters 10 -n 20000 -b 100 -l 20 --dataset physiosepsis --ode-rnn --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.1 --regress

run_models.py --niters 10 -n 20000 -b 100 -l 20 --dataset physiosepsis --rnn-vae --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.1 --regress

run_models.py --niters 10 -n 20000 -b 100 -l 20 --dataset physiosepsis --latent-ode --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.1 --regress

* Physiosepsis classification

run_models.py --niters 10 -n 20000 -b 100 -l 20 --dataset physiosepsis --classic-rnn --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.1 --regress --load XXXXX


* Physiosepsis interpolation

run_models.py --niters 10 -n 20000 -b 100 -l 20 --dataset physiosepsis --classic-rnn --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.1 --regress --load XXXXX --interp-test

* Physiosepsis extrapolation

run_models.py --niters 10 -n 20000 -b 100 -l 20 --dataset physiosepsis --classic-rnn --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.1 --regress --load XXXXX --extrap-test
