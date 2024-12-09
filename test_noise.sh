#!/bin/bash


python test.py --alg LS --epochs 50000 --data CIFAR --seed 1 --log_dir logs/noise
python test.py --alg LS --epochs 50000 --data CIFAR --seed 1 --log_dir logs/noise --add_noise