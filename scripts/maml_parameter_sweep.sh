#!/bin/bash


DATASETS="Polynomial CIFAR 7Scenes Ant"
INTERNAL_LR="0.01 0.005 0.001 0.0005 0.0001 0.00005"
EPOCHS=1000

for dataset in $DATASETS; do
  for lr in $INTERNAL_LR; do
    python test.py  --dataset $dataset --alg MAML1 --epochs $EPOCHS --maml_internal_learning_rate $lr --log_dir logs_param_sweep
    python test.py  --dataset $dataset --alg MAML5 --epochs $EPOCHS --maml_internal_learning_rate $lr --log_dir logs_param_sweep
  done
done
