#!/bin/bash




#python src/datasets/CIFAR100.py # downloads CIFAR data
#python src/datasets/7Scenes.py # downloads 7scenes data
python src/datasets/MuJoCoAnt.py # computes MuJoCo data

#ALGS="LS IP AE Transformer TFE Oracle BFB BF MAML1 MAML5 Siamese Proto"
#DATASETS="Polynomial CIFAR 7Scenes Ant"
#EPOCHS=10000
#
#for dataset in $DATASETS
#do
#    for alg in $ALGS
#    do
#        python test.py --dataset $dataset --alg $alg --epochs $EPOCHS
#        sleep 1
#    done
#done
python test.py --dataset Ant --alg Transformer --epochs 10000  --n_params 10000000
python test.py --dataset Ant --alg Transformer --epochs 10000  --n_params 1000000
python test.py --dataset Ant --alg LS --epochs 10000 --n_basis 100 --n_params 10000000
python test.py --dataset Ant --alg LS --epochs 10000 --n_basis 200 --n_params 10000000
python test.py --dataset Ant --alg LS --epochs 10000 --n_basis 300 --n_params 10000000
python test.py --dataset Ant --alg LS --epochs 10000 --n_basis 400 --n_params 10000000
python test.py --dataset Ant --alg LS --epochs 10000 --n_basis 500 --n_params 10000000