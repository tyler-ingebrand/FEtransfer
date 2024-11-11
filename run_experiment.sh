#!/bin/bash

#python src/datasets/CIFAR100.py # downloads CIFAR data
#python src/datasets/7Scenes.py # downloads 7scenes data
#python src/datasets/MuJoCoAnt.py # computes MuJoCo data

#EPOCHS=1000
#python test.py --alg LS --epochs $EPOCHS
#python test.py --alg IP --epochs $EPOCHS
#python test.py --alg AE --epochs $EPOCHS
#python test.py --alg Transformer --epochs $EPOCHS
#python test.py --alg TFE --epochs $EPOCHS
#python test.py --alg Oracle --epochs $EPOCHS
#python test.py --alg BFB --epochs $EPOCHS
#python test.py --alg BF --epochs $EPOCHS
#python test.py --alg MAML1 --epochs $EPOCHS
#python test.py --alg MAML5 --epochs $EPOCHS

#EPOCHS=10000
#python test.py  --dataset CIFAR --alg LS --epochs $EPOCHS
#python test.py  --dataset CIFAR --alg IP --epochs $EPOCHS
#python test.py  --dataset CIFAR --alg AE --epochs $EPOCHS
#python test.py  --dataset CIFAR --alg Transformer --epochs $EPOCHS
#python test.py  --dataset CIFAR --alg TFE --epochs $EPOCHS
#python test.py  --dataset CIFAR --alg Oracle --epochs $EPOCHS
#python test.py  --dataset CIFAR --alg BFB --epochs $EPOCHS
#python test.py  --dataset CIFAR --alg BF --epochs $EPOCHS
#python test.py  --dataset CIFAR --alg MAML1 --epochs $EPOCHS --cross_entropy

#EPOCHS=1000
#python test.py  --dataset Categorical --alg LS --epochs $EPOCHS
#python test.py  --dataset Categorical --alg IP --epochs $EPOCHS
#python test.py  --dataset Categorical --alg AE --epochs $EPOCHS
#python test.py  --dataset Categorical --alg Transformer --epochs $EPOCHS
#python test.py  --dataset Categorical --alg TFE --epochs $EPOCHS
#python test.py  --dataset Categorical --alg Oracle --epochs $EPOCHS
#python test.py  --dataset Categorical --alg BFB --epochs $EPOCHS
#python test.py  --dataset Categorical --alg BF --epochs $EPOCHS
#python test.py  --dataset Categorical --alg MAML --epochs $EPOCHS

#EPOCHS=1000
#python test.py  --dataset 7Scenes --alg LS --epochs $EPOCHS
#python test.py  --dataset 7Scenes --alg IP --epochs $EPOCHS
#python test.py  --dataset 7Scenes --alg AE --epochs $EPOCHS
#python test.py  --dataset 7Scenes --alg Transformer --epochs $EPOCHS
#python test.py  --dataset 7Scenes --alg TFE --epochs $EPOCHS
#python test.py  --dataset 7Scenes --alg Oracle --epochs $EPOCHS
#python test.py  --dataset 7Scenes --alg BFB --epochs $EPOCHS
#python test.py  --dataset 7Scenes --alg BF --epochs $EPOCHS
#python test.py  --dataset 7Scenes --alg MAML1 --epochs $EPOCHS
#python test.py  --dataset 7Scenes --alg MAML5 --epochs $EPOCHS

EPOCHS=1000
python test.py  --dataset Ant --alg LS --epochs $EPOCHS
python test.py  --dataset Ant --alg IP --epochs $EPOCHS
python test.py  --dataset Ant --alg AE --epochs $EPOCHS
python test.py  --dataset Ant --alg Transformer --epochs $EPOCHS
python test.py  --dataset Ant --alg TFE --epochs $EPOCHS
python test.py  --dataset Ant --alg Oracle --epochs $EPOCHS
python test.py  --dataset Ant --alg BFB --epochs $EPOCHS
python test.py  --dataset Ant --alg BF --epochs $EPOCHS
python test.py  --dataset Ant --alg MAML1 --epochs $EPOCHS
python test.py  --dataset Ant --alg MAML5 --epochs $EPOCHS