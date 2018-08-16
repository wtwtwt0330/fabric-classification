#!/bin/sh

python build_dataset.py --data_dir=../data --seed=0
python train_resnet34.py --data_dir=../data --model_dir=../data/experiments/resnet34 --seed=0
python train_vgg16.py --data_dir=../data --model_dir=../data/experiments/vgg16 --seed=0
python predict_resnet34.py --data_path=../data/test_b.csv --model_dir=../data/experiments/resnet34 --seed=0
python predict_vgg16.py --data_path=../data/test_b.csv --model_dir=../data/experiments/vgg16 --seed=0
python predict_ensemble.py --submission1=../data/experiments/resnet34/submission.csv --submission2=../data/experiments/vgg16/submission.csv
