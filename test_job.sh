#!/bin/bash

source ~/.bashrc
conda activate the_count_env

python ./pytorch_roberta.py -task sem -epochs 1 -model roberta -train ./data/en_sem_train.csv -test ./data/en_sem_test.csv -mb 32 -v 1 > test.txt