#!/bin/bash

source ~/.bashrc
conda activate count_2
python3 ./models.py -v 100 -train ./data/en_sem_train.csv -test ./data/en_sem_test.csv -model en-roberta -epochs 100 -mb 32 -out_f ./results/en_sem_en-roberta.txt
python3 ./models.py -v 100 -train ./data/en_syn_train.csv -test ./data/en_syn_test.csv -model en-roberta -epochs 100 -mb 32 -out_f ./results/en_syn_en-roberta.txt
#python3 ./exps.py -lr 0.001 > exps.out
