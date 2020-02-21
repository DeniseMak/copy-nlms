#!/bin/bash

source ~/.bashrc
conda activate the_count_env

python3 ./pytorch_roberta.py -data ./data/en_syn_pair_words.txt -labels ./data/en_syn_labels.txt -v 100 -epochs 5 > roberta_output_syn

python3 ./pytorch_roberta.py -data ./data/en_sem_pair_words.txt -labels ./data/en_sem_labels.txt -v 100 -epochs 5 > roberta_output_sem
