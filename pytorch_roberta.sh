#!/bin/bash

python3 ./pytorch_roberta.py -data ./data/ja_syn_pair_words.txt -labels ./data/ja_syn_labels.txt -v 1 -epochs 5 > roberta_output
