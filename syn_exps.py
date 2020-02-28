import os

langs = ['en', 'ja', 'dk', 'fr']
models = ['bert', 'xlm']
tasks = ['syn', 'sem']

for lang in langs:
    for model in models:
        for task in tasks:
            os.system('python3 ./pytorch_roberta.py -v 100 -train {}_{}_train.csv \
                -test {}_{}_test.csv -model {}'.format(lang, task, lang, task, model))