import subprocess

langs = ['en', 'ja', 'dk', 'fr']
models = ['bert', 'xlm']
tasks = ['syn', 'sem']

for lang in langs:
    for model in models:
        for task in tasks:
            result = subprocess.check_output('python3 ./pytorch_roberta.py -v 100 \
                -train {}_{}_train.csv \
                -test {}_{}_test.csv -model {} -epochs 20'.format(lang, task, lang, task, model), shell=True)
            with open('./results/{}_{}_{}.txt'.format(lang, task, model), 'w+') as f:
                f.write(result)