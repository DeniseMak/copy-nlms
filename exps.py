import subprocess

langs = ['en', 'ja', 'dk', 'fr']
models = ['bert', 'xlm']
tasks = ['syn', 'sem']

for task in tasks:
    for model in models:
        for lang in langs:
            print('Starting Training: \n\
                Task: {}, Model: {}, Lang: {}'.format(task, model, lang))
            result = subprocess.check_output('python3 ./pytorch_roberta.py -v 100 \
                -train {}_{}_train.csv \
                -test {}_{}_test.csv -model {} -epochs 20'.format(lang, task, lang, task, model), shell=True)
            with open('./results/{}_{}_{}.txt'.format(task, model, lang), 'w+') as f:
                f.write(result)