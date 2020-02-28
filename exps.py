import subprocess

langs = ['en', 'ja', 'dk', 'fr']
models = ['roberta', 'bert', 'xlm']
tasks = ['syn', 'sem']

for task in tasks:
    for model in models:
        for lang in langs:
            print('Starting Training: \nTask: {}, Model: {}, Lang: {}'.format(task.upper(), model.upper(), lang.upper()))
            result = subprocess.check_output('python3 ./pytorch_roberta.py -v 100 -train ./data/{}_{}_train.csv -test ./data/{}_{}_test.csv -model {} -epochs 1'.format(lang, task, lang, task, model), shell=True)
            with open('./results/{}_{}_{}.txt'.format(task, model, lang), 'w+') as f:
                f.write(result)