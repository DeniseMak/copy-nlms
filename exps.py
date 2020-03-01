import os

langs = ['en', 'ja', 'dk', 'fr']
tasks = ['sem', 'syn']

for task in tasks:
    for model in models:
        for lang in langs:
            print('Starting Training: \nTask: {}, Model: {}, Lang: {}'.format(task.upper(), model.upper(), lang.upper()))
            os.system('python3 ./models.py -v 1 -train ./data/{}_{}_train.csv -test ./data/{}_{}_test.csv -model {} -epochs 10 -mb 256 -out_f ./results/{}_{}_{}.txt'.format(lang, task, lang, task, model, lang, task, model))