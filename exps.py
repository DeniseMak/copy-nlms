import os

langs = ['en', 'ja', 'dk', 'fr']
models = ['roberta', 'bert', 'xlm']
tasks = ['syn']#, 'sem']

for task in tasks:
    for model in models:
        for lang in langs:
            print('Starting Training: \nTask: {}, Model: {}, Lang: {}'.format(task.upper(), model.upper(), lang.upper()))
            os.system('python3 ./models.py -v 100 -train ./data/{}_{}_train.csv -test ./data/{}_{}_test.csv -model {} -epochs 1 -out_f ./results/{}_{}_{}.txt'.format(lang, task, lang, task, model, lang, task, model))
            # with open('./results/{}_{}_{}.txt'.format(task, model, lang), 'w+') as f:
            #     f.write(result)