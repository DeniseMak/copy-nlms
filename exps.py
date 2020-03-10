import os

langs = ['ja', 'en', 'dk', 'fr']
tasks = ['sem', 'syn']
models = ['d-bert']#, 'roberta', 'xlm', 'bert']

print("Starting experiments")

for task in tasks:
    for model in models:
        for lang in langs:
            print('Starting Training: \nTask: {}, Model: {}, Lang: {}'.format(task.upper(), model.upper(), lang.upper()))
            os.system('python3 ./models.py -v 100 -train ./data/{}_{}_train.csv -test ./data/{}_{}_test.csv -model {} -epochs 100 -mb 32 -out_f ./results/{}_{}_{}.txt'.format(lang, task, lang, task, model, lang, task, model))
