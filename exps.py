import os

langs = ['en', 'ja', 'dk', 'fr']
tasks = ['sem', 'syn']
models = ['xlm', 'bert']
with open('test', 'w+') as f:
   f.write('starting exps')
for task in tasks:
    with open('test', 'w+') as f:
        f.write(task)
    for model in models:
        with open('test', 'w+') as f:
            f.write(model)
        for lang in langs:
            with open('test', 'w+') as f:
                f.write('Starting Training: \nTask: {}, Model: {}, Lang: {}'.format(task.upper(), model.upper(), lang.upper()))
            # print('Starting Training: \nTask: {}, Model: {}, Lang: {}'.format(task.upper(), model.upper(), lang.upper()))
            os.system('python3 ./models.py -v 1 -train ./data/{}_{}_train.csv -test ./data/{}_{}_test.csv -model {} -epochs 10 -mb 10 -out_f ./results/{}_{}_{}.txt'.format(lang, task, lang, task, model, lang, task, model))
