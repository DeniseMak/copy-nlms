import os

langs = ['ja', 'en', 'dk', 'fr']
tasks = ['syn']
models = ['d-bert']
sents = ['sent']
print("Starting experiments")

for task in tasks:
    for model in models:
        for lang in langs:
            for sent in sents:
                print('Starting Training: \nTask: {}, Model: {}, Lang: {}'.format(task.upper(), model.upper(), lang.upper()))
                os.system('python3 ./models.py -freeze -v 100 -train ./data/{}/{}_{}_train.csv -test ./data/{}/{}_{}_test.csv -model {} -epochs 100 -lr 1e-5 -mb 32 -out_f ./results/{}/{}_{}_{}.txt -lr 1e-5'.format(sent, lang, task, sent, lang, task, model, sent, lang, task, model))
