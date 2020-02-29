import os

langs = ['en', 'ja', 'dk', 'fr']
tasks = ['syn', 'sem']

for task in tasks:
    for lang in langs:
        print('Generating: \nTask: {}, Lang: {}'.format(task.upper(), lang.upper()))
        os.system('python3 ./gen_{}_data.py -lang {} -samples 10000 -range 5000'.format(task, lang))