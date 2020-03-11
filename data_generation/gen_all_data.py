import os

langs = ['en', 'ja', 'dk', 'fr']
tasks = ['syn', 'sem']
# NOTE: Should loop through this, but want to keep old no-sent data for now
sent = ['sent', '']

for task in tasks:
    for lang in langs:
        print('Generating: \nTask: {}, Lang: {}'.format(task.upper(), lang.upper()))
        os.system('python3 ./gen_{}_data.py -lang {} -samples 50000 -range 1000 -sent'.format(task, lang))