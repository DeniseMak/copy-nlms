import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style

langs = ['en', 'ja', 'dk', 'fr', 'es', 'ar', 'fi', 'ru', 'th']
models = ['transparency']


def plot(langs):

    data = pd.read_csv("./transparency_results.csv")

    # style.use("fivethirtyeight")
    scores = []
    for lang in langs:
        scores.append(int(data[data.lang == lang]["score"]))
    plt.title("Transparency Score (Higher is Better)")
    plt.bar(langs, scores)
    plt.show()

for model in models:
    output = "./" + model+ "_results.csv"
    with open(output, "w+") as f:
        f.write("lang,score\n")
    for lang in langs:
        print('Model: {}, Lang: {}'.format(model.upper(), lang.upper()))
        os.system('python ./{} -lang {} -output {}'.format(model + ".py", lang, output))

plot(langs)
