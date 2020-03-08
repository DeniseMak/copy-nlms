import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style

langs = ['en', 'ja', 'dk', 'fr', 'es', 'ar', 'fi', 'ru', 'th']
models = ['edit_distance']


def plot(langs):

    data = pd.read_csv("./edit_distance_results.csv")

    style.use("fivethirtyeight")
    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)

    # Subplot 1 (line)
    ax1 = fig.add_subplot(211)
    plt.xlabel("Window Size")
    plt.ylabel("Average Edit Distance in Window")
    plt.title("Complexity by Window Size")
    
    legend = []
    for lang in langs:
        lang_data = data[data.lang == lang]
        ax1.plot(lang_data.window, lang_data.score)
        legend.append(lang)
    plt.legend(legend)
    

    # Subplot 2 (bargraph)
    ax2 = fig.add_subplot(212)
    averages = []
    for lang in langs:
        averages.append(data[data.lang == lang]["score"].mean())
    plt.title("Average Complexity (All Window Sizes)")
    plt.bar(langs, averages, label="label")
    
    
    plt.show()

for model in models:
    output = "./" + model+ "_results.csv"
    with open(output, "w+") as f:
        f.write("lang,window,score\n")
    for lang in langs:
        for window_size in range(1, 31):
            print('Model: {}, Lang: {}, Window_Size: {}'.format(model.upper(), lang.upper(), window_size))
            os.system('python ./{} -lang {} -window_size {} -output {}'.format(model + ".py", lang, window_size, output))


plot(langs)
