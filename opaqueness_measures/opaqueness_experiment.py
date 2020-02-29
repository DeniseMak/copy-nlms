import os
import pandas as pd
from matplotlib import pyplot as plt

langs = ['en', 'ja', 'dk', 'fr']
models = ['edit_distance']

# for model in models:
#     output = "opaqueness_measures/" + model+ "_results.csv"
#     with open(output, "w+") as f:
#         f.write("lang,window,score\n")
#     for lang in langs:
#         for window_size in range(1, 31):
#             print('Model: {}, Lang: {}, Window_Size: {}'.format(model.upper(), lang.upper(), window_size))
#             os.system('python opaqueness_measures/{} -lang {} -window_size {} -output {}'.format(model + ".py", lang, window_size, output))
    
# Output model results graphically
data = pd.read_csv("opaqueness_measures/edit_distance_results.csv")
en = data[data.lang == "en"]
ja = data[data.lang == "ja"]
fr = data[data.lang == "fr"]
dk = data[data.lang == "dk"]
plt.plot(en.window, en.score)
plt.plot(ja.window, ja.score)
plt.plot(fr.window, fr.score)
plt.plot(dk.window, dk.score)
plt.legend(["English", "Japanese", "French", "Danish"])
plt.xlabel("Window Size")
plt.ylabel("Complexity Score")
plt.title("Language Opaqueness")
plt.show()