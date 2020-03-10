import os
from matplotlib import pyplot as plt
from matplotlib import style


with open("./en_sem_d-bert.txt", "r") as f:
    text = f.read()
    text_split = text.split("Loss:")
    epochs = []
    losses = []
    losses2 = []
    for i in range(1, len(text_split), 2):
        epochs.append(text_split[i-1].split("Epoch")[1])
        losses.append(float(text_split[i].split(",")[0]))
        losses2.append(float((text_split[i+1].split("Test")[0])))

        if i == 199:
            break
    

    plt.plot(epochs, losses2)
    plt.plot(epochs, losses)
    plt.show()

