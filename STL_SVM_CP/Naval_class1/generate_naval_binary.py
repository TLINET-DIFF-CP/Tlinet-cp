import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from random import shuffle

if __name__ == "__main__":
    with open('naval_class.pkl', 'rb') as f:
        train_data, train_label, val_data, val_label = pickle.load(f)

    train_label[train_label!=1] = -1
    train_label[train_label!=-1] = 1
    val_label[val_label!=1] = -1
    val_label[val_label!=-1] = 1

    n = val_label.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    cal_data = val_data[indices[:n // 2],:]
    cal_label = val_label[indices[:n // 2]]
    pred_data = val_data[indices[n // 2:],:]
    pred_label = val_label[indices[n // 2:]]

    plt.figure(figsize=(8, 4))
    for j in range(val_data.shape[0]):
        if val_label[j] == 1:
            plt.plot(val_data[j,:,0],val_data[j,:,1], color = 'blue')
        else:
            plt.plot(val_data[j,:,0],val_data[j,:,1], color = 'red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Naval dataset")
    plt.grid()
    plt.show()

    with open('naval.pkl', "wb") as file:
        pickle.dump([train_data, train_label, cal_data, cal_label, pred_data, pred_label],file)