import torch
import numpy as np
import scipy.io
from random import shuffle
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open('naval_dataset.pkl', 'rb') as f:
        train_data, train_label, val_data, val_label = pickle.load(f)

    train_data = np.transpose(train_data,(0,2,1))
    val_data = np.transpose(val_data,(0,2,1))

    train_label = []
    for k in range(train_data.shape[0]):
        if np.min(train_data[k,:,1])<20:
            train_label.append(2)
        elif train_data[k,-1,0] > 30:
            train_label.append(3)
        else:
            train_label.append(1)
    train_label = np.array(train_label)

    val_label = []
    for k in range(val_data.shape[0]):
        if np.min(val_data[k,:,1])<20:
            val_label.append(2)
        elif val_data[k,-1,0] > 30:
            val_label.append(3)
        else:
            val_label.append(1)
    val_label = np.array(val_label)

    f = open('naval_class.pkl', 'wb')
    pickle.dump([train_data, train_label, val_data, val_label], f)
    f.close()