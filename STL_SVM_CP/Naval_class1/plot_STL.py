import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from train_telex import *
from neurons_prob import *
from functions import *


def denormalize(train_data):
    min_train = torch.min(train_data.view(-1, train_data.size(-1)), dim=0).values
    max_train = torch.max(train_data.view(-1, train_data.size(-1)), dim=0).values
    return min_train, max_train

def get_predicate(stl):
    x_pred = []
    y_pred = []
    for k, predi in enumerate(stl.pred):
        if stl.p[0,k] == 1:
            if predi.dim==0:
                x_pred.append(predi.b.item()/predi.a.item())
            if predi.dim==1:
                y_pred.append(predi.b.item()/predi.a.item())
    return x_pred, y_pred


if __name__ == "__main__":
    with open('naval_positive.pkl', 'rb') as f:
        positive_data, data, label = pickle.load(f)
    # positive_data, data = NormalizeDataset(positive_data, data)
    positive_data = torch.tensor(positive_data, requires_grad=False)
    data = torch.tensor(data, requires_grad=False)
    label = torch.tensor(label, requires_grad=False)
    min_train, max_train = denormalize(positive_data)

    nsample = data.shape[0]
    length = data.shape[1]
    dim = data.shape[2]
    nf = dim*4
    nc = 2
    weight_reg = [1e-3,1e-2,1e-1]
    weight_a = 1e-3
    weight_eps = 1e-3
    lmb = 1

    stl = STL_Anomaly(nf,nc,length,weight_reg,weight_a)
    stl.load_state_dict(torch.load("naval_positive_model97.pth"))
    stl.denormalize_predicate(min_train,max_train)
    p_buffer = stl.extract_formula(data,label)
    stl.translate_formula()
    r_val = stl.robustness(data)
    label_predicted = Clip.apply(r_val)
    acc = torch.sum((label_predicted==label))/nsample
    print('validation mean robustness = {:.2f}, val_accuracy = {:.2f}'.format(torch.mean(r_val), acc))

    plt.figure(figsize=(8, 5))
    for j in range(positive_data.shape[0]):
        plt.plot(positive_data[j,:,0],positive_data[j,:,1], color = 'green', label = 'positive')
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("naval")
    plt.grid(True)
    figname = "naval_positive (training).png"
    plt.savefig(figname, format='png')
    plt.show()

    x_pred, y_pred = get_predicate(stl)
    plt.figure(figsize=(8, 5))
    for j in range(data.shape[0]):
        if label[j] == 1:
            plt.plot(data[j,:,0],data[j,:,1], color = 'green', label = 'positive')
        else:
            plt.plot(data[j,:,0],data[j,:,1], color = 'red', label = 'negative')
    for xi in x_pred:
        plt.axvline(x=xi)
    for yi in y_pred:
        plt.axhline(y=yi)
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("naval")
    plt.grid(True)
    figname = "naval validation set.png"
    plt.savefig(figname, format='png')
    plt.show()
    