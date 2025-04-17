import torch
from torch.nn import functional as F
import numpy as np
import sys
import matplotlib.pyplot as plt

def normalize_dataset(sample):
    nsample = sample.shape[0]
    dim = sample.shape[2]
    mean = torch.mean(sample,0)
    std = torch.std(sample,0)
    # mean = torch.empty(nsample,1,dim)
    # std = torch.empty(nsample,1,dim)
    # for i in range(dim):
    #     mean[:,:,i] = torch.mean(sample,i)
    #     std[:,:,i] = torch.std(sample,i)
    return (sample - mean) / std


def get_time(w):
    w = w.bool()
    l = w.shape[0]
    t = []
    tf = False
    for j in range(l):
        if w[j] != tf:
            if tf == True:
                t.append(j-1)
            else:
                t.append(j)
            tf = not tf
        if len(t) == 2:
            break
    if tf == True:
        t.append(l-1)
    if len(t)==0:
        return t, t
    else:
        return t[0], t[1]
    
def plot_2d(data,labels,filename):
    ns = data.shape[0]
    plt.figure(figsize=(8, 5))
    for i in range(ns):
        if labels[i] == 1:
            plt.plot(data[i,:,0],data[i,:,1],color='g',label='1')
        else:
            plt.plot(data[i,:,0],data[i,:,1],color='r',label='-1')
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("dataset", fontsize=16)
    figname = filename+'.png'
    plt.savefig(figname)
    plt.show()

