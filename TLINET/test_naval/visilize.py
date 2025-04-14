import torch
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import pickle
import os
from utils import *

import sys
sys.path.append('../')
from neurons import *

def plot_timed_data(ax, x, y, interval, nsample):
    t1 = interval[0]
    t2 = interval[1]
    if t2-t1 == 0:
        m = '.'
    else:
        m = ''
    for i in range(nsample):
        path = x[i,t1:(t2+1),:]
        label = y[i]
        if label == 1:
            p1 = ax.plot(path[:,0],path[:,1], color='red',marker=m,label='1')
        else:
            p2 = ax.plot(path[:,0],path[:,1], color='blue',marker=m,label='-1')


if __name__ == "__main__":
    # load data
    file = '/naval_dataset.pkl'
    path = os.getcwd()+file
    with open(path, 'rb') as f:
        train_data, train_label, val_data, val_label = pickle.load(f)
    train_data = np.moveaxis(train_data,2,1)
    val_data = np.moveaxis(val_data,2,1)
    m = np.expand_dims(np.amax(val_data,axis=1),1)
    val_data = np.divide(val_data,m)

    file_w = "/W_best.pkl"
    file_network = "/network_best.pkl"
    path = os.getcwd()+file_w
    with open(path, 'rb') as f:
        a, b, t1, t2, pc, pd, tvar_temporal, tvar_logical_c, tvar_logical_d = pickle.load(f)
    path = os.getcwd()+file_network
    with open(path, 'rb') as f:
        pred, temporal, logical_c, logical_d = pickle.load(f)

    # get formulas
    print('get formulas ------------------')
    formula_t = [] # temporal operator
    formula_time = [] # time interval
    formula_pred = [] # x, y
    formula_sym = [] # >, <
    formula_const = [] # predicate constant
    length = train_data.shape[1]
    w = torch.tensor(range(length), requires_grad=False)
    for k, (predi, Ti) in enumerate(zip(pred, temporal)):
        if temporal[k].tvar>0:
            formula_t.append('F')
        else:
            formula_t.append('G')
        t1, t2 = get_time(temporal[k].time_weight.forward(w))
        formula_time.append([t1,t2])
        if pred[k].dim==0:
            formula_pred.append('x')
        else:
            formula_pred.append('y')
        if pred[k].a>0:
            formula_sym.append('>')
        else:
            formula_sym.append('<')
        const = (pred[k].b)/(pred[k].a)
        formula_const.append(const)

    pcb = STEstimator.apply(pc)
    pdb = STEstimator.apply(pd)
    disjunction_index = torch.where(pdb==1)[0]
    for i in disjunction_index:
        print('{')
        pci = pcb[i,:]
        conjunction_index = torch.where(pci==1)[0]
        for j in conjunction_index:
            print(formula_t[j]+"["+str(formula_time[j][0])+","+str(formula_time[j][1])+"]"+formula_pred[j]+formula_sym[j]+"{:.2f}".format(formula_const[j]))
        print('}')

    
    print('plot formulas ------------------')
    val_nsample = val_data.shape[0]
    pcb = STEstimator.apply(pc)
    pdb = STEstimator.apply(pd)
    nc = 0
    for pci in pcb:
        if torch.sum(pci)>nc:
            nc = int(torch.sum(pci))
    nd = int(torch.sum(pdb))
    fig, axes = plt.subplots(nd,nc,figsize=(5*nc,5*nd))
    disjunction_index = torch.where(pdb==1)[0]
    for i, di in enumerate(disjunction_index):
        pci = pcb[di,:]
        conjunction_index = torch.where(pci==1)[0]
        for j, ci in enumerate(conjunction_index):
            ax = axes[i,j]
            plot_timed_data(ax, val_data, val_label, formula_time[ci], val_nsample)
            t1 = formula_time[ci][0]
            t2 = formula_time[ci][1]
            if formula_t[ci]=='F':
                fcolor ='orange'
            else:
                fcolor ='green'
            const = formula_const[ci].detach().numpy()
            if formula_pred=='x':
                p1 = ax.axvline(x=const, color=fcolor)
            else:
                x = val_data[:,t1:(t2+1),0]
                y = np.ones(x.shape)*const
                p2 = ax.plot(x, y, color=fcolor)
            li = formula_t[ci]+"["+str(t1)+","+str(t2)+"]"+formula_pred[ci]+formula_sym[ci]+"{:.2f}".format(formula_const[ci])
            ax.set_title(li,fontsize=10)
    plt.show()