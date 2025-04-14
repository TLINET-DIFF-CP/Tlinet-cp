import torch
import numpy as np
import pickle
import os
import sys
#sys.path.append('../src')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from neurons import *
from functions import *

def hard_accuracy(x, y, file_w, file_network):
    path = os.getcwd()+file_w
    with open(path, 'rb') as f:
        a, b, t1, t2, pc, pd, tvar_temporal, tvar_logical_c, tvar_logical_d = pickle.load(f)
    path = os.getcwd()+file_network
    with open(path, 'rb') as f:
        pred, temporal, logical_c, logical_d = pickle.load(f)
    nf = len(pred)
    nc = len(logical_c)
    nsample = x.shape[0]
    length = x.shape[1]
    Nmax = Normalization_max(dim=1)
    x = Nmax.forward(x)
    w = torch.tensor(range(length), requires_grad=False)
    r1o = torch.empty((nsample,nf))
    for k, (predi, Ti) in enumerate(zip(pred, temporal)):
        r1 = predi.forward(x)
        t1, t2 = get_time(Ti.time_weight.forward(w))
        if isinstance(t1,list):
            r1o[:,k] = 0
            continue
        rt = r1[:,t1:t2+1]
        if Ti.tvar>0:
            r1o[:,k] = torch.max(rt,1,keepdim=False)[0]
        else:
            r1o[:,k] = torch.min(rt,1,keepdim=False)[0]
    rc = torch.empty((nsample,nc))
    pcb = STEstimator.apply(pc)
    pdb = STEstimator.apply(pd)
    for k, li in enumerate(logical_c):
        if torch.sum(pcb[k,:])==0:
            rc[:,k] = 0
            continue
        rp = r1o[:,pcb[k,:]==1]
        if li.tvar>0:
            rc[:,k] = torch.max(rp,1,keepdim=False)[0]
        else:
            rc[:,k] = torch.min(rp,1,keepdim=False)[0]
    rp = rc[:,pdb[0,:]==1]
    if torch.sum(pdb[0,:])==0:
        R = torch.zeros((nsample))
    elif logical_d.tvar>0:
        R = torch.max(rp,1,keepdim=False)[0]
    else:
        R = torch.min(rp,1,keepdim=False)[0]
    Rl = Clip.apply(R)
    acc = sum(y==Rl)/nsample
    return acc

def network_accuracy(x, y, file_w, file_network):
    path = os.getcwd()+file_w
    with open(path, 'rb') as f:
        a, b, t1, t2, pc, pd, tvar_temporal, tvar_logical_c, tvar_logical_d = pickle.load(f)
    path = os.getcwd()+file_network
    with open(path, 'rb') as f:
        pred, temporal, logical_c, logical_d = pickle.load(f)
    nf = len(pred)
    nc = len(logical_c)
    nsample = x.shape[0]
    length = x.shape[1]
    Nmax = Normalization_max(dim=1)
    x = Nmax.forward(x)
    r1o = torch.empty((nsample,nf))
    for k, (predi, Ti) in enumerate(zip(pred, temporal)):
        r1 = predi.forward(x)
        r1o[:,k] = Ti.forward(r1,padding=False)
    rc = torch.empty((nsample,nc))
    pcb = STEstimator.apply(pc)
    pdb = STEstimator.apply(pd)
    for k, li in enumerate(logical_c):
        rc[:,k] = li.forward(r1o,pcb[k,:],keepdim=False)
    R = logical_d.forward(rc,pdb[0,:],keepdim=False)
    label_R = Clip.apply(R)
    acc = torch.sum((label_R==y))/nsample
    return acc
    
def get_formula_num(p_list):
    ln = len(p_list)
    pi_array = p_list[-1]
    for i in range(ln-1, -1, -1):
        pi_new = []
        pib = STEstimator.apply(pi_array)
        if torch.sum(pib) == 0:
            return torch.tensor(0)
        if torch.sum(torch.isnan(pib)):
            return torch.tensor(0)
        if i == 0:
            continue
        for pi_row in pib:
            one_pos = (torch.where(pi_row==1)[0]).numpy()
            for j in one_pos:
                pi_new.append(p_list[i-1][j:j+1])
        pi_array = torch.cat(pi_new, dim=0)
    num_formula = torch.sum(STEstimator.apply(pi_array))
    return num_formula