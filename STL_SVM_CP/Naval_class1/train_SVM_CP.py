import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import sin, cos, pi
import numpy as np
import torch.nn.functional as F
import pickle
import random
from random import shuffle
import sys
sys.path.insert(0, '..')
from neurons_prob import *
from functions import *
from fast_soft_sort.pytorch_ops import *



def NormalizeDataset(train_data, cal_data, pred_data):
    min_train = np.min(train_data, axis=(0, 1))
    max_train = np.max(train_data, axis=(0, 1))
    train_data[:,:,0] = (train_data[:,:,0] - min_train[0]) / (max_train[0] - min_train[0])
    train_data[:,:,1] = (train_data[:,:,1] - min_train[1]) / (max_train[1] - min_train[1])
    cal_data[:,:,0] = (cal_data[:,:,0] - min_train[0]) / (max_train[0] - min_train[0])
    cal_data[:,:,1] = (cal_data[:,:,1] - min_train[1]) / (max_train[1] - min_train[1])
    pred_data[:,:,0] = (pred_data[:,:,0] - min_train[0]) / (max_train[0] - min_train[0])
    pred_data[:,:,1] = (pred_data[:,:,1] - min_train[1]) / (max_train[1] - min_train[1])
    return train_data, cal_data, pred_data



class SVMConformalTraining(torch.nn.Module):
    def __init__(self, alpha, T, beta):
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.beta = beta

    def forward(self, x, y):
        n = x.shape[0]//2
        x_cal = x[:n]
        y_cal = y[:n]
        x_pred = x[n:]
        y_pred = y[n:]
        # calibrate
        thresh = self.threshold(x_cal, y_cal, dim=0) # threshold [-a, a]
        cal_score = self.score(x_cal*y_cal, thresh)  # scores of calibration set 
        # prediction
        score_pos = self.score(x_pred, thresh) # score for x_pred belongs to class 1
        score_neg = self.score(-1*x_pred, thresh) # score for x_pred belongs to class -1
        out = 0
        for i in range(n):
            C_pos = 1/(1+torch.exp(-(cal_score-score_pos[i]+1/self.T)/self.T))
            prob_pos = torch.sum(C_pos)/(n+1)
            C_neg = 1/(1+torch.exp(-(cal_score-score_neg[i]+1/self.T)/self.T))
            prob_neg = torch.sum(C_neg)/(n+1)
            out += y_pred[i]*prob_pos - y_pred[i]*prob_neg # if y==1, prob = prob_pos-prob_neg, if y == -1, prob = prob_neg-prob_pos
        return out/n
    
    def threshold(self, x, y, dim):
        relu = nn.ReLU()
        x_new = relu(x*y)
        # exp_neg_x = torch.exp(-self.beta*x_new)
        # threshold = torch.sum(x_new*exp_neg_x, dim=dim) / torch.sum(exp_neg_x, dim=dim)
        threshold = torch.min(x_new[x_new>0])
        return threshold
    
    def score(self, x, threshold, sharpness=10.0, scale_large=20.0):
        sigmoid = torch.sigmoid
        left = sigmoid(-sharpness * (x - threshold)) # ~1 if x < threshold, ~0 if x > threshold
        right = sigmoid(sharpness * (x + threshold)) # ~1 if x > -threshold, ~0 if x < -threshold
        middle = left*right # ~1 in (-threshold, threshold), ~0 outside
        large_penalty = scale_large * sigmoid(-sharpness * (x + threshold)) # x < -threshold → sigmoid(-sharp * (x + threshold)) ≈ 1 → large value
        return middle + large_penalty
    
    def predict(self, x_cal, y_cal, x_pred, y_pred):
        with torch.no_grad():
            relu = nn.ReLU()
            n = x_cal.shape[0]
            scale_large = 20
            thresh = self.threshold(x_cal, y_cal, dim=0)
            cal_score = relu(-(x_cal*y_cal-thresh))
            cal_score[cal_score>2*thresh] = scale_large
            # cal_score = torch.where(torch.logical_and(cal_score>0,cal_score<2*thresh), torch.tensor(2.0, dtype=cal_score.dtype), cal_score)
            cal_score[torch.logical_and(cal_score>0,cal_score<=2*thresh)] = 1
            min_prob = 1
            ave_prob = 0
            y_pred_cp = []
            for xi in x_pred:
                #score for class 1
                if xi>thresh:
                    spos = 0
                elif xi<=-thresh:
                    spos = 20
                else:
                    spos = 1
                #score for class -1
                if xi<-thresh:
                    sneg = 0
                elif xi>thresh:
                    sneg = 20
                else:
                    sneg = 1
                prob_pos = sum((cal_score>=spos).int())/(n+1)
                prob_neg = sum((cal_score>=sneg).int())/(n+1)
                if prob_pos>prob_neg:
                    prob = prob_pos-prob_neg
                    min_prob = min(prob,min_prob)
                    ave_prob += prob
                    y_pred_cp.append(1)
                else:
                    prob = prob_neg-prob_pos
                    min_prob = min(prob,min_prob)
                    ave_prob += prob
                    y_pred_cp.append(-1)
                # print(prob)
            acc = torch.sum(y_pred==torch.tensor(y_pred_cp))/x_pred.shape[0]
            return acc, min_prob, ave_prob/x_pred.shape[0]



class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


    
class STL_CP(torch.nn.Module):
    def __init__(self, nf, nc, length, weight_reg, weight_a):
        super().__init__()
        self.nf = nf
        self.nc = nc
        self.length = length
        self.t1 = torch.nn.Parameter(torch.randint(0,length,(nf,),dtype=torch.float64,requires_grad=True))
        self.t2 = torch.nn.Parameter(torch.randint(0,length,(nf,),dtype=torch.float64,requires_grad=True))
        a = [1,1,-1,-1]*(nf//2)
        self.a = torch.tensor(a, dtype=torch.float64, requires_grad=False)
        self.b = torch.nn.Parameter(torch.rand((nf,), dtype=torch.float64, requires_grad=True))
        self.p = torch.nn.Parameter(torch.rand((1,nf), dtype=torch.float64, requires_grad=True))
        self.tvar_temporal = torch.nn.Parameter(torch.rand(nf, dtype=torch.float64, requires_grad=True))
        self.tvar_logical = torch.nn.Parameter(torch.rand(1, dtype=torch.float64, requires_grad=True))
        variable_based = True
        if variable_based:
            temporal_type = ['temporal' for i in range(nf)]
            logical_type = 'logical'
        else:
            temporal_type = ['F' for i in range(nf//2)] + ['G' for i in range(nf//2)]
            logical_type = 'and'
        self.tau = torch.tensor(1, requires_grad=False)
        self.pred = []
        for i in range(nf):
            self.pred.append(Predicate(self.a[i],self.b[i],dim=i%2))
        self.temporal = []
        for i in range(nf):
            self.temporal.append(TemporalOperator(temporal_type[i],self.tau,self.t1[i],self.t2[i],beta=5,h=1,type_var=self.tvar_temporal[i]))
        self.logical = LogicalOperator(logical_type,dim=1,avm=True,beta=2,type_var=self.tvar_logical)
        self.penalty = Bimodal_reg(weight_reg)
        self.weight_a = weight_a
        self.eps = torch.nn.Parameter(torch.rand(1, dtype=torch.float64, requires_grad=True))

    def forward(self, x):
        with torch.no_grad():
            for p in [self.p]:
                p.clamp_(0, 1)
                if torch.all(p<0.5):
                    p_new = torch.rand(p.shape)
                    p.data.copy_(p_new)
            self.t1[self.t1<0] = 0
            self.t2[self.t2<0] = 0
            self.t1[self.t1>self.length-1] = self.length-1
            self.t2[self.t2>self.length-1] = self.length-1
            self.t1[self.t1>=self.t2-1] = self.t2[self.t1>=self.t2-1]-1
            for tvar in [self.tvar_temporal,self.tvar_logical]:
                tvar.clamp_(0, 1)
                tvar.clamp_(0, 1)
            self.eps[self.eps<0] = 1e-5

        batch_size = x.shape[0]
        r1 = torch.empty((batch_size,self.nf))
        for k, (predi, Ti) in enumerate(zip(self.pred, self.temporal)):
            rp = predi.forward(x)
            r1[:,k] = Ti.forward(rp,padding=False)
        r = self.logical.forward(r1,self.p[0,:],keepdim=False)
        out = r
        reg = self.penalty.get_reg([self.p,self.tvar_temporal,self.tvar_logical])
        reg_a = self.weight_a*torch.mean(torch.square(torch.square(self.a)-1))
        return out, reg, reg_a
    
    def accuracy_avm(self, x, y):
        with torch.no_grad():
            self.register_buffer("tvar_temporal_buffer", self.tvar_temporal.detach().clone())
            self.register_buffer("tvar_logical_buffer", self.tvar_logical.detach().clone())
            self.tvar_temporal[self.tvar_temporal<0.5] = 0
            self.tvar_temporal[self.tvar_temporal>0.5] = 1
            self.tvar_logical[self.tvar_logical<0.5] = 0
            self.tvar_logical[self.tvar_logical>0.5] = 1
            batch_size = x.shape[0]
            r1 = torch.empty((batch_size,self.nf))
            for k, (predi, Ti) in enumerate(zip(self.pred, self.temporal)):
                rp = predi.forward(x)
                r1[:,k] = Ti.forward(rp,padding=False)
            p1b = STEstimator.apply(self.p)
            r = self.logical.forward(r1,p1b[0,:],keepdim=False)
            y_pred = Clip.apply(r)
            acc = torch.sum((y_pred==y))/batch_size
            self.tvar_temporal.data = self.tvar_temporal_buffer.detach().clone()
            self.tvar_logical.data = self.tvar_logical_buffer.detach().clone()
            del self.tvar_temporal_buffer, self.tvar_logical_buffer
            return acc
    
    def translate_formula(self, val_data=False, val_label=False):
        formula_T1 = [] # temporal operator
        formula_time = [] # time interval
        formula_const = [] # x, y
        formula_xy = []
        formula_sym = [] # >, <
        str_list = ['x','y']
        w = torch.tensor(range(self.length), requires_grad=False)
        for k, (predi, Ti1) in enumerate(zip(self.pred, self.temporal)):
            if Ti1.type_var>0.5:
                formula_T1.append('F')
            else:
                formula_T1.append('G')
            t11, t12 = get_time(Ti1.time_weight.forward(w,self.tau,self.t1[k],self.t2[k]))
            formula_time.append([t11,t12])
            d = predi.dim
            formula_xy.append(str_list[d])
            if predi.a>0:
                formula_sym.append('>')
                formula_const.append(predi.b.item()/predi.a.item())
            else:
                formula_sym.append('<')
                formula_const.append(predi.b.item()/predi.a.item())
        p1b = STEstimator.apply(self.p)
        # p1b = Clip.apply(self.p)
        logical_index = torch.where(torch.squeeze(p1b)==1)[0]
        for indexc, j in enumerate(logical_index):
            if indexc > 0 and len(logical_index)>1:
                if self.logical.type_var>0.5:
                    print(' or ')
                else:
                    print(' and ')
            print(formula_T1[j]+"["+str(formula_time[j][0])+","+str(formula_time[j][1])+"]"
                +formula_xy[j]+formula_sym[j]+'{:.2f}'.format(formula_const[j]))
            
    def extract_formula(self, val_data, val_label):
        acc = self.accuracy_avm(val_data, val_label)
        with torch.no_grad():
            p_buffer = torch.clone(self.p)
            for i in range(self.p.shape[1]):
                if self.p[0,i]<0.5:
                    continue
                else:
                    self.p[0,i] = 0
                    acci = self.accuracy_avm(val_data, val_label)
                    if acci == acc:
                        continue
                    else:
                        self.p[0,i] = 1
        return p_buffer
    
    def denormalize_predicate(self, min_train, max_train):
        with torch.no_grad():
            for k, predi in enumerate(self.pred):
                if predi.dim==0:
                    a = predi.a.item()
                    b = predi.b.item()
                    b_new = a*min_train[0]+b*(max_train[0]-min_train[0])
                    predi.b = torch.tensor(b_new, dtype=torch.float64, requires_grad=False)
                if predi.dim==1:
                    a = predi.a.item()
                    b = predi.b.item()
                    b_new = a*min_train[1]+b*(max_train[1]-min_train[1])
                    predi.b = torch.tensor(b_new, dtype=torch.float64, requires_grad=False)



# -----------Train inference network---------------------------------------
if __name__ == "__main__":
    random.seed(2)
    with open('naval.pkl', 'rb') as f:
        train_data, train_label, cal_data, cal_label, pred_data, pred_label = pickle.load(f)
    train_data, cal_data, pred_data = NormalizeDataset(train_data, cal_data, pred_data)
    train_data = torch.tensor(train_data, requires_grad=False)
    train_label = torch.tensor(train_label, requires_grad=False)
    cal_data = torch.tensor(cal_data, requires_grad=False)
    cal_label = torch.tensor(cal_label, requires_grad=False)
    pred_data = torch.tensor(pred_data, requires_grad=False)
    pred_label = torch.tensor(pred_label, requires_grad=False)

    nsample = train_data.shape[0]
    length = train_data.shape[1]
    dim = train_data.shape[2]
    nf = dim*4
    nc = 2
    weight_reg = [1e-2,1e-1,1e-1]
    weight_a = 1e-3
    weight_eps = 1e-3
    

    dataset = CustomDataset(train_data, train_label)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    relu = torch.nn.ReLU()
    stl = STL_CP(nf,nc,length,weight_reg,weight_a)
    stl_optimizer = torch.optim.Adam(stl.parameters(), lr=1e-2)
    stl.train()

    alpha = 0.9
    T = 2
    beta = 4
    CP = SVMConformalTraining(alpha,T,beta)

    ################## unsupervised training ##################
    epoch_num = 2000
    stl.train()
    acc_best = 0
    for epoch in range(epoch_num):
        for batch_idx, (data_batch, labels_batch) in enumerate(dataloader):
            stl_optimizer.zero_grad()
            r, reg, reg_a = stl(data_batch)
            lcp = CP.forward(r, labels_batch)
            # loss_inf = torch.mean(torch.exp(-r*labels_batch)) + lcp + reg
            # loss_inf = torch.mean(relu(stl.eps-r*labels_batch)) - weight_eps*stl.eps + 0.1*reg
            # loss_inf = torch.mean(relu(stl.eps-r*labels_batch)) - weight_eps*stl.eps + 0.1*reg + lcp
            # loss_inf = torch.mean(relu(stl.eps-r*labels_batch)) - weight_eps*stl.eps - lcp
            loss_inf = -lcp
            # loss_inf = lcp
            if reg == 0:
                stop = 1
            loss_inf.backward()
            if stl.p.grad == None:
                stop = 1
            stl_optimizer.step()
        
        if (epoch+1) % 10 ==0:
            with torch.no_grad():
                r_cal, _, _ = stl.forward(cal_data)
                r_pred, _, _ = stl.forward(pred_data)
                cp_acc, cp_min_prob, cp_ave_prob = CP.predict(r_cal, cal_label, r_pred, pred_label)
                # cp_acc = 0
                label_predicted = Clip.apply(r_pred)
                acc = torch.sum((label_predicted==pred_label))/pred_data.shape[0]
                accb = stl.accuracy_avm(pred_data, pred_label)
            print('epoch = {:3d}, pred_robustness_accuracy = {:.2f}, cp accuracy = {:.2f} with min probability {:.2f} and average probability {:.2f}, pred_avm_accuracy = {:.2f}'.format(epoch+1, acc, cp_acc, cp_min_prob, cp_ave_prob, accb))
            if acc>=acc_best:
                acc_best = acc
                stl.translate_formula()
                torch.save(stl.state_dict(), f'naval_model.pth')
        stl_optimizer.zero_grad()
        
    print('best accuracy is ',acc_best)