import torch
import pickle
import os
import random as rd
from utils import *
import sys
sys.path.append('../src')
from neurons import *

def model_naval(train_data, train_label, val_data, val_label, n_iters, round, avm, variable_based, hyper):
    nsample = train_data.shape[0]
    val_nsample = val_data.shape[0]
    length = train_data.shape[1]
    dim = train_data.shape[2]
    
    # initialize parameters
    nf = 8
    nc = 2
    t1 = torch.randint(0,length-1,(nf,),dtype=torch.float64,requires_grad=True)
    t2 = torch.randint(0,length-1,(nf,),dtype=torch.float64,requires_grad=True)
    a = torch.rand(nf, dtype=torch.float64, requires_grad=True)
    b = torch.rand(nf, dtype=torch.float64, requires_grad=True)
    pc = torch.rand((nc,nf), requires_grad=True)
    pd = torch.rand((1,nc), requires_grad=True)
    tvar_temporal = torch.rand(nf, dtype=torch.float64, requires_grad=True)
    tvar_logical_c = torch.rand(nf, dtype=torch.float64, requires_grad=True)
    tvar_logical_d = torch.rand(nf, dtype=torch.float64, requires_grad=True)
    if variable_based:
        temporal_type = ['temporal' for i in range(nf)]
        logical_in = 'logical'
        logical_out = 'logical'
    else:
        temporal_type = ['F' for i in range(nf//2)] + ['G' for i in range(nf//2)]
        logical_in = 'and'
        logical_out = 'or'
    dimension = []
    for i in range(nf//2):
        dimension.extend([0,1]) 
    tau = torch.tensor(0.1, requires_grad=False)
    eps = torch.rand((1), requires_grad=True)
    
    # load file
    file = '/initialization/case'+str(round)+'.pkl'
    path = os.getcwd()+file
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            a, b, t1, t2, pc, pd, tvar_temporal, tvar_logical_c, tvar_logical_d = pickle.load(f)
    else:
        with open(path, 'wb') as f:
            pickle.dump([a, b, t1, t2, pc, pd, tvar_temporal, tvar_logical_c, tvar_logical_d], f)

    # initialize neural network structure
    pred = []
    for i in range(nf):
        pred.append(Predicate(a[i],b[i],dim=dimension[i]))
    temporal = []
    for i in range(nf):
        temporal.append(TemporalOperator(temporal_type[i],tau,t1[i],t2[i],avm=False,beta=2.5,tvar=tvar_temporal[i]))
    logical_c = []
    for i in range(nc):
        logical_c.append(LogicalOperator(logical_in,dim=1,avm=avm,beta=2.5,tvar=tvar_logical_c[i]))
    logical_d = LogicalOperator(logical_out,dim=1,avm=avm,beta=2.5,tvar=tvar_logical_d[i])

    # initialize funcitions
    relu = torch.nn.ReLU()
    STE= STEstimator.apply
    clip = Clip.apply
    penalty = Bimodal_reg([0.1,0.1])
    Nmax = Normalization_max(dim=1)

    optimizer1 = torch.optim.Adam([a,b,t1,t2], lr=0.1)
    optimizer2 = torch.optim.Adam([pc,pd], lr=0.1)
    optimizer3 = torch.optim.Adam([tvar_temporal,tvar_logical_c,tvar_logical_d], lr=0.1)
    optimizer4 = torch.optim.Adam([eps], lr=0.001)

    batch_size = 64
    acc = 0
    acc_bnl = 0
    acc_best = 0
    acc_bnl_best = 0

    for epoch in range(1,n_iters):
        rand_num = rd.sample(range(0,nsample),batch_size)
        x = train_data[rand_num,:,:]
        y = train_label[rand_num]
        x = Nmax.forward(x)
    
        r1o = torch.empty((batch_size,nf))
        for k, (predi, Ti) in enumerate(zip(pred, temporal)):
            r1 = predi.forward(x)
            r1o[:,k] = Ti.forward(r1,padding=False)

        rc = torch.empty((batch_size,nc))
        for k, li in enumerate(logical_c):
            rc[:,k] = li.forward(r1o,pc[k,:],keepdim=False)

        R = logical_d.forward(rc,pd[0,:],keepdim=False)

        if epoch >= 2000:
            penalty.alpha_list = [1,2]
        reg = penalty.get_reg([pc,pd])
        reg_a = 0.01*torch.sum(torch.square(a))
        l = torch.sum(relu(eps-y*R)) + torch.sum(1/eps) + reg + reg_a 
        l.backward()

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        
        with torch.no_grad():
            pc[pc<=0] = 0
            pc[pc>=1] = 1
            pd[pd<=0] = 0
            pd[pd>=1] = 1
            if torch.sum(pd)==0:
                pd = torch.rand(pd.shape)
            if torch.sum(pc)==0:
                pc = torch.rand(pc.shape)
        with torch.no_grad():
            t1[t1<0] = 0
            t2[t2<0] = 0
            t1[t1>length-1] = length-1
            t2[t2>length-1] = length-1
            for k, t in enumerate(t1):
                if t>t2[k]:
                    t1[k] = t2[k]-1
        with torch.no_grad():
            tvar_temporal[tvar_temporal<-1] = -1
            tvar_temporal[tvar_temporal>1] = 1
            tvar_logical_c[tvar_logical_c<-1] = -1
            tvar_logical_c[tvar_logical_c>1] = 1
            tvar_logical_d[tvar_logical_d<-1] = -1
            tvar_logical_d[tvar_logical_d>1] = 1
        with torch.no_grad():
            eps[eps<0] = 1e-5

        if epoch % 100 ==0:
            x = val_data
            y = val_label
            x = Nmax.forward(x)
            r1o = torch.empty((val_nsample,nf))
            for k, (predi, Ti) in enumerate(zip(pred, temporal)):
                r1 = predi.forward(x)
                r1o[:,k] = Ti.forward(r1,padding=False)
            rc = torch.empty((val_nsample,nc))
            rcb = torch.empty((val_nsample,nc))
            pcb = STE(pc)
            pdb = STE(pd)
            for k, li in enumerate(logical_c):
                rc[:,k] = li.forward(r1o,pc[k,:],keepdim=False)
                rcb[:,k] = li.forward(r1o,pcb[k,:],keepdim=False)
            R = logical_d.forward(rc,pd[0,:],keepdim=False)
            Rb = logical_d.forward(rcb,pdb[0,:],keepdim=False)
            label_R = clip(R)
            label_Rb = clip(Rb)
            acc = torch.sum((label_R==y))/val_nsample
            acc_bnl = torch.sum((label_Rb==y))/val_nsample
            num_formula = torch.sum(pcb[(pdb==1)[0],:])
            print('epoch {epoch}, loss = {l}, accuracy = {acc}, accuracy_bernoulli = {accb}'.format(epoch=epoch,l=l,acc=acc,accb=acc_bnl))
            if acc_bnl>=acc_best:
                acc_best = acc_bnl
                f = open('W_crossval.pkl', 'wb')
                pickle.dump([a, b, t1, t2, pc, pd, tvar_temporal, tvar_logical_c, tvar_logical_d], f)
                f.close()
                f = open('network_crossval.pkl', 'wb')
                pickle.dump([pred, temporal, logical_c, logical_d], f)
                f.close()
    return acc_best, num_formula

