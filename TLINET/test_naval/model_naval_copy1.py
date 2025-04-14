import torch
import pickle
import os
import random as rd
from utils import *
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from neurons import *
import torch.nn.functional as F 

def smooth_quantile(scores, quantile, temperature):
    sorted_scores, _ = torch.sort(scores)
    N = scores.size(0)
    idx = torch.linspace(-1, 1, steps=N, device=scores.device, dtype=scores.dtype)
    weights = F.softmax(idx / temperature, dim=0)
    tau = torch.sum(weights * sorted_scores)
    return tau

def model_naval(train_data, train_label, val_data, val_label, n_iters, round, avm, variable_based):
    nsample = train_data.shape[0]
    val_nsample = val_data.shape[0]
    length = train_data.shape[1]
    dim = train_data.shape[2]
    
   
    nf = 8
    nc = 2
    t1 = torch.randint(0, length-1, (nf,), dtype=torch.float64, requires_grad=True)
    t2 = torch.randint(0, length-1, (nf,), dtype=torch.float64, requires_grad=True)
    a = torch.rand(nf, dtype=torch.float64, requires_grad=True)
    b = torch.rand(nf, dtype=torch.float64, requires_grad=True)
    pc = torch.rand((nc, nf), requires_grad=True)
    pd = torch.rand((1, nc), requires_grad=True)
    tvar_temporal = torch.rand(nf, dtype=torch.float64, requires_grad=True)
    tvar_logical_c = torch.rand(nc, dtype=torch.float64, requires_grad=True)
    tvar_logical_d = torch.rand(1, dtype=torch.float64, requires_grad=True)
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
        dimension.extend([0, 1]) 
    tau = torch.tensor(0.1, requires_grad=False)
    eps = torch.rand((1), requires_grad=True)
    
  
    file = '/initialization/case' + str(round) + '.pkl'
    path = os.getcwd() + file
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            a, b, t1, t2, pc, pd, tvar_temporal, tvar_logical_c, tvar_logical_d = pickle.load(f)
    else:
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, 'initialization', f'case{round}.pkl')
        with open(path, 'wb') as f:
            pickle.dump([a, b, t1, t2, pc, pd, tvar_temporal, tvar_logical_c, tvar_logical_d], f)
    
    pred = []
    for i in range(nf):
        pred.append(Predicate(a[i], b[i], dim=dimension[i]))
    temporal = []
    for i in range(nf):
        temporal.append(TemporalOperator(temporal_type[i], tau, t1[i], t2[i], avm=False, beta=2.5, tvar=tvar_temporal[i]))
    logical_c = []
    for i in range(nc):
        logical_c.append(LogicalOperator(logical_in, dim=1, avm=avm, beta=2.5, tvar=tvar_logical_c[i]))
    logical_d = LogicalOperator(logical_out, dim=1, avm=avm, beta=2.5, tvar=tvar_logical_d)
    
    relu = torch.nn.ReLU()
    STE = STEstimator.apply
    clip = Clip.apply
    penalty = Bimodal_reg([0.1, 0.1])
    Nmax = Normalization_max(dim=1)
    
    optimizer1 = torch.optim.Adam([a, b, t1, t2], lr=0.1)
    optimizer2 = torch.optim.Adam([pc, pd], lr=0.1)
    optimizer3 = torch.optim.Adam([tvar_temporal, tvar_logical_c, tvar_logical_d], lr=0.1)
    optimizer4 = torch.optim.Adam([eps], lr=0.001)
    
    batch_size = 64
    acc_best = 0
    
    alpha = 0.01              
    T_quantile = 0.5          
    T_pred = 0.3         
    lambda_cp = 0.5         
    lambda_ste = 0.2         
    warmup_epochs = 2000
    
    for epoch in range(1, n_iters):
        rand_num = rd.sample(range(0, nsample), batch_size)
        x = train_data[rand_num, :, :]
        y = train_label[rand_num]
        x = Nmax.forward(x)
        
        half = batch_size // 2
        x_cal = x[:half, :, :]
        x_pred = x[half:, :, :]
        y_pred = y[half:]
        
        r1o_cal = torch.empty((half, nf), dtype=torch.float64)
        for k, (predi, Ti) in enumerate(zip(pred, temporal)):
            r1 = predi.forward(x_cal)
            r1o_cal[:, k] = Ti.forward(r1, padding=False)
        rc_cal = torch.empty((half, nc), dtype=torch.float64)
        for k, li in enumerate(logical_c):
            rc_cal[:, k] = li.forward(r1o_cal, pc[k, :], keepdim=False)
        R_cal = logical_d.forward(rc_cal, pd[0, :], keepdim=False)
        
        tau_cal = smooth_quantile(R_cal, alpha * (1 + 1/half), T_quantile)
        
        r1o_pred = torch.empty((batch_size - half, nf), dtype=torch.float64)
        for k, (predi, Ti) in enumerate(zip(pred, temporal)):
            r1 = predi.forward(x_pred)
            r1o_pred[:, k] = Ti.forward(r1, padding=False)
        rc_pred = torch.empty((batch_size - half, nc), dtype=torch.float64)
        for k, li in enumerate(logical_c):
            rc_pred[:, k] = li.forward(r1o_pred, pc[k, :], keepdim=False)
        R_pred = logical_d.forward(rc_pred, pd[0, :], keepdim=False)
        
        p_pos = torch.sigmoid((R_pred - tau_cal) / T_pred)
        p_neg = torch.sigmoid((-R_pred - tau_cal) / T_pred)
        soft_confidence = torch.stack([p_neg, p_pos], dim=1)
        
        target = torch.zeros_like(soft_confidence, dtype=torch.float64)
        for j in range(y_pred.size(0)):
            if y_pred[j] == 1:
                target[j, 1] = 1.0
            else:
                target[j, 0] = 1.0
        
        loss_soft = F.binary_cross_entropy(soft_confidence, target)
        set_size = p_pos + p_neg
        loss_size = torch.mean(F.relu(set_size - 1))
        
 
        pcb = STE(pc)
        pdb = STE(pd)
        rcb_cal = torch.empty((half, nc), dtype=torch.float64)
        for k, li in enumerate(logical_c):
            rcb_cal[:, k] = li.forward(r1o_cal, pcb[k, :], keepdim=False)
        Rb = logical_d.forward(rcb_cal, pdb[0, :], keepdim=False)
        loss_ste = F.mse_loss(Rb, R_cal.detach())
     
        reg = penalty.get_reg([pc, pd])
        reg_a = 0.01 * torch.sum(torch.square(a))
        
        if epoch < warmup_epochs:
            total_loss = loss_soft + reg + reg_a
        else:
            total_loss = loss_soft + lambda_cp * loss_size + lambda_ste * loss_ste + reg + reg_a
        
        total_loss.backward()
        
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        
        with torch.no_grad():
            pc[pc <= 0] = 0
            pc[pc >= 1] = 1
            pd[pd <= 0] = 0
            pd[pd >= 1] = 1
            if torch.sum(pd) == 0:
                pd = torch.rand(pd.shape)
            if torch.sum(pc) == 0:
                pc = torch.rand(pc.shape)
        with torch.no_grad():
            t1[t1 < 0] = 0
            t2[t2 < 0] = 0
            t1[t1 > length-1] = length-1
            t2[t2 > length-1] = length-1
            for k, t in enumerate(t1):
                if t > t2[k]:
                    t1[k] = t2[k] - 1
        with torch.no_grad():
            tvar_temporal[tvar_temporal < -1] = -1
            tvar_temporal[tvar_temporal > 1] = 1
            tvar_logical_c[tvar_logical_c < -1] = -1
            tvar_logical_c[tvar_logical_c > 1] = 1
            tvar_logical_d[tvar_logical_d < -1] = -1
            tvar_logical_d[tvar_logical_d > 1] = 1
        with torch.no_grad():
            eps[eps < 0] = 1e-5
        
        if epoch % 100 == 0:
            x_val = val_data
            y_val = val_label
            x_val = Nmax.forward(x_val)
            r1o_val = torch.empty((val_nsample, nf), dtype=torch.float64)
            for k, (predi, Ti) in enumerate(zip(pred, temporal)):
                r1 = predi.forward(x_val)
                r1o_val[:, k] = Ti.forward(r1, padding=False)
            rc_val = torch.empty((val_nsample, nc), dtype=torch.float64)
            rcb_val = torch.empty((val_nsample, nc), dtype=torch.float64)
            pcb = STE(pc)
            pdb = STE(pd)
            for k, li in enumerate(logical_c):
                rc_val[:, k] = li.forward(r1o_val, pc[k, :], keepdim=False)
                rcb_val[:, k] = li.forward(r1o_val, pcb[k, :], keepdim=False)
            R_val = logical_d.forward(rc_val, pd[0, :], keepdim=False)
            Rb_val = logical_d.forward(rcb_val, pdb[0, :], keepdim=False)
            label_R = clip(R_val)
            label_Rb = clip(Rb_val)
            val_acc = torch.sum((label_R == y_val)) / val_nsample
            val_acc_bnl = torch.sum((label_Rb == y_val)) / val_nsample
            num_formula_val = torch.sum(pcb[(pdb == 1)[0], :])
            print('epoch {epoch}, loss = {loss:.4f}, accuracy = {acc:.8f}, accuracy_bernoulli = {accb:.8f}'.format(
                epoch=epoch, loss=total_loss.item(), acc=val_acc.item(), accb=val_acc_bnl.item()))
            if val_acc_bnl >= acc_best:
                acc_best = val_acc_bnl
                with open('W_best.pkl', 'wb') as f:
                    pickle.dump([a, b, t1, t2, pc, pd, tvar_temporal, tvar_logical_c, tvar_logical_d], f)
                with open('network_best.pkl', 'wb') as f:
                    pickle.dump([pred, temporal, logical_c, logical_d], f)
    
    return acc_best, num_formula_val
