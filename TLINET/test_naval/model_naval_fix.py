
import torch
import pickle
import os
import random as rd
from utils import *
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from neurons import *
import torch.nn.functional as F

def svm_score(x, threshold, sharpness=9.0, scale_large=0.5):
    sigmoid = torch.sigmoid
    left  = sigmoid(-sharpness * (x - threshold))
    right = sigmoid( sharpness * (x + threshold))
    middle = left * right
    large_penalty = scale_large * sigmoid(-sharpness * (x + threshold))
    return middle + large_penalty


def model_naval(train_data, train_label,
                cal_data, cal_label,
                pred_data, pred_label,
                n_iters, round, avm, variable_based):
    device = train_data.device
    train_label = train_label.to(device)
    cal_label   = cal_label.to(device)

    nsample, length, dim = train_data.shape
    nf, nc = 8, 2

    t1 = torch.randint(0, length-1, (nf,), dtype=torch.float64, requires_grad=True, device=device)
    t2 = torch.randint(0, length-1, (nf,), dtype=torch.float64, requires_grad=True, device=device)
    a  = torch.rand(nf, dtype=torch.float64, requires_grad=True, device=device)
    b  = torch.rand(nf, dtype=torch.float64, requires_grad=True, device=device)
    pc = torch.rand((nc, nf), dtype=torch.float64, requires_grad=True, device=device)
    pd = torch.rand((1, nc), dtype=torch.float64, requires_grad=True, device=device)
    tvar_temporal  = torch.rand(nf, dtype=torch.float64, requires_grad=True, device=device)
    tvar_logical_c = torch.rand(nc, dtype=torch.float64, requires_grad=True, device=device)
    tvar_logical_d = torch.rand(1, dtype=torch.float64, requires_grad=True, device=device)

    if variable_based:
        temporal_type = ['temporal']*nf
        logical_in, logical_out = 'logical', 'logical'
    else:
        temporal_type = ['F']*(nf//2) + ['G']*(nf//2)
        logical_in,   logical_out = 'and', 'or'
    dimension = [i%2 for i in range(nf)] 

    predicates = [Predicate(a[i], b[i], dim=dimension[i]) for i in range(nf)]
    tau0 = torch.tensor(0.1, device=device) 
    temporals  = [TemporalOperator(temporal_type[i], tau0,
                    t1[i], t2[i], avm=False, beta=2.5, tvar=tvar_temporal[i])
                  for i in range(nf)]
    logical_cs = [LogicalOperator(logical_in,  dim=1, avm=avm, beta=2.5, tvar=tvar_logical_c[i])
                  for i in range(nc)]
    logical_d  = LogicalOperator(logical_out, dim=1, avm=avm, beta=2.5, tvar=tvar_logical_d)

    Nmax = Normalization_max(dim=1)
    optimizer1 = torch.optim.Adam([a, b, t1, t2], lr=0.01)
    optimizer2 = torch.optim.Adam([pc, pd],       lr=0.01)
    optimizer3 = torch.optim.Adam([tvar_temporal, tvar_logical_c, tvar_logical_d], lr=0.01)

    batch_size    = 64
    half          = batch_size // 2
    alpha         = 0.05
    T_pred        = 0.1   
    lambda_cp     = 0.2   
    lambda_ste    = 0.1  
    warmup_epochs = 2000
    acc_best      = 0.0
    num_formula   = 0.0


    with torch.no_grad():
        x_cal_full = Nmax.forward(cal_data)
        r1o = torch.stack([ tmp.forward(pr.forward(x_cal_full), padding=False)
                            for pr, tmp in zip(predicates, temporals) ], dim=1)
        rc_full = torch.stack([ lc.forward(r1o, pc[k], keepdim=False)
                                for k, lc in enumerate(logical_cs) ], dim=1)
        R_cal_full = logical_d.forward(rc_full, pd[0], keepdim=False)
        signed = R_cal_full * cal_label 
        x_new  = F.relu(signed)         
        thresh = x_new[x_new>0].min() 

        sp_cal = svm_score( R_cal_full, thresh )
        sn_cal = svm_score(-R_cal_full, thresh )
        nonconf_cal = 1.0 - 0.5*(sp_cal + sn_cal)

        mask_pos = (cal_label == 1)
        m_pos    = mask_pos.sum().item()
        level_pos = 1.0 - alpha * (1.0 + 1.0/m_pos)
        tau_pos  = torch.quantile(nonconf_cal[mask_pos], level_pos)

        mask_neg = (cal_label == -1)
        m_neg    = mask_neg.sum().item()
        level_neg = 1.0 - alpha * (1.0 + 1.0/m_neg)
        tau_neg  = torch.quantile(nonconf_cal[mask_neg], level_neg)
    for epoch in range(1, n_iters):
        idx = rd.sample(range(nsample), batch_size)
        x_batch = Nmax.forward(train_data[idx].to(device))
        y_batch = train_label[idx].to(device)
        x_cal   = x_batch[:half]
        x_pred  = x_batch[half:]
        y_pred  = y_batch[half:]

        r1o_cal = torch.stack([ tmp.forward(pr.forward(x_cal), padding=False)
                                for pr, tmp in zip(predicates, temporals) ], dim=1)
        rc_cal  = torch.stack([ lc.forward(r1o_cal, pc[k], keepdim=False)
                                for k, lc in enumerate(logical_cs) ], dim=1)
        R_cal   = logical_d.forward(rc_cal, pd[0], keepdim=False)

        r1o_pred = torch.stack([ tmp.forward(pr.forward(x_pred), padding=False)
                                 for pr, tmp in zip(predicates, temporals) ], dim=1)
        rc_pred  = torch.stack([ lc.forward(r1o_pred, pc[k], keepdim=False)
                                 for k, lc in enumerate(logical_cs) ], dim=1)
        R_pred   = logical_d.forward(rc_pred, pd[0], keepdim=False)

        logits   = torch.stack([(-R_pred - tau_neg)/T_pred,
                                 ( R_pred - tau_pos)/T_pred], dim=1)
        target_i = ((y_pred + 1)//2).long()
        loss_soft= F.cross_entropy(logits, target_i)

        sp_pred = svm_score( R_pred,  tau_pos )
        sn_pred = svm_score(-R_pred,  tau_neg )
        nonconf_pred = 1.0 - 0.5*(sp_pred + sn_pred)
        loss_cp = nonconf_pred.mean()

        pcb = STEstimator.apply(pc)
        pdb = STEstimator.apply(pd)
        rcb_cal = torch.stack([ lc.forward(r1o_cal, pcb[k], keepdim=False)
                                for k, lc in enumerate(logical_cs) ], dim=1)
        Rb = logical_d.forward(rcb_cal, pdb[0], keepdim=False)
        loss_ste = F.mse_loss(Rb, R_cal.detach())

        reg1 = Bimodal_reg([0.1,0.1]).get_reg([pc, pd])
        reg2 = 0.01 * torch.sum(torch.square(a))

        if epoch < warmup_epochs:
            loss = loss_soft + reg1 + reg2
        else:
            loss = (loss_soft
                    + lambda_cp  * loss_cp
                    + lambda_ste * loss_ste
                    + reg1 + reg2)
 
        optimizer1.zero_grad(); optimizer2.zero_grad(); optimizer3.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [a, b, t1, t2, pc, pd, tvar_temporal, tvar_logical_c, tvar_logical_d],
            max_norm=5.0)
        optimizer1.step(); optimizer2.step(); optimizer3.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                x_val = Nmax.forward(cal_data)
                r1o_val = torch.stack([ tmp.forward(pr.forward(x_val), padding=False)
                                        for pr, tmp in zip(predicates, temporals) ], dim=1)
                rc_val  = torch.stack([ lc.forward(r1o_val, pc[k], keepdim=False)
                                        for k, lc in enumerate(logical_cs) ], dim=1)
                R_val   = logical_d.forward(rc_val, pd[0], keepdim=False)
                val_acc = (torch.sign(R_val)==cal_label).float().mean().item()

            print(f"[Epoch {epoch:4d}] loss={loss.item():.4f}, val_acc={val_acc:.4f}")

            if val_acc > acc_best:
                acc_best = val_acc
                with open("W_best.pkl","wb") as f:
                    pickle.dump([a,b,t1,t2, pc.clone(), pd.clone(),
                             tvar_temporal, tvar_logical_c, tvar_logical_d,tau_pos.clone(), tau_neg.clone()], f)
                with open("network_best.pkl","wb") as f:
                    pickle.dump([predicates, temporals, logical_cs, logical_d], f)
                with torch.no_grad():
                    pbin = (STEstimator.apply(pc)>0.5).float()
                    num_formula = pbin.sum().item()

    return acc_best, num_formula, tau_pos, tau_neg
