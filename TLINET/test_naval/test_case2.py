# Part 1: Imports and setup
import torch
import pickle
import os
from model_naval import *
from utils import *
from robustness import *
import sys
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


def cp_predict(cal_data, cal_label,
               test_data, test_label,
               file_w, file_network,
               alpha=0.05,
               tau_pos_init=None, tau_neg_init=None):
    with open(file_w, 'rb') as f:
        a, b, t1, t2, pc, pd, *_ = pickle.load(f)
    with open(file_network, 'rb') as f:
        predicates, temporals, logical_cs, logical_d = pickle.load(f)

    device = cal_data.device

    with torch.no_grad():
        r1o_cal = torch.stack([ tmp.forward(pr.forward(cal_data), padding=False)
                                for pr,tmp in zip(predicates, temporals) ], dim=1)
        rc_cal  = torch.stack([ lc.forward(r1o_cal, pc[k], keepdim=False)
                                for k,lc in enumerate(logical_cs) ], dim=1)
        R_cal   = logical_d.forward(rc_cal, pd[0], keepdim=False)
        hinge_cal = torch.clamp(1.0 - R_cal * cal_label, min=0.0)   

        mask_pos = (cal_label == 1)
        m_pos    = int(mask_pos.sum().item())               
        level_pos = 1.0 - alpha * (1.0 + 1.0/m_pos)         
        tau_pos = (tau_pos_init 
            if tau_pos_init is not None 
            else torch.quantile(hinge_cal[mask_pos], level_pos))

        mask_neg = (cal_label == -1)
        m_neg    = int(mask_neg.sum().item())
        level_neg = 1.0 - alpha * (1.0 + 1.0/m_neg)
        tau_neg = (tau_neg_init 
            if tau_neg_init is not None 
            else torch.quantile(hinge_cal[mask_neg], level_neg))
   
    with torch.no_grad():
        r1o_test = torch.stack([ tmp.forward(pr.forward(test_data), padding=False)
                                 for pr,tmp in zip(predicates, temporals) ], dim=1)
        rc_test  = torch.stack([ lc.forward(r1o_test, pc[k], keepdim=False)
                                 for k,lc in enumerate(logical_cs) ], dim=1)
        R_test   = logical_d.forward(rc_test, pd[0], keepdim=False)
        hinge_test = torch.clamp(1.0 - R_test * test_label, min=0.0)

    y_pred_formula = torch.where(R_test > 0,
                                 torch.ones_like(R_test),
                                 -torch.ones_like(R_test))
    formula_acc = (y_pred_formula == test_label).float().mean()

    cov_pos = (hinge_test[test_label==1]  <= tau_pos).float().mean()
    cov_neg = (hinge_test[test_label==-1] <= tau_neg).float().mean()
    coverage = 0.5 * (cov_pos + cov_neg)

    return coverage, formula_acc, tau_pos, tau_neg


# Part 2: Load and preprocess dataset
if __name__ == "__main__":
    file = '/naval_dataset.pkl'
    path = os.path.join(os.path.dirname(__file__), 'naval_dataset.pkl')
    with open(path, 'rb') as f:
        train_data, train_label, cal_data, cal_label, pred_data, pred_label = pickle.load(f)
    train_data  = torch.tensor(train_data, dtype=torch.float64, requires_grad=False)
    train_label = torch.tensor(train_label, dtype=torch.float64, requires_grad=False)
    val_data    = torch.tensor(cal_data,    dtype=torch.float64, requires_grad=False)
    val_label   = torch.tensor(cal_label,   dtype=torch.float64, requires_grad=False)
    pred_data   = torch.tensor(pred_data,   dtype=torch.float64, requires_grad=False)
    pred_label  = torch.tensor(pred_label,  dtype=torch.float64, requires_grad=False)
    train_data  = train_data.permute(0,2,1)
    val_data    = val_data.permute(0,2,1)
    pred_data   = pred_data.permute(0,2,1)
    print('training sample: ', train_data.shape[0])
    print('val sample: ',      val_data.shape[0])
    print('pred sample: ',      pred_data.shape[0])

    # Part 3: Prepare results storage
    test_cases = 4

    num_formula = []
    acc         = []
    result_file = 'result.pkl'
    result_path = os.path.join(os.getcwd(), result_file)
    if os.path.isfile(result_path):
        with open(result_path, 'rb') as f:
            loaded = pickle.load(f)
        if (isinstance(loaded, (list, tuple)) and len(loaded)==2
            and isinstance(loaded[0], list)
            and isinstance(loaded[1], list)):
            num_formula, acc = loaded
        else:
            print("Warning: result.pkl")
            num_formula, acc = [], []
    file_w       = 'W_best.pkl'
    file_network = 'network_best.pkl'
    print('total round number is ' + str(test_cases))
        
    
    # Part 4: Training loop
    Nmax = Normalization_max(dim=1)
    val_norm  = Nmax.forward(val_data)
    pred_norm = Nmax.forward(pred_data)
    for i in range(test_cases):
        print('round ' + str(i))
        epoch = 3001
        """
        print('----- AveragedMax, DNF -----')
        a1_tensor, n1_tensor, tau_pos_best, tau_neg_best = model_naval(train_data, train_label,
                     val_data,   val_label,
                     pred_data,  pred_label,
                     epoch, i,
                     avm=True, variable_based=False)
        a_hard_tensor = hard_accuracy(val_norm, val_label, file_w, file_network)
        a1     = a1_tensor.item()
        a_hard = a_hard_tensor.item()
        n1     = n1_tensor.item()
        if abs(a1 - a_hard) > 1e-2:
            raise Exception(f'Hard accuracy mismatch: nn={a1:.4f} vs hard={a_hard:.4f}')
        print('----- SparseMax, DNF -----')
        a2_tensor, n2_tensor, tau_pos_best, tau_neg_best = model_naval(train_data, train_label,
                     val_data,   val_label,
                     pred_data,  pred_label,
                     epoch, i,
                     avm=False, variable_based=False)
        
        a_hard_tensor = hard_accuracy(val_norm, val_label, file_w, file_network)
        a2     = a2_tensor.item()
        a_hard = a_hard_tensor.item()
        n2     = n2_tensor.item()
        if abs(a2 - a_hard) > 1e-3:
            raise Exception(f'Hard accuracy mismatch: nn={a2:.4f} vs hard={a_hard:.4f}')
        """
        print('----- AveragedMax, non-DNF -----')
        a3_tensor, n3_tensor, tau_pos_best, tau_neg_best = model_naval(train_data, train_label,
                     val_data,   val_label,
                     pred_data,  pred_label,
                     epoch, i,
                     avm=True,  variable_based=True)
        Nmax = Normalization_max(dim=1)
        a_hard_tensor = hard_accuracy(val_norm, val_label, file_w, file_network)
        a3     = float(a3_tensor)
        a_hard = float(a_hard_tensor)
        n3     = float(n3_tensor)
        if abs(a3 - a_hard) > 1e-3:
            raise Exception(f'Hard accuracy mismatch: nn={a3:.4f} vs hard={a_hard:.4f}')
        """
        print('----- SparseMax, non-DNF -----')
        a4_tensor, n4_tensor, tau_pos_best, tau_neg_best = model_naval(train_data, train_label,
                     val_data,   val_label,
                     pred_data,  pred_label,
                     epoch, i,
                     avm=False, variable_based=True)
        a_hard_tensor = hard_accuracy(val_norm, val_label, file_w, file_network)
        a4     = a4_tensor.item()
        a_hard = a_hard_tensor.item()
        n4     = n4_tensor.item()
        if abs(a4 - a_hard) > 1e-3:
            raise Exception(f'Hard accuracy mismatch: nn={a4:.4f} vs hard={a_hard:.4f}')
        """


        #print('Best accuracy for {AveragedMax, DNF} is: ' + str(a1))
        #print('Best accuracy for {SparseMax, DNF} is: '   + str(a2))
        print('Best accuracy for {AveragedMax, variable-based} is: ' + str(a3))
        #print('Best accuracy for {SparseMax, variable-based} is: '   + str(a4))

        #print('Number of formulas for {AveragedMax, DNF} is: ' + str(n1))
        #print('Number of formulas for {SparseMax, DNF} is: '   + str(n2))
        print('Number of formulas for {AveragedMax, variable-based} is: ' + str(n3))
        #print('Number of formulas for {SparseMax, variable-based} is: '   + str(n4))

        #num_formula.append([n1, n2, n3, n4])
        #acc.append([a1, a2, a3, a4])

        num_formula.append(n3)
        acc.append(a3)
