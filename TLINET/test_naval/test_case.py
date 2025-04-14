import torch
import pickle
import os
from model_naval import *
from utils import *
from robustness import *
import sys
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


if __name__ == "__main__":
    # load dataset
    file = '/naval_dataset.pkl'
    path = os.path.join(os.path.dirname(__file__), 'naval_dataset.pkl')
    with open(path, 'rb') as f:
        train_data, train_label, val_data, val_label = pickle.load(f)
    train_data = torch.tensor(train_data, requires_grad=False)
    train_label = torch.tensor(train_label, requires_grad=False)
    val_data = torch.tensor(val_data, requires_grad=False)
    val_label = torch.tensor(val_label, requires_grad=False)
    train_data = train_data.permute(0,2,1)
    val_data = val_data.permute(0,2,1)
    print('training sample: ',train_data.shape[0])
    print('val sample: ',val_data.shape[0])

    test_cases = 100

    num_formula = []
    acc = []
    file = '/result.pkl'
    path = os.getcwd()+file
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            num_formula, acc = pickle.load(f)
    file_w = '/W_best.pkl'
    file_network = '/network_best.pkl'
    # a_hard = hard_accuracy(val_data,val_label,file_w,file_network)
    print('total round number is '+str(test_cases))
    for i in range(test_cases):
        print('round '+str(i))
        epoch = 10001
        print('----- AveragedMax, DNF -----')
        a1, n1 = model_naval(train_data, train_label, val_data, val_label, epoch, i, avm=True, variable_based=False)
        a_hard = hard_accuracy(val_data,val_label,file_w,file_network)
        if not torch.equal(a1,a_hard):
            print('neural network accuracy: ', a1)
            raise Exception('Hard accuracy dismatch!')
        a1 = a1.detach().numpy()
        n1 = n1.detach().numpy()
        print('----- SparseMax, DNF -----')
        a2, n2 = model_naval(train_data, train_label, val_data, val_label, epoch, i, avm=False, variable_based=False)
        a_hard = hard_accuracy(val_data,val_label,file_w,file_network)
        if not torch.equal(a2,a_hard):
            print('neural network accuracy: ', a2)
            raise Exception('Hard accuracy dismatch!')
        a2 = a2.detach().numpy()
        n2 = n2.detach().numpy()
        print('----- AveragedMax, non-DNF -----')
        a3, n3 = model_naval(train_data, train_label, val_data, val_label, epoch, i, avm=True, variable_based=True)
        a_hard = hard_accuracy(val_data,val_label,file_w,file_network)
        if not torch.equal(a3,a_hard):
            print('neural network accuracy: ', a3)
            raise Exception('Hard accuracy dismatch!')
        a3 = a3.detach().numpy()
        n3 = n3.detach().numpy()
        print('----- SparseMax, non-DNF -----')
        a4, n4 = model_naval(train_data, train_label, val_data, val_label, epoch, i, avm=False, variable_based=True)
        a_hard = hard_accuracy(val_data,val_label,file_w,file_network)
        if not torch.equal(a4,a_hard):
            print('neural network accuracy: ', a4)
            raise Exception('Hard accuracy dismatch!')
        a4 = a4.detach().numpy()
        n4 = n4.detach().numpy()

        print('Best accuracy for {AveragedMax, DNF} is: '+str(a1))
        print('Best accuracy for {SparseMax, DNF} is: '+str(a2))
        print('Best accuracy for {AveragedMax, variable-based} is: '+str(a3))
        print('Best accuracy for {SparseMax, variable-based} is: '+str(a4))

        print('Number of formulas for {AveragedMax, DNF} is: '+str(n1))
        print('Number of formulas for {SparseMax, DNF} is: '+str(n2))
        print('Number of formulas for {AveragedMax, variable-based} is: '+str(n3))
        print('Number of formulas for {SparseMax, variable-based} is: '+str(n4))

        num_formula.append([n1, n2, n3, n4])
        acc.append([a1, a2, a3, a4])

        f = open('result.pkl', 'wb')
        pickle.dump([num_formula, acc], f)
        f.close()

        half_val = val_data.shape[0] // 2
        val_cal = val_data[:half_val, :, :]
        val_test = val_data[half_val:, :, :]

        def cp_predict(val_cal, val_test, file_w, file_network, T_quantile=0.1, T_pred=0.1, alpha=0.01):
            with open(file_w, 'rb') as f:
                best_params = pickle.load(f)
            with open(file_network, 'rb') as f:
                best_network = pickle.load(f)
            pred, temporal, logical_c, logical_d = best_network


            half = val_cal.shape[0]
            nsample_test = val_test.shape[0]
            nf = len(pred)
            nc = len(logical_c)

    
            r1o_cal = torch.empty((half, nf), dtype=torch.float64)
            for k, (predi, Ti) in enumerate(zip(pred, temporal)):
                r1 = predi.forward(val_cal)
                r1o_cal[:, k] = Ti.forward(r1, padding=False)
                rc_cal = torch.empty((half, nc), dtype=torch.float64)
            for k, li in enumerate(logical_c):
                rc_cal[:, k] = li.forward(r1o_cal, None, keepdim=False)
            R_cal = logical_d.forward(rc_cal, None, keepdim=False)
    
    
            tau = smooth_quantile(R_cal, alpha * (1 + 1/half), T_quantile)
    
            r1o_test = torch.empty((nsample_test, nf), dtype=torch.float64)
            for k, (predi, Ti) in enumerate(zip(pred, temporal)):
                r1 = predi.forward(val_test)
                r1o_test[:, k] = Ti.forward(r1, padding=False)
                rc_test = torch.empty((nsample_test, nc), dtype=torch.float64)
            for k, li in enumerate(logical_c):
                rc_test[:, k] = li.forward(r1o_test, None, keepdim=False)
            R_test = logical_d.forward(rc_test, None, keepdim=False)
    
            cp_soft_output = torch.sigmoid((R_test - tau) / T_pred)
            return cp_soft_output
    

    