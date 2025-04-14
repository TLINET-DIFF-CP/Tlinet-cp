import torch
import pickle
import os
from model_naval_cross_validation import *
from utils import *

def k_fold_split(X, y, k):
    n_samples = X.shape[0]
    fold_size = n_samples // k
    x_train = np.empty((k,fold_size*(k-1),X.shape[1],X.shape[2]))
    y_train = np.empty((k,fold_size*(k-1)))
    x_val = np.empty((k,fold_size,X.shape[1],X.shape[2]))
    y_val = np.empty((k,fold_size))
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  # Shuffle the indices randomly for random partitioning
    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        test_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

        x_train[i,:], x_val[i,:] = X[train_indices,:], X[test_indices,:]
        y_train[i,:], y_val[i,:] = y[train_indices], y[test_indices]
    x_train = torch.tensor(x_train, requires_grad=False)
    y_train = torch.tensor(y_train, requires_grad=False)
    x_val = torch.tensor(x_val, requires_grad=False)
    y_val = torch.tensor(y_val, requires_grad=False)
    return x_train, y_train, x_val, y_val

if __name__ == "__main__":
    # load dataset
    file = '/naval_dataset.pkl'
    path = os.getcwd()+file
    with open(path, 'rb') as f:
        train_data, train_label, test_data, test_label = pickle.load(f)
    test_data = torch.tensor(test_data, requires_grad=False)
    test_label = torch.tensor(test_label, requires_grad=False)
    test_data = test_data.permute(0,2,1)

    n_fold = 5
    train_data = np.transpose(train_data,(0,2,1))
    x_train, y_train, x_val, y_val = k_fold_split(train_data,train_label,n_fold)
    num_formula = []
    acc = []
    file_w = '/W_crossval.pkl'
    file_network = '/network_crossval.pkl'
    for i in range(n_fold):
        print('iteration '+str(i))
        epoch = 10001
        a, n = model_naval(x_train[i], y_train[i], x_val[i], y_val[i], epoch, i, avm=True, variable_based=True, hyper=2.5)
        a_test = hard_accuracy(test_data,test_label,file_w,file_network)
        a = a.detach().numpy()
        n = n.detach().numpy()
        print('Best accuracy for is: '+str(a))
        print('Number of formulas is: '+str(n))
        num_formula.append([n])
        acc.append([a])

        f = open('crossval_result.pkl', 'wb')
        pickle.dump([num_formula, acc], f)
        f.close()