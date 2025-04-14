import os
import sys
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd()

W_PATH = "./W_best.pkl"
NETWORK_PATH = "./network_best.pkl"

with open(W_PATH, "rb") as f:
    W_best = pickle.load(f)
with open(NETWORK_PATH, "rb") as f:
    pred, temporal, logical_c, logical_d = pickle.load(f)

from utils import Normalization_max
Nmax = Normalization_max(dim=1)

class STL_Formula(torch.nn.Module):
    def __init__(self):
        super(STL_Formula, self).__init__()
    def robustness_trace(self, trace, pscale=1, scale=1, keepdim=True, **kwargs):
        raise NotImplementedError("robustness_trace not implemented")
    def robustness(self, trace, time=0, **kwargs):
        rt = self.robustness_trace(trace, **kwargs)
        return rt[:, -1, :]  # Taking the last time step.
    def __str__(self):
        raise NotImplementedError("__str__ not implemented")

class GreaterThan(STL_Formula):
    def __init__(self, var_name, val):
        super(GreaterThan, self).__init__()
        self.var_name = var_name
        self.val = val
    def robustness_trace(self, trace, pscale=1.0, **kwargs):
        return (trace - self.val) * pscale
    def __str__(self):
        return f"{self.var_name} >= {self.val}"

class LessThan(STL_Formula):
    def __init__(self, var_name, val):
        super(LessThan, self).__init__()
        self.var_name = var_name
        self.val = val
    def robustness_trace(self, trace, pscale=1.0, **kwargs):
        return (self.val - trace) * pscale
    def __str__(self):
        return f"{self.var_name} <= {self.val}"

class Eventually(STL_Formula):
    def __init__(self, subformula, interval):
        super(Eventually, self).__init__()
        self.subformula = subformula
        self.interval = interval  # e.g., [0, T]
    def robustness_trace(self, trace, pscale=1, scale=5, keepdim=True, **kwargs):
        sub_trace = self.subformula.robustness_trace(trace, pscale=pscale, scale=scale, keepdim=keepdim, **kwargs)
        T = self.interval[1]
        if scale > 0:
            soft_max = torch.logsumexp(sub_trace[:, :T, :] * scale, dim=1, keepdim=True) / scale
            return soft_max
        else:
            return torch.max(sub_trace[:, :T, :], dim=1, keepdim=True)[0]
    def __str__(self):
        return f"F[{self.interval[0]},{self.interval[1]})({self.subformula})"

class And(STL_Formula):
    def __init__(self, subformula1, subformula2):
        super(And, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
    def robustness_trace(self, trace, pscale=1, scale=5, keepdim=True, **kwargs):
        r1 = self.subformula1.robustness_trace(trace, pscale=pscale, scale=scale, keepdim=keepdim, **kwargs)
        r2 = self.subformula2.robustness_trace(trace, pscale=pscale, scale=scale, keepdim=keepdim, **kwargs)
        return torch.min(r1, r2)
    def __str__(self):
        return f"({self.subformula1}) ∧ ({self.subformula2})"


def construct_formula(W_params):
    c1 = float(W_params[1][2].item())
    c2 = float(W_params[2][0].item())
    T = int(W_params[3][0].item())
    atom1 = GreaterThan("x", c1)
    atom2 = LessThan("x", c2)
    F1 = Eventually(atom1, interval=[0, T])
    F2 = Eventually(atom2, interval=[0, T])
    formula = And(F1, F2)
    return formula

formula_obj = construct_formula(W_best)
print("Learned STL Formula:")
print(formula_obj)


def infer(sample, pred, temporal, logical_c, logical_d, pc, pd, Nmax):
    x = Nmax.forward(sample)
    nf = len(pred)
    batch_size = sample.shape[0]
    r1o = torch.empty((batch_size, nf), dtype=torch.float64)
    for k, (predi, Ti) in enumerate(zip(pred, temporal)):
        r1 = predi.forward(x)
        r1o[:, k] = Ti.forward(r1, padding=False)
    nc = len(logical_c)
    rc = torch.empty((batch_size, nc), dtype=torch.float64)
    for k, li in enumerate(logical_c):
        rc[:, k] = li.forward(r1o, pc[k, :], keepdim=False)
    R = logical_d.forward(rc, pd[0, :], keepdim=False)
    return R

def compute_score(sample):
    sample_tensor = torch.tensor(sample, dtype=torch.float64).unsqueeze(0)
    with torch.no_grad():
        R = infer(sample_tensor, pred, temporal, logical_c, logical_d, W_best[4], W_best[5], Nmax)
    return R.item()


def sigmoid_based_score(R, alpha=5.0):
    s = 1 / (1 + np.exp(-alpha * R))
    score = 2 * (s - 0.5)
    return score

def compute_nonconformity_sigmoid(sample, alpha=5.0):
    R = compute_score(sample)
    return sigmoid_based_score(R, alpha)

    
def nonconformity_scores_logistic(R_array, y, alpha=5.0):
    p = 1 / (1 + np.exp(-alpha * R_array))
    y_bin = (y == 1).astype(int)  
    prob_true = np.where(y_bin == 1, p, 1.0 - p)
    scores = 1.0 - prob_true
    return scores

def compute_nonconformity_logistic(sample, label, alpha=5.0):
    R = compute_score(sample)
    return nonconformity_scores_logistic(np.array([R]), np.array([label]), alpha)[0]


DATA_PATH = "/home/libbywang9/Downloads/naval_dataset.pkl"
with open(DATA_PATH, "rb") as f:
    train_data, train_label, val_data, val_label = pickle.load(f)
val_data = np.transpose(val_data, (0, 2, 1))

np.random.seed(42)
num_val = val_data.shape[0]
indices = np.random.permutation(num_val)
cal_end = int(0.5 * num_val)
cal_idx, test_idx = indices[:cal_end], indices[cal_end:]
cal_data, test_data = val_data[cal_idx], val_data[test_idx]
cal_labels, test_labels = val_label[cal_idx], val_label[test_idx]


cal_scores_sigmoid = np.array([compute_nonconformity_sigmoid(sample) for sample in cal_data])
#cal_scores_sigmoid = np.array([compute_nonconformity_sigmoid(sample, cal_labels[i]) for i, sample in enumerate(cal_data)])
normal_mask = (cal_labels == 1)
if np.sum(normal_mask) == 0:
    raise ValueError("No normal samples in calibration set to compute threshold.")
normal_scores_sigmoid = cal_scores_sigmoid[normal_mask]
threshold_sigmoid = np.quantile(normal_scores_sigmoid, 0.03)
print("CP threshold (5th percentile of sigmoid-based score): {:.4f}".format(threshold_sigmoid))



cal_scores_logistic = np.array([compute_nonconformity_logistic(sample, label) 
                                 for sample, label in zip(cal_data, cal_labels)])
normal_scores_logistic = cal_scores_logistic[normal_mask]
threshold_logistic = np.quantile(normal_scores_logistic, 0.03)
print("CP threshold (5th percentile of logistic-based score): {:.4f}".format(threshold_logistic))


test_scores_sigmoid = np.array([compute_nonconformity_sigmoid(sample) for sample in test_data])
"""
test_scores_sigmoid = np.array([
    compute_nonconformity_sigmoid(sample, test_labels[i]) 
    for i, sample in enumerate(test_data)
])
"""
pred_labels_sigmoid = (test_scores_sigmoid > threshold_sigmoid).astype(int)
true_labels = (test_labels == 1).astype(int)

acc_sigmoid = np.mean(pred_labels_sigmoid == true_labels) * 100
fp_sigmoid = np.mean((pred_labels_sigmoid == 1) & (true_labels == 0)) * 100
fn_sigmoid = np.mean((pred_labels_sigmoid == 0) & (true_labels == 1)) * 100

print("CP Accuracy (Sigmoid-based): {:.2f}%".format(acc_sigmoid))
print("False Positive Rate (Sigmoid-based): {:.2f}%".format(fp_sigmoid))
print("False Negative Rate (Sigmoid-based): {:.2f}%".format(fn_sigmoid))

test_scores_logistic = np.array([compute_nonconformity_logistic(sample, label) 
                                  for sample, label in zip(test_data, test_labels)])
pred_labels_logistic = (test_scores_logistic > threshold_logistic).astype(int)
acc_logistic = np.mean(pred_labels_logistic == true_labels) * 100
fp_logistic = np.mean((pred_labels_logistic == 1) & (true_labels == 0)) * 100
fn_logistic = np.mean((pred_labels_logistic == 0) & (true_labels == 1)) * 100

print("CP Accuracy (Logistic-based): {:.2f}%".format(acc_logistic))
print("False Positive Rate (Logistic-based): {:.2f}%".format(fp_logistic))
print("False Negative Rate (Logistic-based): {:.2f}%".format(fn_logistic))


test_scores_raw = np.array([compute_score(sample) for sample in test_data])
direct_pred_labels = (test_scores_raw > 0).astype(int)
direct_acc = np.mean(direct_pred_labels == true_labels) * 100
direct_fp = np.mean((direct_pred_labels == 1) & (true_labels == 0)) * 100
direct_fn = np.mean((direct_pred_labels == 0) & (true_labels == 1)) * 100

print("\nDirect classification (threshold = 0):")
print("Accuracy: {:.2f}%".format(direct_acc))
print("False Positive Rate: {:.2f}%".format(direct_fp))
print("False Negative Rate: {:.2f}%".format(direct_fn))
print("\nSample predictions (Direct):")
for i in range(5):
    score = test_scores_raw[i]
    label = "Normal" if score > 0 else "Anomaly"
    true = "Normal" if true_labels[i] == 1 else "Anomaly"
    print(f"Sample {i}: Score = {score:.4f}, Predicted = {label}, Actual = {true}")


fig, axs = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

axs[0].scatter(np.arange(len(test_scores_sigmoid)), test_scores_sigmoid, c=true_labels, cmap='bwr', edgecolors='k')
axs[0].axhline(y=threshold_sigmoid, color='k', linestyle='--', label='CP Threshold (Sigmoid)')
axs[0].set_title('CP-based Sigmoid Nonconformity Scores on Test Set')
axs[0].set_ylabel('Sigmoid-based Score')
axs[0].legend()
cbar0 = fig.colorbar(axs[0].collections[0], ax=axs[0])
cbar0.set_label('True Label (1 = Normal, 0 = Anomaly)')

axs[1].scatter(np.arange(len(test_scores_logistic)), test_scores_logistic, c=true_labels, cmap='bwr', edgecolors='k')
axs[1].axhline(y=threshold_logistic, color='k', linestyle='--', label='CP Threshold (Logistic)')
axs[1].set_title('CP-based Logistic Nonconformity Scores on Test Set')
axs[1].set_ylabel('Logistic-based Score')
axs[1].legend()
cbar1 = fig.colorbar(axs[1].collections[0], ax=axs[1])
cbar1.set_label('True Label (1 = Normal, 0 = Anomaly)')

axs[2].scatter(np.arange(len(test_scores_raw)), test_scores_raw, c=true_labels, cmap='bwr', edgecolors='k')
axs[2].axhline(y=0, color='k', linestyle='--', label='Threshold = 0')
axs[2].set_title('Raw Robustness Scores on Test Set (Direct Classification)')
axs[2].set_xlabel('Test Sample Index')
axs[2].set_ylabel('Raw Robustness Score')
axs[2].legend()
cbar2 = fig.colorbar(axs[2].collections[0], ax=axs[2])
cbar2.set_label('True Label (1 = Normal, 0 = Anomaly)')

plt.tight_layout()
plt.show()
