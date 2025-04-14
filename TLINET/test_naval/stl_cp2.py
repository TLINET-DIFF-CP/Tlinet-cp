#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import pickle
import numpy as np
import torch

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
        return rt[:, -1, :] 
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
        self.interval = interval    
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
    sample_tensor = torch.tensor(sample, dtype=torch.float64).unsqueeze(0)  # shape [1, time, signal_dim]
    with torch.no_grad():
        R = infer(sample_tensor, pred, temporal, logical_c, logical_d, W_best[4], W_best[5], Nmax)
    return R.item()

# Now, define a sigmoid-based nonconformity score.
# We want to map the raw robustness R to a score in [-1, 1],
# such that high R yields scores near +1 (more normal)
# and low R (negative) yields scores near -1 (more abnormal).
def sigmoid_based_score(R, alpha=5.0):
    # Compute the sigmoid: s = 1 / (1 + exp(-alpha * R))
    s = 1 / (1 + np.exp(-alpha * R))
    # Rescale to [-1, 1]: score = 2*(s - 0.5)
    score = 2 * (s - 0.5)
    return score

def compute_nonconformity_sigmoid(sample, alpha=5.0):
    """
    sample: a single sample, numpy array of shape [time, signal_dim]
    Returns the transformed nonconformity score computed from raw robustness.
    The score will be in [-1, 1]: higher scores indicate more normal,
    and lower (more negative) scores indicate more anomalous.
    """
    R = compute_score(sample)
    return sigmoid_based_score(R, alpha)


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


cal_scores = np.array([compute_nonconformity_sigmoid(sample) for sample in cal_data])
normal_mask = (cal_labels == 1)
if np.sum(normal_mask) == 0:
    raise ValueError("No normal samples in calibration set to compute threshold.")
normal_scores = cal_scores[normal_mask]
threshold = np.quantile(normal_scores, 0.05)
print("CP threshold (5th percentile of normal score): {:.4f}".format(threshold))


test_scores = np.array([compute_nonconformity_sigmoid(sample) for sample in test_data])
pred_labels = (test_scores > threshold).astype(int)
true_labels = (test_labels == 1).astype(int)

acc = np.mean(pred_labels == true_labels) * 100
fp = np.mean((pred_labels == 1) & (true_labels == 0)) * 100
fn = np.mean((pred_labels == 0) & (true_labels == 1)) * 100

print("CP Accuracy: {:.2f}%".format(acc))
print("False Positive Rate (normal -> anomaly): {:.2f}%".format(fp))
print("False Negative Rate (anomaly -> normal): {:.2f}%".format(fn))

print("\nSample predictions (CP based):")
for i in range(5):
    score = test_scores[i]
    label = "Normal" if score > threshold else "Anomaly"
    true = "Normal" if true_labels[i] == 1 else "Anomaly"
    print(f"Sample {i}: Score = {score:.4f}, Predicted = {label}, Actual = {true}")


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



