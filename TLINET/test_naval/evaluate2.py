import os
import sys
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model_naval import svm_score
from utils import Normalization_max

def infer(batch_x, predicates, temporals, logical_cs, logical_d, pc, pd, Nmax):
    x = Nmax.forward(batch_x)
    B = x.shape[0]
    nf = len(predicates)
    # predicate + temporal
    r1o = torch.empty((B, nf), dtype=torch.float64, device=x.device)
    for k, (pr, Ti) in enumerate(zip(predicates, temporals)):
        r = pr.forward(x)
        r1o[:, k] = Ti.forward(r, padding=False)
    # logical_c
    nc = len(logical_cs)
    rc = torch.empty((B, nc), dtype=torch.float64, device=x.device)
    for k, li in enumerate(logical_cs):
        rc[:, k] = li.forward(r1o, pc[k], keepdim=False)
    # logical_d
    R = logical_d.forward(rc, pd[0], keepdim=False)
    return R.detach().cpu().numpy()

if __name__ == "__main__":
    with open("W_best.pkl", "rb") as f:
        pack = pickle.load(f)
    a, b, t1, t2, pc, pd, \
    tvar_temporal, tvar_logical_c, tvar_logical_d, *rest = pack
    if len(rest) == 2:
        tau_pos, tau_neg = rest
    else:
        raise RuntimeError("W_best.pkl 中缺少 tau_pos, tau_neg，请重新训练保存它们")
    if isinstance(tau_pos, torch.Tensor): tau_pos = tau_pos.item()
    if isinstance(tau_neg, torch.Tensor): tau_neg = tau_neg.item()

    with open("network_best.pkl", "rb") as f:
        predicates, temporals, logical_cs, logical_d = pickle.load(f)

    DATA = os.path.join(os.path.dirname(__file__), "naval_dataset.pkl")
    with open(DATA, "rb") as f:
        train_data, train_label, val_data, val_label, pred_data, pred_label = pickle.load(f)
    pred_data = np.transpose(pred_data, (0,2,1))

    labels = pred_label.astype(int)
    y_test = 2*labels-1 if set(labels)=={0,1} else labels

    test_t = torch.tensor(pred_data, dtype=torch.float64)

    Nmax = Normalization_max(dim=1)
    R_test = infer(test_t, predicates, temporals, logical_cs, logical_d, pc, pd, Nmax)

    # 5) nonconformity（SVM-CP）
    sp_test = svm_score(torch.from_numpy(R_test),   tau_pos).numpy()
    sn_test = svm_score(torch.from_numpy(-R_test),  tau_neg).numpy()
    nonconf_test = 1.0 - 0.5*(sp_test + sn_test)
    nonconf_test = np.clip(nonconf_test, 0.0, 1.0)

    cov_pos = np.mean(nonconf_test[y_test==1]  <= tau_pos) if np.any(y_test==1)  else 0.0
    cov_neg = np.mean(nonconf_test[y_test==-1] <= tau_neg) if np.any(y_test==-1) else 0.0
    coverage = 0.5*(cov_pos + cov_neg) * 100

    pred_sign   = np.where(R_test > 0,  1, -1)
    formula_acc = np.mean(pred_sign == y_test) * 100

    print(f"[SVM-CP Prediction] τ+={tau_pos:.4f}, τ-={tau_neg:.4f}")
    print("cov_pos =", cov_pos, "cov_neg =", cov_neg)
    print(f"Coverage: {coverage:.2f}%")
    print(f"Formula Accuracy: {formula_acc:.2f}%")
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].hist(R_test[y_test==1],   bins=30, alpha=0.6, label='+1')
    axes[0].hist(R_test[y_test==-1],  bins=30, alpha=0.6, label='-1')
    axes[0].axvline(0, color='k', ls='--')
    axes[0].set_title('R Distribution (pred_data)')
    axes[0].legend()

    all_min, all_max = nonconf_test.min(), nonconf_test.max()
    bins = np.linspace(all_min, all_max, 30)
    axes[1].hist(nonconf_test[y_test==1],   bins=bins, alpha=0.6, label='+1')
    axes[1].hist(nonconf_test[y_test==-1],  bins=bins, alpha=0.6, label='-1')
    axes[1].axvline(tau_pos, color='b', ls='--', label='τ+')
    axes[1].axvline(tau_neg, color='r', ls='--', label='τ-')
    axes[1].set_title('Nonconformity Scores')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
