import torch
import numpy as np

def compute_robustness_region(cal_data, pred, temporal):
    N = cal_data.shape[0]
    nf = len(pred)
    robustness_values = [[] for _ in range(nf)]
    
    for i in range(N):
        x = cal_data[i:i+1, :, :]  
        for k, (predi, Ti) in enumerate(zip(pred, temporal)):
            r_pred = predi.forward(x)  
            r_val = Ti.forward(r_pred, padding=False)  
            robustness_values[k].append(r_val.item())
    
    predicate_regions = []
    for k in range(nf):
        values = np.array(robustness_values[k])
        lower = np.percentile(values, 5)
        upper = np.percentile(values, 95)
        region_width = upper - lower
        predicate_regions.append(region_width)
    
    temporal_regions = []
    for Ti in temporal:
        region_length = (Ti.t2 - Ti.t1).item()
        temporal_regions.append(region_length)
    
    return predicate_regions, temporal_regions
