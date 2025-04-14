import torch
import numpy

def get_time(w):
    w = w.bool()
    l = w.shape[0]
    t = []
    tf = False
    for j in range(l):
        if w[j] != tf:
            if tf == True:
                t.append(j-1)
            else:
                t.append(j)
            tf = not tf
        if len(t) == 2:
            break
    if tf == True:
        t.append(l-1)
    if len(t)==0:
        return t, t
    else:
        return t[0], t[1]