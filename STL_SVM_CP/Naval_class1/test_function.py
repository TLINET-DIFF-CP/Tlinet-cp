import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = torch.linspace(-5, 5, 100)
    sigmoid = torch.sigmoid
    sharpness = 10
    a = 2
    scale_large = 20
    left = sigmoid(-sharpness * (x - a))       # ~1 if x < a, ~0 if x > a
    right = sigmoid(sharpness * (x + a))        # ~1 if x > -a, ~0 if x < -a

    middle = left*right # ~1 in (-a, a), ~0 outside

    # left region: x < -a → sigmoid(-sharp * (x + a)) ≈ 1 → large value
    large_penalty = scale_large * sigmoid(-sharpness * (x + a))

    plt.figure(figsize=(10,5))
    plt.plot(x, middle, label='~1 in (-a, a), ~0 outside')
    plt.plot(x, large_penalty, label='x < -a → large value')
    plt.plot(x, middle + large_penalty, label='Nonconformity scores')
    plt.legend()
    plt.title("Score Function")
    figname = "ScoreFunction.png"
    plt.savefig(figname, format='png')
    plt.show()