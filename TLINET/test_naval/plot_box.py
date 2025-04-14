import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

file = '/result.pkl'
path = os.getcwd()+file
with open(path, 'rb') as f:
    num_f, acc = pickle.load(f)

print('total number of rounds: ', str(len(acc)))

f1 = []
f2 = []
f3 = []
f4 = []
for fi in num_f:
    f1.append(fi[0])
    f2.append(fi[1])
    f3.append(fi[2])
    f4.append(fi[3])

data = [f1,f2,f3,f4]
fig = plt.figure()
ax = fig.add_subplot(111)
bp = ax.boxplot(data)
ax.set_xticklabels(['averaged,DNF', 'sparse,DNF','averaged,nonDNF', 'sparse,nonDNF'])
plt.show()

a1 = []
a2 = []
a3 = []
a4 = []
for acci in acc:
    a1.append(acci[0])
    a2.append(acci[1])
    a3.append(acci[2])
    a4.append(acci[3])

data = [a1, a2, a3, a4]
fig = plt.figure()
ax = fig.add_subplot(111)
bp = ax.boxplot(data)
ax.set_xticklabels(['averaged,DNF', 'sparse,DNF','averaged,nonDNF', 'sparse,nonDNF'])
plt.show()