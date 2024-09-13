#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 08:52:04 2022

@author: n
"""

#Three armed bandit

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.titlesize'] = 25 
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.labelsize'] = 22 
mpl.rcParams['xtick.labelsize'] = 22 
mpl.rcParams['ytick.labelsize'] = 22 
mpl.rcParams['legend.fontsize'] = 22

def get_R(a):
    if a == 0:
        return np.random.normal(7.5,2)
    elif a == 1:
        return np.random.normal(10,1)
    else:
        return np.random.normal(15,5)


def smooth(v,window):
    n = v.shape[0]
    smoothed = [v[0]]
    for i in range(1,n):
        if i <= window:
            smoothed.append(np.average(v[:i]))
        else:
            smoothed.append(np.average(v[i-window:i]))
    return np.array(smoothed)



Q = np.zeros((3,3))


epsilon = 0.8
alpha = 0.1
gamma=1
R_list = []


s = 0
for episode in range(50000):
    if episode == 30000:
        epsilon = 0
    
    if np.random.uniform() > epsilon:
        a = np.argmax(Q[s])
        R = get_R(a)
        s_new = a
    else:
        a = np.random.choice([0,1,2])
        R = get_R(a)
        s_new = a
    
    R_list.append(R)
    if episode < 30000:
        Q[s,a]  = Q[s,a] + alpha*(R + gamma*np.argmax(Q[s_new]) - Q[s,a])
    s = s_new
    
R_list = np.array(R_list)
plt.plot(smooth(R_list,1000), label=fr'$\epsilon=${epsilon}')
print(np.mean(R_list[30000:]))
plt.legend(fontsize=30)
plt.ylabel('R',fontsize=30)
plt.xlabel('Episode')
