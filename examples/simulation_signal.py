import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
from 

###########################
#####simulation-signal#####
###########################
N = 200
s0 = np.linspace(0, N//4-1, N//4)[list(set(np.random.choice(N//4, int(N//1.5))))]*4 + 0
s1 = np.linspace(0, N//4-1, N//4)[list(set(np.random.choice(N//4, int(N//7.5))))]*4 + 1
s2 = np.linspace(0, N//4-1, N//4)[list(set(np.random.choice(N//4, int(N//1.5))))]*4 + 2
s3 = np.linspace(0, N//4-1, N//4)*4 + 3
sample_point = np.sort(np.hstack((s1, s2, s3))).astype(int)

T = 1
t = np.linspace(0, T, N)[sample_point]
f = np.cos(2*np.pi*31.25*t)

nu_D,nu_W,W,B,R,S,power_spec = CLEANed(flux, bjd, 1/err**2,n_B=10)
