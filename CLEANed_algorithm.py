#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:27:50 2021

@author: wqs
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting


def CLEANed(f, t, w=(0,), g=0.1, stop=None, N_B=4):
    '''
    cleaned algorithm of \citep{Roberts et al. 1987_APJ:93:4}
    
    {f, t, w}: the data f_i sampled at t_i with wigthed w_i 
    D: the dirty spectrum
    W: the spectral window function, the Fourier transform of the sampling function.
    B: the clean beam
    S: the cleaned spectrum
    g: clean gain, 0.1 <= g <= 1.
    N_B: the desired number of points per beam
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.modeling import models, fitting
    
    #the grid of frequency
    delta_t = np.min(t[1:] - t[:-1])
    nu_max = 1/(2*delta_t)
    N = len(t)
    m = N//2 * N_B
    range_D = np.linspace(-1*m, 1*m, 2*m+1)
    range_W = np.linspace(-2*m, 2*m, 4*m+1)
    nu_D = range_D * nu_max/m
    nu_W = range_W * nu_max/m
    
    #prepare the data
    if len(w) != len(t):
        w = np.ones_like(t)
    w_prime = w / np.sum(w)
    f_prime = f - np.mean(f)
    t_prime = t - np.mean(t)

    D = np.sum(w_prime * f_prime * np.exp(-2j*np.pi*(nu_D.reshape(len(nu_D), 1) * t_prime)), axis=1)
    W = np.sum(w_prime * np.exp(-2j*np.pi*(nu_W.reshape(len(nu_W), 1) * t_prime)), axis=1)
    
    #fit the dominate peak of W for B, B is real
    gauss_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1., fixed={'amplitude':1, 'mean':0})
    fit_g = fitting.LevMarLSQFitter()
    gauss = fit_g(gauss_init, nu_W[len(nu_W)//2-N_B*5:len(nu_W)//2+N_B*5], np.abs(W[len(nu_W)//2-N_B*5:len(nu_W)//2+N_B*5]))
    B = gauss(nu_W)
    
    if not stop:
        stop = int(np.log10(np.median(np.abs(D))/np.max(np.abs(D)))/ np.log10(1-g))
    
    S_i = 0
    for i in range(stop):
        if i==0:
            R_i = D
        peak_i = np.argmax(np.abs(R_i))
        a_i = (R_i[peak_i]-R_i[peak_i].conjugate()*W[2*peak_i])/(1-np.abs(W[2*peak_i])**2)
        c_i = g * a_i
        W_minus = W[2*m-peak_i:4*m+1-peak_i]
        W_plus = W[peak_i:peak_i+2*m+1]
        R_i = R_i - (c_i*W_minus+c_i.conjugate()*W_plus)
        
        B_minus = B[2*m-peak_i:4*m+1-peak_i]
        B_plus = B[peak_i:peak_i+2*m+1]
        S_i = S_i + c_i*B_minus + c_i.conjugate()*B_plus 
    S = S_i + R_i
    
    nu_clean = nu_D[m+1:]
    power_clean = np.abs(S)[m+1:]+np.abs(S)[:m][::-1]
    
    fig, ax = plt.subplots(2)
    fig.set_size_inches(10,10)
    ax[0].plot(nu_D, np.abs(D),'k-', label='D')
    ax[0].plot(nu_W, np.abs(W),'r--', label='W')
    ax[0].plot(nu_W, np.abs(B),'g-.', label='B')
    ax[1].plot(nu_clean, np.abs(R_i)[m+1:]+np.abs(R_i)[:m][::-1], 'k', label='Residual spectrum')
    ax[1].plot(nu_clean, power_clean, 'b-', label='CLEANed spectrum')
    ax[1].plot(nu_clean[np.argmax(power_clean)], power_clean[np.argmax(power_clean)], 'r*', label='{:.2f} Hz'.format(nu_clean[np.argmax(power_clean)]), ms = 8)
    ax[1].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Power')
    ax[1].set_ylabel('Power')
    for ax_i in ax:
        ax_i.legend(loc='upper right')
    plt.show()
    
    return nu_D,D,nu_W,W,B,S,np.vstack([nu_clean, power_clean])
        

N = 200
s0 = np.linspace(0, N//4-1, N//4)[list(set(np.random.choice(N//4, int(N//1.5))))]*4 + 0
s1 = np.linspace(0, N//4-1, N//4)[list(set(np.random.choice(N//4, int(N//7.5))))]*4 + 1
s2 = np.linspace(0, N//4-1, N//4)[list(set(np.random.choice(N//4, int(N//1.5))))]*4 + 2
s3 = np.linspace(0, N//4-1, N//4)*4 + 3
sample_point = np.sort(np.hstack((s1, s2, s3))).astype(int)

T = 1
t = np.linspace(0, T, N)[sample_point]
f = np.cos(2*np.pi*31.25*(t - np.mean(t)) + np.pi/2)
plt.plot(t, f, '*')

delta_min = np.min(t[1:] - t[:-1])
nu_max = 1/(2*delta_min)
N = len(t)
N_B = 4
m = N//2 * N_B
range_D = np.linspace(-1*m, 1*m, 2*m+1)
range_W = np.linspace(-2*m, 2*m, 4*m+1)
nu_D = range_D * nu_max/m
nu_W = range_W * nu_max/m

f_prime = f - np.mean(f)
t_prime = t - np.mean(t)
D = np.sum(f_prime * np.exp(-2j*np.pi*(nu_D.reshape(len(nu_D), 1) * t_prime)), axis=1)/N
W = np.sum(np.exp(-2j*np.pi*(nu_W.reshape(len(nu_W), 1) * t_prime)), axis=1)/N

g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1., fixed={'amplitude':1, 'mean':0})
fit_g = fitting.LevMarLSQFitter()
g = fit_g(g_init, nu_W[len(nu_W)//2-20:len(nu_W)//2+20], np.abs(W[len(nu_W)//2-20:len(nu_W)//2+20]))
B = g(nu_W)


g = 0.1
S_i = 0
stop = int(np.log10(np.median(np.abs(D))/np.max(np.abs(D)))/ np.log10(1-g))
for i in range(stop):
    if i==0:
        R_i = D
    peak_i = np.argmax(np.abs(R_i))
    a_i = (R_i[peak_i]-R_i[peak_i].conjugate()*W[2*peak_i])/(1-np.abs(W[2*peak_i])**2)
    c_i = g * a_i
    W_minus = W[2*m-peak_i:4*m+1-peak_i]
    W_plus = W[peak_i:peak_i+2*m+1]
    R_i = R_i - (c_i*W_minus+c_i.conjugate()*W_plus)
    
    B_minus = B[2*m-peak_i:4*m+1-peak_i]
    B_plus = B[peak_i:peak_i+2*m+1]
    S_i = S_i + c_i*B_minus + c_i.conjugate()*B_plus 

S = S_i + R_i
nu_clean = nu_D[m+1:]
power_clean = np.abs(S)[m+1:]+np.abs(S)[:m][::-1]
    
fig, ax = plt.subplots(2)
fig.set_size_inches(10,10)
ax[0].plot(nu_D, np.abs(D),'k-', label='D')
ax[0].plot(nu_W, np.abs(W),'r--', label='W')
ax[0].plot(nu_W, np.abs(B),'g-.', label='B')
ax[1].plot(nu_clean, np.abs(R_i)[m+1:]+np.abs(R_i)[:m][::-1], 'k', label='Residual spectrum')
ax[1].plot(nu_clean, power_clean, 'b-', label='CLEANed spectrum')
ax[1].plot(nu_clean[np.argmax(power_clean)], power_clean[np.argmax(power_clean)], 'r*', label='{:.2f} Hz'.format(nu_clean[np.argmax(power_clean)]), ms = 8)
ax[1].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel('Power')
ax[1].set_ylabel('Power')
for ax_i in ax:
    ax_i.legend(loc='upper right')
plt.show()


CLEANed(f, t, np.abs(f));

plt.errorbar(t, f, np.abs(f), fmt='*')










