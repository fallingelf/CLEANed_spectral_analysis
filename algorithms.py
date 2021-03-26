#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def CLEANed(f, t, w=(0,), g=0.1, stop=None, n_B=4):
    '''
    PYTHON implementation of CLEAN algorithm of \citep{Roberts et al. 1987_APJ:93:4}.
    
    {f, t, w}: the data f_i sampled at t_i with weigth w_i;
    D: the raw dirty spectrum;
    W: the spectral window function, the Fourier transform of the sampling function;
    B: the clean beam;
    S[i]: the cleaned spectrum after the i-th iteration;
    R[i]: the dirty spectrum after the i-th iteration;
    C[i]: the clean spectrum after the i-th iteration, S = R + C;
    g: the clean gain, 0.1 <= g < 1;
    n_B: the desired number of points per beam;
    stop: number of iterations can be specified.
    
    Created on Mon Mar 22 13:27:50 2021
    @author: wqs
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from numba import jit, float64, complex128
    from astropy.modeling import models, fitting

    @jit(complex128[:](float64[:],float64[:],float64[:],float64[:]), nopython=True, parallel=True)
    def D_jit(w_prime, f_prime, t_prime, nu_D):
        return np.array([np.sum(w_prime * f_prime * np.exp(-2j*np.pi*nu_i*t_prime)) for nu_i in nu_D])
    
    @jit(complex128[:](float64[:],float64[:],float64[:]), nopython=True, parallel=True)
    def W_jit(w_prime, t_prime, nu_W):
        return np.array([np.sum(w_prime * np.exp(-2j*np.pi*nu_i*t_prime)) for nu_i in nu_W])
    
    #the grid of frequency
    delta_t = np.min(t[1:] - t[:-1])
    nu_max = 1/(2*delta_t)
    m = int(len(t)/2 * n_B)  #int((t.max()- t.min())/delta_t/2*n_B) #int(len(t)/2*n_B) 
    range_D = np.linspace(-1*m, 1*m, 2*m+1)
    range_W = np.linspace(-2*m, 2*m, 4*m+1)
    nu_D = range_D * nu_max/m
    nu_W = range_W * nu_max/m
    
    #prepare the data
    if len(w) != len(t):
        w = np.ones_like(t)
    w_prime = w.astype('float64') / np.sum(w)
    f_prime = f.astype('float64') - np.sum(w_prime * f)
    t_prime = t.astype('float64') - np.sum(w_prime * t)
    
    D = D_jit(w_prime, f_prime, t_prime, nu_D)
    W = W_jit(w_prime, t_prime, nu_W)

    #fit the dominate peak of W for B, B is real
    center_peak = (nu_W[2*m-n_B*5:2*m+n_B*5], np.abs(W[2*m-n_B*5:2*m+n_B*5]))
    stddev_gauss = center_peak[0][center_peak[1]>0.8][-1]*3
    gauss_init = models.Gaussian1D(amplitude=1., mean=0, stddev=stddev_gauss, fixed={'amplitude':True, 'mean':True})
    fit_g = fitting.LevMarLSQFitter()
    gauss = fit_g(gauss_init, center_peak[0], center_peak[1])
    B = gauss(nu_W)
    
    Noise_L = np.median(np.abs(W))
    Signal_L = np.sum(np.abs(D))
    if not stop:
        stop_i = int(np.log10(Noise_L/Signal_L)/ np.log10(1-g))
        stop = int(np.log10(Noise_L/stop_i/Signal_L)/ np.log10(1-g))
        while stop!=stop_i:
            stop_i = stop
            stop = int(np.log10(Noise_L/stop_i/Signal_L)/ np.log10(1-g))
    
    R = np.copy(D)
    C = np.zeros_like(D)
    C_i = 0
    R_i = D
    for i in range(stop):
        R_i[m] = 0
        peak_i = np.argmax(np.abs(R_i))
        a_i = (R_i[peak_i]-R_i[peak_i].conjugate()*W[2*peak_i])/(1-np.abs(W[2*peak_i])**2)
        c_i = g * a_i
        W_minus = W[2*m-peak_i:4*m+1-peak_i]
        W_plus = W[peak_i:peak_i+2*m+1]
        R_i = R_i - (c_i*W_minus+c_i.conjugate()*W_plus)
        
        B_minus = B[2*m-peak_i:4*m+1-peak_i]
        B_plus = B[peak_i:peak_i+2*m+1]
        C_i = C_i + c_i*B_minus + c_i.conjugate()*B_plus 
    
        R = np.vstack([R, R_i])
        C = np.vstack([C, C_i])
    S = C + R
    
    frequ_clean = nu_D[m+1:]
    power_clean = np.abs(S[-1])[m+1:]+np.abs(S[-1])[:m][::-1]
    
    fig, ax = plt.subplots(2)
    fig.set_size_inches(20,20)
    ax[0].plot(nu_D, np.abs(D),'k-', label='D')
    ax[0].plot(nu_W, np.abs(W),'r--', label='W')
    ax[0].plot(nu_W, np.abs(B),'g-.', label='B')
    ax[1].plot(frequ_clean, np.abs(R_i)[m+1:]+np.abs(R_i)[:m][::-1], 'k', label='Residual spectrum')
    ax[1].plot(frequ_clean, power_clean, 'b-', label='CLEANed spectrum')
    ax[1].plot(frequ_clean[np.argmax(power_clean)], power_clean[np.argmax(power_clean)], 'r*', \
               label='{:.5f}$\pm${:.5f} Hz'.format(frequ_clean[np.argmax(power_clean)], gauss.stddev.value), ms = 16)
    ax[1].set_xlabel('Frequency (Hz)',fontsize=25)
    ax[0].set_ylabel('Power',fontsize=25)
    ax[1].set_ylabel('Power',fontsize=25)
    for ax_i in ax:
        ax_i.legend(loc='upper right',fontsize=20)
        ax_i.tick_params(labelsize=20)
    plt.show()
    
    return nu_D,nu_W,W,B,R,S,np.vstack([frequ_clean, power_clean])










