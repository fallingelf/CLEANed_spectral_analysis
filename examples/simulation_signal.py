import sys
sys.path.append('../')
import numpy as np
from algorithms import CLEANed
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

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

g = 0.1
nu_D,nu_W,W,B,R,S,power_spec = CLEANed(f, t, g=g)



#####################
#for continue func
t_c = np.linspace(0, 1, 2000)
f_c = np.cos(2*np.pi*31.25*t_c)
nu_c = (np.linspace(0, len(t_c)-1, len(t_c))-len(t_c)//2)/len(t_c)*(len(t_c)-1)/t_c[-1]
D_c = np.sum(f_c * np.exp(-2j*np.pi*(nu_c.reshape(len(nu_c), 1) * t_c)), axis=1)/len(t_c)


###############################################
######plot for fig:spectral_simult_signal######
###############################################
fig=plt.figure(figsize=(20,16))
plt.subplots_adjust(hspace=0.2, wspace=0.2)
ax1=plt.subplot(2,2,1)
ax1.plot(t, f, 'k*')
ax1.plot(t_c, f_c, 'k--')
ax1.set_xlim(-0.02,1.02)

ax2=plt.subplot(2,2,2)
ax2.plot(nu_c, np.abs(D_c),'k-', label='F')
ax2.set_ylim(-0.025,0.525)

ax3=plt.subplot(2,2,3)
ax3.plot(nu_D, np.abs(R[0]),'k-', label='D')

ax4=plt.subplot(2,2,4)
m = len(nu_W)//4
ax4.plot(nu_W[m-10:3*m+11], np.abs(W[m-10:3*m+11]),'k-', label='W')
ax4.plot(nu_W[m-10:3*m+11], np.abs(B[m-10:3*m+11]),'k-.', label='B')

y_ticks_ax1=np.array([-1,-0.5,0,0.5,1])
ax1.set_yticks(y_ticks_ax1)
ax1.set_yticklabels(list(y_ticks_ax1),fontsize=16)

for (ax, label) in zip([ax1,ax2,ax3,ax4],['(a)','(b)', '(c)', '(d)']):
    ax.tick_params(labelsize=15)
    ax.text(-0.12, 0.96, label, horizontalalignment='center', fontsize=25, 
            verticalalignment='center', transform=ax.transAxes)
    
for ax_i in [ax2,ax3,ax4]:
    ax_i.set_xlim(-105,105)
    ax_i.set_xlabel(u'频率 (Hz)',fontsize=25, labelpad = 10)
    ax_i.set_ylabel(u'功率',fontsize=25, labelpad = 15)
    ax_i.legend(loc='upper right', fontsize=20)
    ax_i.tick_params(labelsize=20)
    
ax1.set_xlabel(u'时间 (秒)',fontsize=25, labelpad = 10)
ax1.set_ylabel(u'信号',fontsize=25, labelpad = 0)
ax1.tick_params(labelsize=20)

#plt.savefig('spectral_simult_signal.png', bbox_inches='tight')
plt.show()


################################################
######plot for fig:spectral_simult_clean_g######
################################################
fig, ax = plt.subplots(4,2, sharey=True)
plt.subplots_adjust(hspace=0.05, wspace=0.03)
fig.set_size_inches(20,20)
ax[0, 0].plot(nu_D, np.abs(R[len(R)//8]),'k-')
ax[0, 1].plot(nu_D, np.abs(S[len(R)//8]),'k-')
ax[1, 0].plot(nu_D, np.abs(R[len(R)//4]),'k-')
ax[1, 1].plot(nu_D, np.abs(S[len(R)//4]),'k-')
ax[2, 0].plot(nu_D, np.abs(R[len(R)//2]),'k-')
ax[2, 1].plot(nu_D, np.abs(S[len(R)//2]),'k-')
ax[3, 0].plot(nu_D, np.abs(R[-1]),'k-')
ax[3, 1].plot(nu_D, np.abs(S[-1]),'k-')

iter_num= [len(R)//8, len(R)//4, len(R)//2, len(R)-1]
for ax_i, iter_i in zip(ax,iter_num):
    ax_i[0].set_ylabel(u'{0}*{1} 功率'.format(iter_i, g), fontsize=25, labelpad = 15)
    for ax_ii in ax_i:
        ax_ii.set_xlim(-105,105)
        ax_ii.set_ylim(-0.01, 0.52)
        ax_ii.tick_params(labelsize=20)
ax[0,0].set_title(u'脏谱',fontsize=30, pad = 20)
ax[0,1].set_title(u'洁谱',fontsize=30, pad = 20)
ax[-1,0].set_xlabel(u'频率 (Hz)',fontsize=25, labelpad = 10)
ax[-1,1].set_xlabel(u'频率 (Hz)',fontsize=25, labelpad = 10)
        
for ax_i in ax[:-1]:
    for ax_ii in ax_i:
        ax_ii.xaxis.set_tick_params(size=0)
        ax_ii.set_xticklabels('')
#plt.savefig('spectral_simult_clean_g{0}.png'.format(g), bbox_inches='tight')
plt.show()
