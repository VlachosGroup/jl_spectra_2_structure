# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:17:05 2019

@author: lansf
"""
import numpy as np
from scipy.special import kl_div
import matplotlib.pyplot as plt
from matplotlib import gridspec

a = np.linspace(0,1,num=100,endpoint=True)
B = [0,0,0,1]
Wl = np.zeros_like(a)
Wl2 = np.zeros_like(a)
Wl3 = np.zeros_like(a)
KL = np.zeros_like(a)
dEdO1 = np.zeros_like(a)
dEdO2 = np.zeros_like(a)
dEdO3 = np.zeros_like(a)
dEdO14 = np.zeros_like(a)
dEdO24 = np.zeros_like(a)
dEdO34 = np.zeros_like(a)
dKLdOi = np.zeros_like(a)
for i in range(len(a)):
    A = np.array([a[i],0,0,1-a[i]])
    Akl = [a[i]+10**-12,+10**-12,+10**-12,1-a[i]+10**-12]
    Bkl = [10**-12,10**-12,10**-12,1+10**-12]
    KL[i] = np.sum(kl_div(Bkl,Akl))
    dKLdOi[i] = a[i]
    W = (1/len(A)*np.sum((np.cumsum(A)-np.cumsum(B))**2))**0.5
    dEdO = 2*A*(np.cumsum((np.cumsum(A)-np.cumsum(B))[::-1])[::-1]-np.sum(np.cumsum(A)*(np.cumsum(A)-np.cumsum(B))))
    dEdO1[i] = dEdO[0]
    dEdO14[i] = dEdO[3]
    Wl[i]= W
    A = np.array([0,a[i],0,1-a[i]])
    W = (1/len(A)*np.sum((np.cumsum(A)-np.cumsum(B))**2))**0.5
    dEdO = 2*A*(np.cumsum((np.cumsum(A)-np.cumsum(B))[::-1])[::-1]-np.sum(np.cumsum(A)*(np.cumsum(A)-np.cumsum(B))))
    dEdO2[i] = dEdO[1]
    dEdO24[i] = dEdO[3]
    Wl2[i]= W
    A = np.array([0,0,a[i],1-a[i]])
    W = (1/len(A)*np.sum((np.cumsum(A)-np.cumsum(B))**2))**0.5
    dEdO = 2*A*(np.cumsum((np.cumsum(A)-np.cumsum(B))[::-1])[::-1]-np.sum(np.cumsum(A)*(np.cumsum(A)-np.cumsum(B))))
    dEdO3[i] = dEdO[2]
    dEdO34[i] = dEdO[3]
    Wl3[i]= W
   
KL/= np.max(KL)
G = gridspec.GridSpec(2, 1)
plt.figure(0,figsize=(3.5,3.8),dpi=300)
ax1 = plt.subplot(G[0,0])
ax1.plot(a,Wl,'g',a,Wl2,'b',a,Wl3,'darkorange',a,KL,'k')
#plt.xlabel('a')
plt.xticks([])
plt.ylabel('Loss')
ax1.text(0.01,0.92,'(a)',size=8,name ='Calibri',transform=ax1.transAxes)
#plt.legend(['[a,0,0,1-a]','[0,a,0,1-a]','[0,0,a,1-a]','kl-div'])
#plt.savefig('../Figures/New_Figures/Wasserstein_paper.png', format='png')
#plt.close()
ax2 = plt.subplot(G[1,0])
ax2.plot(a,dEdO1,'g--')
ax2.plot(a,dEdO2,'b--')
ax2.plot(a,dEdO3,'darkorange',linestyle='--')
ax2.plot(a,dKLdOi,'k--')
ax2.plot(a,dEdO14,'g:')
ax2.plot(a,dEdO24,'b:')
ax2.plot(a,dEdO34,'darkorange',linestyle=':')
ax2.plot(a,-dKLdOi,'k:')
plt.xlabel('a')
plt.ylabel('Derivative wrt\n nonzero elements')
#plt.legend(['[a,0,0,1-a]','[0,a,0,1-a]','[0,0,a,1-a]','kl-div'])
ax2.text(0.01,0.92,'(b)',size=8,name ='Calibri',transform=ax2.transAxes)
plt.savefig('../Figures/New_Figures/Wasserstein_paper.png', format='png')
plt.close()

