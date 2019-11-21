# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:46:31 2017

@author: lansford
"""

from __future__ import division
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from jl_spectra_2_structure import HREEL_2_scaledIR
from jl_spectra_2_structure.plotting_tools import set_figure_settings
from jl_spectra_2_structure.cross_validation import LOAD_CROSS_VALIDATION
from jl_spectra_2_structure import get_exp_data_path
from jl_spectra_2_structure import fold
set_figure_settings('paper')
#loading Spectrum files
#Loading Neural Networks
CV_DATA_PATH = r'C:\Users\lansf\Box Sync\Synced_Files\Coding\Python\Github\jl_spectra_2_structure\scripts\cross_validation_CO_binding_type_high'
Downloads = r'C:\Users\lansf\Downloads'
CV_class = LOAD_CROSS_VALIDATION(cross_validation_path=CV_DATA_PATH)
CV_class.load_CV_class(1)
NN_CNCO = CV_class.NN
X = np.linspace(CV_class.LOW_FREQUENCY,CV_class.HIGH_FREQUENCY,num=CV_class.ENERGY_POINTS,endpoint=True).reshape((-1,1))
EXP_FILES = get_exp_data_path()
COc4x2Pt111 = HREEL_2_scaledIR(np.loadtxt(EXP_FILES[3], delimiter=',', usecols=(0, 1)).T, frequency_range=X)
COp1x2Pt110 = HREEL_2_scaledIR(np.loadtxt(EXP_FILES[0], delimiter=',', usecols=(0, 1)).T, frequency_range=X)
COPt111LowCov = HREEL_2_scaledIR(np.loadtxt(EXP_FILES[1], delimiter=',', usecols=(0, 1)).T, frequency_range=X)
COPtnano = HREEL_2_scaledIR(np.loadtxt(EXP_FILES[2], delimiter=',', usecols=(0, 1)).T, frequency_range=X)
#NN_CNCO_LowCov = get_NN(NN_CNCO_CV_LowCov[-1])
#NN_GCN =
CNCO_COc4x2Pt111 = NN_CNCO.predict(COc4x2Pt111.reshape(1,-1))[0]
CNCO_COPt111LowCov = NN_CNCO.predict(COPt111LowCov.reshape(1,-1))[0]
CNCO_COp1x2Pt110 = NN_CNCO.predict(COp1x2Pt110.reshape(1,-1))[0]
CNCO_COPtnano = NN_CNCO.predict(COPtnano.reshape(1,-1))[0]
#CNCO_LowCov_COc4x2Pt111 = NN_CNCO_LowCov.predict(COc4x2Pt111.reshape(1,-1))[0]
#CNCO_LowCov_COPt111LowCov = NN_CNCO_LowCov.predict(COPt111LowCov.reshape(1,-1))[0]
#CNCO_LowCov_COp1x2Pt110 = NN_CNCO_LowCov.predict(COp1x2Pt110.reshape(1,-1))[0]
#CNCO_LowCov_COPtnano = NN_CNCO_LowCov.predict(COPtnano.reshape(1,-1))[0]
#GCN_COc4x2Pt111 = NN_GCN.predict(COc4x2Pt111.reshape(1,-1))[0]
#GCN_COPt111LowCov = NN_GCN.predict(COPt111LowCov.reshape(1,-1))[0]
#GCN_COp1x2Pt110 = NN_GCN.predict(COp1x2Pt110.reshape(1,-1))[0]
#GCN_COPtnano = NN_GCN.predict(COPtnano.reshape(1,-1))[0]
Bridge_Pt111 = np.all((CV_class.MAINconv.BINDING_TYPES==2,np.round(CV_class.MAINconv.GCNList,5)==7.33333),axis=0)
Atop_Pt111 = np.all((CV_class.MAINconv.BINDING_TYPES==1,np.round(CV_class.MAINconv.GCNList,1)==7.5),axis=0)
X = CV_class.MAINconv.scaling_factor_shift(CV_class.MAINconv.X0cov)
FWHM = 200
spectra1 = fold(X[Bridge_Pt111][0][0],X[Bridge_Pt111][0][1],200,2200,500,FWHM,1)
Gaussian2 = CV_class.MAINconv._generate_spectra(X[Bridge_Pt111][0][0].reshape(1,-1),X[Bridge_Pt111][0][1].reshape(1,-1),CV_class.MAINconv.ENERGIES)
transform = CV_class.MAINconv._mixed_lineshape(FWHM, 1, CV_class.MAINconv.ENERGIES.shape[0], CV_class.MAINconv.ENERGIES[1]-CV_class.MAINconv.ENERGIES[0])
spectra2 = np.convolve(Gaussian2[0], transform, mode='valid')
plt.figure(0)
plt.plot(spectra2)
plt.plot(spectra1,'--')
plt.show()





print(CNCO_COc4x2Pt111)
print(CNCO_COPt111LowCov)
print(CNCO_COp1x2Pt110)
print(CNCO_COPtnano)
#print(GCN_COc4x2Pt111)
#print(GCN_COPt111LowCov)
#print(GCN_COp1x2Pt110)
#print(GCN_COPtnano)

G = gridspec.GridSpec(2, 1)
plt.figure(0,figsize=(3.5,4),dpi=300)
ax1 = plt.subplot(G[0,0])
x = np.arange(1,CNCO_COc4x2Pt111.size+1)
ax1.bar(x-0.4, CNCO_COc4x2Pt111,width=0.2,color='g',align='edge', edgecolor='black', hatch='/')
ax1.bar(x-0.2, CNCO_COPt111LowCov,width=0.2,color='b',align='edge', edgecolor='black', hatch='\\')
ax1.bar(x, CNCO_COp1x2Pt110,width=0.2,color='orange',align='edge', edgecolor='black', hatch='-')
ax1.bar(x+0.2, CNCO_COPtnano,width=0.2,color='darkorchid',align='edge', edgecolor='black')
ax1.set_xlim([0.5,CNCO_COc4x2Pt111.size+0.5])
#plt.legend(['Pt(111) c(4x2)','Pt(111) 0.17 ML CO', 'Pt(110)','55 nm Au@0.7 nm Pt/Pt'],loc=1)
plt.xlabel('Site-type')
plt.ylabel('CO site distribution')
#plt.title('Predicted Site-type Histograms')
ax1.set_xticks(range(1,len(x)+1))
ax1.set_xticklabels(['Atop','Bridge','3-fold','4-fold'])
ax1.text(0.01,0.92,'(a)', transform=ax1.transAxes)

#x = np.arange(1,CNCO_LowCov_COc4x2Pt111.size+1)
#ax2 = plt.subplot(G[1,0])
#ax2.bar(x-0.4, CNCO_LowCov_COc4x2Pt111,width=0.2,color='g',align='edge', edgecolor='black', hatch="/")
#ax2.bar(x-0.2, CNCO_LowCov_COPt111LowCov,width=0.2,color='b',align='edge', edgecolor='black', hatch="\\")
#ax2.bar(x, CNCO_LowCov_COp1x2Pt110,width=0.2,color='orange',align='edge', edgecolor='black', hatch="-")
#ax2.bar(x+0.2, CNCO_LowCov_COPtnano,width=0.2,color='darkorchid',align='edge', edgecolor='black')
#ax2.set_xlim([0.5,CNCO_LowCov_COc4x2Pt111.size+0.5])
#plt.legend(['Pt(111) c(4x2)','Pt(111) 0.17 ML CO', 'Pt(110)','55 nm Au@0.7 nm Pt/Pt'],loc=2)
#plt.xlabel('Generalized Coordination Group')
#plt.ylabel('CO site distribution')
#plt.title('Predicted GCN Histogram')
#ax2.set_xticklabels(['Atop','Bridge','3-fold','4-fold'])
#ax2.set_xticks(range(1,len(x)+1))
#ax2.text(0.01,0.92,'(b)', transform=ax2.transAxes)
#plt.savefig('../Figures/High_vs_LowCov_Hist_paper.png', format='png')
#plt.close()

linestyle = ['-',':','-.','--']
G = gridspec.GridSpec(2, 2)
plt.figure(1,figsize=(7.2,5.2),dpi=300)
ax1 = plt.subplot(G[1,1])
section = np.arange(350,X.size)
plt.plot(X[section],COc4x2Pt111[section],'g',linestyle=linestyle[0])
plt.plot(X[section],COPt111LowCov[section],'b',linestyle=linestyle[1])
plt.plot(X[section],COp1x2Pt110[section],'orange',linestyle=linestyle[2])
plt.plot(X[section],COPtnano[section],'darkorchid',linestyle=linestyle[3])
plt.xlabel('Frequency [cm$^{-1}$]')
plt.ylabel('Relative Intensity')
#plt.title('C-O Frequency Range')
#plt.legend(['Pt(111) c(4x2)','Pt(111) 0.17 ML CO', 'Pt(110)','55 nm Au@0.7 nm Pt/Pt'],loc=2)
ax1.text(0.01,0.92,'(c)', transform=ax1.transAxes)

#plt.figure(2,figsize=(3.5,1.8),dpi=300)
ax2 = plt.subplot(G[1,0])
section = np.arange(0,120)
plt.plot(X[section],COc4x2Pt111[section],'g',linestyle=linestyle[0])
plt.plot(X[section],COPt111LowCov[section],'b',linestyle=linestyle[1])
plt.plot(X[section],COp1x2Pt110[section],'orange',linestyle=linestyle[2])
plt.plot(X[section],COPtnano[section],'darkorchid',linestyle=linestyle[3])
plt.xlabel('Frequency [cm$^{-1}$]')
plt.ylabel('Relative Intensity')
#plt.title('Pt-CO Frequency Range')
#plt.legend(['Pt(111) c(4x2)','Pt(111) 0.17 ML CO', 'Pt(110)','55 nm Au@0.7 nm Pt/Pt'],loc=2)
ax2.text(0.01,0.92,'(b)', transform=ax2.transAxes)

#plt.figure(3,figsize=(7.2,2),dpi=300)
ax3 = plt.subplot(G[0,:])
plt.plot(X,COc4x2Pt111,'g',linestyle=linestyle[0])
plt.plot(X,COPt111LowCov,'b',linestyle=linestyle[1])
plt.plot(X,COp1x2Pt110,'orange',linestyle=linestyle[2])
plt.plot(X,COPtnano,'darkorchid',linestyle=linestyle[3])
plt.xlabel('Frequency [cm$^{-1}$]')
plt.ylabel('Relative Intensity')
ax3.text(0.01,0.92, '(a)', transform=ax3.transAxes)
#plt.title('Experimental Spectroscopy')
#plt.legend(['Pt(111) c(4x2)','Pt(111) 0.17 ML CO', 'Pt(110)','55 nm Au@0.7 nm Pt/Pt'],loc=2)
plt.savefig(os.path.join(Downloads,'Experimental_Spectra_paper.png'), format='png')
plt.close()


G = gridspec.GridSpec(2, 2)
G.update(wspace=0.0,hspace=.6)

linestyle = ['-',':','-.','--']
plt.figure(2,figsize=(7.2,4),dpi=400)
ax3 = plt.subplot(G[0,:])
plt.plot(X,COc4x2Pt111,'g',linestyle=linestyle[0])
plt.plot(X,COPt111LowCov,'b',linestyle=linestyle[1])
plt.plot(X,COp1x2Pt110,'orange',linestyle=linestyle[2])
plt.plot(X,COPtnano,'darkorchid',linestyle=linestyle[3])
plt.xlabel('Frequency [cm$^{-1}$]')
plt.ylabel('Relative Intensity')
ax3.text(0.002,0.93,'(a)', transform=ax3.transAxes)

ax1 = plt.subplot(G[1,0])
x = np.arange(1,CNCO_COc4x2Pt111.size+1)
ax1.bar(x-0.4, CNCO_COc4x2Pt111,width=0.2,color='g',align='edge', edgecolor='black', hatch='/',linewidth=1)
ax1.bar(x-0.2, CNCO_COPt111LowCov,width=0.2,color='b',align='edge', edgecolor='black', hatch='\\',linewidth=1)
ax1.bar(x, CNCO_COp1x2Pt110,width=0.2,color='orange',align='edge', edgecolor='black', hatch='-',linewidth=1)
ax1.bar(x+0.2, CNCO_COPtnano,width=0.2,color='darkorchid',align='edge', edgecolor='black',linewidth=1)
ax1.set_xlim([0.5,CNCO_COc4x2Pt111.size+0.5])
#plt.legend(['Pt(111) c(4x2)','Pt(111) 0.17 ML CO', 'Pt(110)','55 nm Au@0.7 nm Pt/Pt'],loc=1)
plt.xlabel('Site-type')
plt.ylabel('CO site distribution')
#plt.title('Predicted Site-type Histograms')
ax1.set_xticks(range(1,len(x)+1))
ax1.set_xticklabels(['Atop','Bridge','3-fold','4-fold'])
ax1.text(0.004,0.93,'(b)', transform=ax1.transAxes)

#x = np.arange(1,GCN_COc4x2Pt111.size+1)
#ax2 = plt.subplot(G[1,1])
#ax2.bar(x-0.4, GCN_COc4x2Pt111,width=0.2,color='g',align='edge', edgecolor='black', hatch="/",linewidth=1)
#ax2.bar(x-0.2, GCN_COPt111LowCov,width=0.2,color='b',align='edge', edgecolor='black', hatch="\\",linewidth=1)
#ax2.bar(x, GCN_COp1x2Pt110,width=0.2,color='orange',align='edge', edgecolor='black', hatch="-",linewidth=1)
#ax2.bar(x+0.2, GCN_COPtnano,width=0.2,color='darkorchid',align='edge', edgecolor='black',linewidth=1)
#ax2.set_xlim([0.5,GCN_COc4x2Pt111.size+0.5])
#plt.legend(['Pt(111) c(4x2)','Pt(111) 0.17 ML CO', 'Pt(110)','55 nm Au@0.7 nm Pt/Pt'],loc=2)
#plt.xlabel('Generalized Coordination Group')
#plt.yticks([])
#plt.title('Predicted GCN Histogram')
#ax2.set_xticks(range(1,len(x)+1))
#ax2.text(0.004,0.93,'(c)', transform=ax2.transAxes)
#plt.gcf().subplots_adjust(bottom=0.09,top=0.98,right=0.98,left=0.06)
#plt.savefig('../Figures/Experimental_Hist_with_data.png', format='png')
#plt.close()