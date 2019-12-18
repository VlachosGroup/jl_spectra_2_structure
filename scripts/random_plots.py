# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 19:33:16 2019

@author: lansf
"""
from __future__ import division
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from jl_spectra_2_structure import IR_GEN
from jl_spectra_2_structure.plotting_tools import set_figure_settings

set_figure_settings('paper')
Downloads_folder = os.path.join(os.path.expanduser("~"),'Downloads')
CNCOconv = IR_GEN(ADSORBATE='CO',INCLUDED_BINDING_TYPES=['1,2,3,4']\
                  ,TARGET='binding_type',NUM_TARGETS=4)

y_convCNCO = CNCOconv._get_probabilities(300000,4)
y_convGCN = CNCOconv._get_probabilities(300000,10)
G = gridspec.GridSpec(1, 2)
plt.figure(0,figsize=(3.5,2.2),dpi=300)
ax1 = plt.subplot(G[0,0])
plt.hist(y_convCNCO[:,0],bins=10,edgecolor='black')
plt.ylabel('Number of Samples')
plt.xlabel('Binding-type fraction')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.text(0.85,0.9,'(a)',transform=ax1.transAxes)
ax2 = plt.subplot(G[0,1])
plt.hist(y_convGCN[:,0],bins=10,edgecolor='black')
#plt.ylabel('Number of Samples')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('GCN group fraction')
ax2.text(0.85,0.9,'(b)',transform=ax2.transAxes)
plt.savefig(os.path.join(Downloads_folder,'Sample_Distribution_paper.png'), format='png')
plt.close()