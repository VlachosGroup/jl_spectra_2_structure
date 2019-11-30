# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:41:16 2019

@author: lansf
"""

#############################################################################################
#Only run this script if you wish to generate your own set of primary data for model training
#############################################################################################
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from jl_spectra_2_structure.primary_data_creation.dft_2_data import Primary_DATA
from jl_spectra_2_structure.plotting_tools import set_figure_settings
import matplotlib.pyplot as plt
from ase.visualize import view
set_figure_settings('paper')

#Folder where figures are to be saved
Downloads_folder = os.path.join(os.path.expanduser("~"),'Downloads')
temp_file = os.path.join(Downloads_folder,'temp.out')
#Directories where VASP data is stored
vasp_CO_nano = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\VASP_FILES\all_nanoparticles\Pt_CO'
vasp_CO_low_coverage = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\VASP_FILES\single_species_LDIPOL'

#Generate Primary Set of Data from DFT for CO

#Only change this code below if you wish to use a different set of adsorbates
#and metal atoms

CO_DATA = Primary_DATA(metal_atoms=['Pt'], adsorbate_atoms=['C','O']\
                       , create_new_vasp_files=False, delta=0.025)
CO_DATA.generate_primary_data(vasp_CO_nano, temp_file, poc=1\
                      ,data_type='nanoparticle', num_adsorbates='single')
    
CO_SURFACE = Primary_DATA(metal_atoms=['Pt'], adsorbate_atoms=['C','O']\
                       , create_new_vasp_files=False, delta=0.025)
CO_SURFACE.generate_primary_data(vasp_CO_low_coverage, temp_file, poc=1\
                      ,data_type='surface', num_adsorbates='multiple')
    
FREQ_SURFACE = np.array(CO_SURFACE.OUTPUT_DICTIONARY['FREQUENCIES'])
INT_SURFACE = np.array(CO_SURFACE.OUTPUT_DICTIONARY['INTENSITIES'])

    
BINDING_TYPES = np.array(CO_DATA.OUTPUT_DICTIONARY['CN_ADSORBATE'])
GCNList = np.array(CO_DATA.OUTPUT_DICTIONARY['GCN'])
FREQUENCIES = np.array(CO_DATA.OUTPUT_DICTIONARY['FREQUENCIES'])
INTENSITIES = np.array(CO_DATA.OUTPUT_DICTIONARY['INTENSITIES'])

POSITIONS = np.array([i.get_positions() for i in CO_DATA.MOLECULES])
FORCES = np.array([i.get_forces(apply_constraint=False) for i in CO_DATA.MOLECULES])
MAX_FORCE = np.array(CO_DATA.OUTPUT_DICTIONARY['MAX_FORCE'])
is_local_minima = np.min(FREQUENCIES, axis=1) > 0
small_force = MAX_FORCE < 0.05
#Bridge and Atop GCN values 7.33333 and 7.5
GCN_Location_Bridge = np.all((np.round(GCNList,5)>6.5, np.round(GCNList,5)<7.5),axis=0)
GCN_Location_Atop = np.all((np.round(GCNList,5) == 7.5, np.round(GCNList,5) ==  7.5),axis=0)
Bridge_Pt111 = np.arange(GCNList.shape[0])[np.all((BINDING_TYPES==2,GCN_Location_Bridge),axis=0)]
Atop_Pt111 = np.arange(GCNList.shape[0])[np.all((BINDING_TYPES==1,GCN_Location_Atop,is_local_minima,small_force),axis=0)]
plt.figure(4,figsize=(7.5,3),dpi=400)
plt.plot(FREQ_SURFACE[-4][-1], INT_SURFACE[-4][-1],'o',color='black')
plt.plot(FREQ_SURFACE[-3][-1], INT_SURFACE[-3][-1],'^',color='black')
plt.plot(FREQUENCIES[Atop_Pt111[23]][-1], INTENSITIES[Atop_Pt111[23]][-1],'o',color='green')
plt.plot(FREQUENCIES[Atop_Pt111[3]][-1], INTENSITIES[Atop_Pt111[3]][-1],'o',color='red')
plt.plot(FREQUENCIES[Atop_Pt111[8]][-1], INTENSITIES[Atop_Pt111[8]][-1],'o',color='blue')
plt.plot(FREQUENCIES[Bridge_Pt111[29]][-1], INTENSITIES[Bridge_Pt111[29]][-1],'^',color='green')
plt.plot(FREQUENCIES[Bridge_Pt111[2]][-1], INTENSITIES[Bridge_Pt111[2]][-1],'^',color='red')
plt.plot(FREQUENCIES[Bridge_Pt111[1]][-1], INTENSITIES[Bridge_Pt111[1]][-1],'^',color='blue')
plt.ylabel('Relative Intensity')
plt.xlabel('C-O frequency [cm$^{-1}$]')
plt.savefig(os.path.join(Downloads_folder,'111_CO.jpg'), format='jpg')
plt.close

plt.figure(3,figsize=(7.5,3),dpi=400)
plt.plot(FREQ_SURFACE[-4][-2], INT_SURFACE[-4][-2],'o',color='black')
plt.plot(FREQ_SURFACE[-3][-2], INT_SURFACE[-3][-2],'^',color='black')
plt.plot(FREQUENCIES[Atop_Pt111[23]][-2], INTENSITIES[Atop_Pt111[23]][-2],'o',color='green')
plt.plot(FREQUENCIES[Atop_Pt111[3]][-2], INTENSITIES[Atop_Pt111[3]][-2],'o',color='red')
plt.plot(FREQUENCIES[Atop_Pt111[8]][-2], INTENSITIES[Atop_Pt111[8]][-2],'o',color='blue')
plt.plot(FREQUENCIES[Bridge_Pt111[29]][-2], INTENSITIES[Bridge_Pt111[29]][-2],'^',color='green')
plt.plot(FREQUENCIES[Bridge_Pt111[2]][-2], INTENSITIES[Bridge_Pt111[2]][-2],'^',color='red')
plt.plot(FREQUENCIES[Bridge_Pt111[1]][-2], INTENSITIES[Bridge_Pt111[1]][-2],'^',color='blue')
plt.ylabel('Relative Intensity')
plt.xlabel('C-O frequency [cm$^{-1}$]')
plt.savefig(os.path.join(Downloads_folder,'111_PtC.jpg'), format='jpg')
plt.close

molecule = np.array(CO_DATA.MOLECULES)[Bridge_Pt111[1]]
molecule.set_constraint()
view(molecule)





