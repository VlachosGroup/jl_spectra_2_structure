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
from jl_spectra_2_structure import get_defaults
from jl_spectra_2_structure.primary_data_creation.dft_2_data import Primary_DATA
from jl_spectra_2_structure.primary_data_creation.dft_2_data import COVERAGE_SCALING
from jl_spectra_2_structure.plotting_tools import set_figure_settings
set_figure_settings('paper')

#Folder where figures are to be saved
Downloads_folder = os.path.join(os.path.expanduser("~"),'Downloads')

#Directories where VASP data is stored
vasp_CO_nano = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\VASP_FILES\all_nanoparticles\Pt_CO'
vasp_NO_nano = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\VASP_FILES\all_nanoparticles\NO_files'
vasp_C2H4_nano = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\VASP_FILES\all_nanoparticles\C2H4_files'
vasp_CO_isotope = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\VASP_FILES\single_species_LDIPOL'
vasp_NO_isotope = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\VASP_FILES\NO_single_species_LDIPOL'
vasp_CO_high_coverage = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\VASP_FILES\CO_High_Coverage_LDIPOL'


#Generate Primary Set of Data from DFT for CO

#Only change this code below if you wish to use a different set of adsorbates
#and metal atoms

nanoparticle_path, isotope_path, high_coverage_path\
, cross_validation_path, coverage_scaling_path = get_defaults('CO')


CO_DATA = Primary_DATA(metal_atoms=['Pt'], adsorbate_atoms=['C','O']\
                       , create_new_vasp_files=False, delta=0.025)
CO_DATA.generate_primary_data(vasp_CO_nano, nanoparticle_path, poc=1\
                      ,data_type='nanoparticle', num_adsorbates='single')
CO_DATA.generate_primary_data(vasp_CO_high_coverage, high_coverage_path, poc=1\
                      ,data_type='surface', num_adsorbates='multiple')
CO_DATA.generate_isotope_data(vasp_CO_isotope, isotope_path\
                              ,masses1=[12,16], masses2=[24,32])
#Coverage Scaling
CO_coverage = COVERAGE_SCALING(isotope_path)
CO_coverage.get_coverage_parameters(coverage_scaling_path)
CO_coverage.save_coverage_figures(Downloads_folder,adsorbate='CO',metal='Pt')

nanoparticle_path, isotope_path, high_coverage_path\
, cross_validation_path, coverage_scaling_path = get_defaults('NO')


#Generate Primary Set of Data from DFT for NO

NO_DATA = Primary_DATA(metal_atoms=['Pt'], adsorbate_atoms=['N','O']\
                       , create_new_vasp_files=False, delta=0.015)

NO_DATA.generate_primary_data(vasp_NO_nano, nanoparticle_path, poc=1\
                      ,data_type='nanoparticle', num_adsorbates='single')

NO_DATA.generate_isotope_data(vasp_NO_isotope, isotope_path\
                              ,masses1=[14,16], masses2=[28,32])
#Coverage Scaling
NO_coverage = COVERAGE_SCALING(isotope_path)
NO_coverage.get_coverage_parameters(coverage_scaling_path)
NO_coverage.save_coverage_figures(Downloads_folder,adsorbate='NO',metal='Pt'\
                                  ,frequency_scale_axis1=[0.92,1.085], frequency_scale_axis2=[0.99,1.16]\
                              ,y_2_ticks=[.98,1.01,1.04,1.07,1.10,1.13,1.16])


#Generate Primary Set of Data from DFT for C2H4
nanoparticle_path, isotope_path, high_coverage_path\
, cross_validation_path, coverage_scaling_path = get_defaults('C2H4')
C2H4_DATA = Primary_DATA(metal_atoms=['Pt'], adsorbate_atoms=['C','H']\
                         , create_new_vasp_files=False, delta=0.015)
C2H4_DATA.generate_primary_data(vasp_C2H4_nano, nanoparticle_path, poc=2\
                      ,data_type='nanoparticle', num_adsorbates='single')
