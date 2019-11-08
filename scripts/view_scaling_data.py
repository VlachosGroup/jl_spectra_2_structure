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
from jl_spectra_2_structure import get_default_data_paths
from jl_spectra_2_structure.primary_data_creation.dft_2_data import Primary_DATA
from jl_spectra_2_structure.primary_data_creation.dft_2_data import COVERAGE_SCALING
from jl_spectra_2_structure.plotting_tools import set_figure_settings
set_figure_settings('presentation')

#Folder where figures are to be saved
Downloads_folder = os.path.join(os.path.expanduser("~"),'Downloads')

#Generate Primary Set of Data from DFT for CO

#Only change this code below if you wish to use a different set of adsorbates
#and metal atoms

nanoparticle_path, isotope_path, high_coverage_path\
, coverage_scaling_path = get_default_data_paths('CO')

print(nanoparticle_path)

CO_DATA = Primary_DATA(metal_atoms=['Pt'], adsorbate_atoms=['C','O']\
                       , create_new_vasp_files=False, delta=0.025)
#Coverage Scaling
CO_coverage = COVERAGE_SCALING(isotope_path)
CO_coverage.save_coverage_figures(Downloads_folder,adsorbate='CO',metal='Pt',presentation=True)

nanoparticle_path, isotope_path, high_coverage_path\
, coverage_scaling_path = get_default_data_paths('NO')


#Generate Primary Set of Data from DFT for NO

NO_DATA = Primary_DATA(metal_atoms=['Pt'], adsorbate_atoms=['N','O']\
                       , create_new_vasp_files=False, delta=0.015)


#Coverage Scaling
NO_coverage = COVERAGE_SCALING(isotope_path)
NO_coverage.save_coverage_figures(Downloads_folder,adsorbate='NO',metal='Pt'\
                                  ,frequency_scale_axis1=[0.92,1.085], frequency_scale_axis2=[0.99,1.16]\
                              ,y_2_ticks=[.98,1.01,1.04,1.07,1.10,1.13,1.16],presentation=True)


