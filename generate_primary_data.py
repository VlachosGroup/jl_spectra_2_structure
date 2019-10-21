# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:41:16 2019

@author: lansf
"""
from __future__ import absolute_import, division, print_function
import os
import pkg_resources
from jl_spectra_2_structure import get_defaults
from jl_spectra_2_structure.data_creation.dft_2_data import Primary_DATA

data_path = pkg_resources.resource_filename(__name__, 'data/')
downloads_folder = os.path.expanduser('~/Downloads')

vasp_CO_nano = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\VASP_FILES\all_nanoparticles\Pt_CO'
vasp_NO_nano = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\VASP_FILES\all_nanoparticles\NO_files'
vasp_C2H4_nano = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\VASP_FILES\all_nanoparticles\C2H4_files'
vasp_CO_isotope = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\VASP_FILES\single_species_LDIPOL'
vasp_NO_isotope = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\VASP_FILES\NO_single_species_LDIPOL'
vasp_CO_high_coverage = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\VASP_FILES\CO_High_Coverage_LDIPOL'

nanoparticle_path, surface_path, high_coverage_path\
, cross_validation_path, coverage_scaling_path = get_defaults('CO')
CO_DATA = Primary_DATA(metal_atoms=['Pt'], adsorbate_atoms=['C','O'], delta=0.025)
CO_DATA.generate_primary_data(vasp_CO_nano, nanoparticle_path, poc=1\
                      ,data_type='nanoparticle', num_adsorbates='single')
