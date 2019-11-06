# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:17:05 2019

@author: lansf
"""
import os
from jl_spectra_2_structure import IR_GEN
from jl_spectra_2_structure.plotting_tools import set_figure_settings
set_figure_settings('paper')

#Folder where figures are to be saved
Downloads_folder = os.path.join(os.path.expanduser("~"),'Downloads')
CO_GCNconv = IR_GEN(ADSORBATE='CO',INCLUDED_BINDING_TYPES=[1,2,3,4],TARGET='GCN',NUM_TARGETS=10)
CO_GCNconv.get_GCNlabels(Minimum=12, showfigures=True,figure_directory=Downloads_folder,BINDING_TYPE_FOR_GCN=[1])

NO_GCNconv = IR_GEN(ADSORBATE='NO',INCLUDED_BINDING_TYPES=[1,2,3,4],TARGET='GCN',NUM_TARGETS=10)
NO_GCNconv.get_GCNlabels(Minimum=12, showfigures=True,figure_directory=Downloads_folder,BINDING_TYPE_FOR_GCN=[1])

C2H4_GCNconv = IR_GEN(ADSORBATE='C2H4',INCLUDED_BINDING_TYPES=[1,2],TARGET='GCN',NUM_TARGETS=10)
C2H4_GCNconv.get_GCNlabels(Minimum=12, showfigures=True,figure_directory=Downloads_folder,BINDING_TYPE_FOR_GCN=[1,2])



