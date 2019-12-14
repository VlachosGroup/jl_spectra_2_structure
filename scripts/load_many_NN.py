# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:46:31 2017

@author: lansford
"""

from __future__ import absolute_import, division, print_function
import os
#import numpy as np
#import matplotlib.pyplot as plt
from jl_spectra_2_structure.cross_validation import LOAD_CROSS_VALIDATION
from jl_spectra_2_structure.plotting_tools import set_figure_settings
#wasserstein_loss
#kl_div_loss
#Use 3350 max for C2H4, 2200 for CO, and 2000 for NO. Use 750 points for C2H4 and 500 for CO and 450 for NO.
#coverage is 'low', 'high' or a float <= 1
#assert TARGET in ['binding_type','GCN','combine_hollow_sites']
Downloads_folder = os.path.join(os.path.expanduser("~"),'Downloads')
cv_folder = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\cv_BW\cv_large_alpha_long'
#cv_folder = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\cv_BW\cv_7_small_alpha'
set_figure_settings('paper')
CV_class = LOAD_CROSS_VALIDATION(cross_validation_path=cv_folder)
#CV_class.load_CV_class(78)
#print(CV_class.NUM_TRAIN)
#print(CV_class.TRAINING_ERROR)
#print(CV_class.NN_PROPERTIES)
CV_class.load_all_CV_data()
CV_class.get_ensemble_cv()
CV_class.plot_models(CV_class.ENSEMBLE_MODELS)
CO_GCN_high_ensemble = CV_class.get_NN_ensemble([30,31,32,33,34])
#CV_class.plot_models(BEST_MODELS,figure_directory=Downloads_folder)
#CV_class.plot_models(CV_class.CV_RESULTS,figure_directory=Downloads_folder,model_list=[6,7,9,10])
#CV_class.plot_parity_plots(figure_directory=Downloads_folder,model_list=[6,7,9,10])