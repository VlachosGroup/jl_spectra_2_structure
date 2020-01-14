# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:46:31 2017

@author: lansford
"""

from __future__ import absolute_import, division, print_function
import os
import numpy as np
#import matplotlib.pyplot as plt
from jl_spectra_2_structure.cross_validation import LOAD_CROSS_VALIDATION
from jl_spectra_2_structure.plotting_tools import set_figure_settings
#wasserstein_loss
#kl_div_loss
#Use 3350 max for C2H4, 2200 for CO, and 2000 for NO. Use 750 points for C2H4 and 500 for CO and 450 for NO.
#coverage is 'low', 'high' or a float <= 1
#assert TARGET in ['binding_type','GCN','combine_hollow_sites']
Downloads_folder = os.path.join(os.path.expanduser("~"),'Downloads')
#cv_folder = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\cv_BW\cv_BW_CV_NN\cv_small_alpha_2L_smooth_long'
cv_folder = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\cv_BW\cv_BW_CV_NN\cv_exclude_low_frequencies\CO_learning_curve'
set_figure_settings('paper')
CV_class = LOAD_CROSS_VALIDATION(cross_validation_path=cv_folder)
#CV_class.load_CV_class()
#print(CV_class.NUM_TRAIN)
#print(CV_class.TRAINING_ERROR)
#print(CV_class.NN_PROPERTIES)
CV_class.load_all_CV_data()
CV_class.get_ensemble_cv()
#CV_class.plot_models(CV_class.CV_RESULTS)
#CV_class.plot_models(CV_class.ENSEMBLE_MODELS)
BEST_MODELS = CV_class.get_best_models(1, 2)
CV_class.plot_models(BEST_MODELS,figure_directory=Downloads_folder)
#plot C2H4 learning curves
#CV_class.plot_models(CV_class.BEST_MODELS,figure_directory=Downloads_folder\
#                     ,model_list=[0,7],xlim=[0,1200],ylim1= [0.01,0.07],ylim2=[0.03,0.23])
#plot CO learning curves
#CV_class.plot_models(CV_class.BEST_MODELS,figure_directory=Downloads_folder\
#                     ,model_list=[24,19,51,45],xlim=[0,1200],ylim1= [0.015,0.13],ylim2=[0.075,0.23])
CV_class.plot_models(CV_class.BEST_MODELS,figure_directory=Downloads_folder\
                     ,model_list=[10,2,20,15],xlim=[0,1200],ylim1= [0.015,0.13],ylim2=[0.075,0.23])

#plot NO learning curves
#CV_class.plot_models(CV_class.BEST_MODELS,figure_directory=Downloads_folder\
#                     ,model_list=[69,71,76,83],xlim=[0,1200],ylim1= [0.015,0.13],ylim2=[0.055,0.23])
    
CV_class.plot_parity_plots(figure_directory=Downloads_folder,use_ensemble=True\
                    ,model_list = [2])

#CV_class.plot_parity_plots(figure_directory=Downloads_folder,use_ensemble=True\
#                    ,model_list = [71])
    
#CV_class.plot_parity_plots(figure_directory=Downloads_folder,use_ensemble=True\
#                    ,model_list = [0])

#Text below is used to see if hyperparameters perform well enough for all model
#types to include this data in the neural network ensembles.
"""
WL_LIST = CV_class.plot_parity_plots(figure_directory=None,use_ensemble=False\
                    ,model_list = np.arange(len(CV_class.CV_FILES)).tolist())

WL_LIST_ENSEMBLE = CV_class.plot_parity_plots(figure_directory=None,use_ensemble=True\
                    ,model_list = np.arange(len(CV_class.CV_FILES)).tolist())

ADSORBATE_LIST = []
TARGET_LIST = []
COVERAGE_LIST = []

WL_VAL_min = []
WL_VAL_std = []
WL_TEST_min = []
WL_TEST_std = []
for ADSORBATE in CV_class.ENSEMBLE_MODELS.keys():
    for TARGET in CV_class.ENSEMBLE_MODELS[ADSORBATE].keys():
        for COVERAGE in CV_class.ENSEMBLE_MODELS[ADSORBATE][TARGET].keys():
            ADSORBATE_LIST.append(ADSORBATE)
            TARGET_LIST.append(TARGET)
            COVERAGE_LIST.append(COVERAGE)
            argval = CV_class.ENSEMBLE_MODELS[ADSORBATE][TARGET][COVERAGE]['WL_VAL_mean'].argmin()
            argtest = CV_class.ENSEMBLE_MODELS[ADSORBATE][TARGET][COVERAGE]['WL_TEST_mean'].argmin()
            WL_VAL_min.append(CV_class.ENSEMBLE_MODELS[ADSORBATE][TARGET][COVERAGE]['WL_VAL_mean'][argval])
            WL_VAL_std.append(CV_class.ENSEMBLE_MODELS[ADSORBATE][TARGET][COVERAGE]['WL_VAL_std'][argval])
            WL_TEST_min.append(CV_class.ENSEMBLE_MODELS[ADSORBATE][TARGET][COVERAGE]['WL_TEST_mean'][argval])
            WL_TEST_std.append(CV_class.ENSEMBLE_MODELS[ADSORBATE][TARGET][COVERAGE]['WL_TEST_std'][argval])
 """           