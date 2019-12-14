# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:46:31 2017

@author: lansford
"""

from __future__ import absolute_import, division, print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from jl_spectra_2_structure.cross_validation import LOAD_CROSS_VALIDATION
from jl_spectra_2_structure.plotting_tools import set_figure_settings
#wasserstein_loss
#kl_div_loss
#Use 3350 max for C2H4, 2200 for CO, and 2000 for NO. Use 750 points for C2H4 and 500 for CO and 450 for NO.
#coverage is 'low', 'high' or a float <= 1
#assert TARGET in ['binding_type','GCN','combine_hollow_sites']
Downloads_folder = os.path.join(os.path.expanduser("~"),'Downloads')
cv_folder = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\cv_BW\cv_files'
#cv_folder = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\cv_cv5'
cv_indices = r'C:\Users\lansf\Documents\Data\IR_Materials_Gap\cv_BW\cv_indices'
set_figure_settings('paper')
CV_class = LOAD_CROSS_VALIDATION(cv_indices_path=cv_indices,cross_validation_path=cv_folder)
#CV_class.load_CV_class(78)
#print(CV_class.NUM_TRAIN)
#print(CV_class.TRAINING_ERROR)
#print(CV_class.NN_PROPERTIES)
BEST_MODELS = CV_class.get_best_models(3, 2)
keys = CV_class.get_keys(BEST_MODELS)
print(keys)

ADSORBATE_LIST = []
TARGET_LIST = []
COVERAGE_LIST = []
SCORE_LIST = []
BATCH_LIST = []
EPSILON_LIST = []
ALPHA_LIST = []
NUM_TRAIN_LIST = []
TRAIN_SETS_LIST = []
REG_LIST = []
TRAIN_ERROR_LIST = []
LAYERS_LIST = []
LEARN_RATE = []
for ADSORBATE in BEST_MODELS.keys():
    for TARGET in BEST_MODELS[ADSORBATE].keys():
        for COVERAGE in BEST_MODELS[ADSORBATE][TARGET].keys():
            SCORE_LIST += BEST_MODELS[ADSORBATE][TARGET][COVERAGE]['SCORES'].tolist()
            for CV_INDEX in BEST_MODELS[ADSORBATE][TARGET][COVERAGE]['CV_FILES_INDEX']:
                CV_class.load_CV_class(CV_INDEX)
                ADSORBATE_LIST.append(ADSORBATE)
                TARGET_LIST.append(TARGET)
                COVERAGE_LIST.append(COVERAGE)
                BATCH_LIST.append(CV_class.NN_PROPERTIES['batch_size'])
                EPSILON_LIST.append(CV_class.NN_PROPERTIES['epsilon'])
                ALPHA_LIST.append(CV_class.NN_PROPERTIES['alpha'])
                NUM_TRAIN_LIST.append(CV_class.NUM_TRAIN)
                TRAIN_SETS_LIST.append(CV_class.NN_PROPERTIES['training_sets'])
                REG_LIST.append(CV_class.NN_PROPERTIES['regularization'])
                TRAIN_ERROR_LIST.append(CV_class.TRAINING_ERROR)
                LAYERS_LIST.append(CV_class.NN_PROPERTIES['hidden_layer_sizes'])
                LEARN_RATE.append(CV_class.NN_PROPERTIES['learning_rate_init'])

ADSORBATE_LIST = np.array(ADSORBATE_LIST)
TARGET_LIST = np.array(TARGET_LIST)
COVERAGE_LIST = np.array(COVERAGE_LIST)
SCORE_LIST = np.array(SCORE_LIST)
BATCH_LIST = np.array(BATCH_LIST)
EPSILON_LIST = np.array(EPSILON_LIST)
ALPHA_LIST = np.array(ALPHA_LIST)
NUM_TRAIN_LIST = np.array(NUM_TRAIN_LIST)
TRAIN_SETS_LIST = np.array(TRAIN_SETS_LIST)
REG_LIST = np.array(REG_LIST)
TRAIN_ERROR_LIST = np.array(TRAIN_ERROR_LIST)
LAYERS_LIST = np.array(LAYERS_LIST)
LEARN_RATE = np.array(LEARN_RATE)

PARAMETER_SETS = [ALPHA_LIST,EPSILON_LIST,NUM_TRAIN_LIST,TRAIN_SETS_LIST,LEARN_RATE, BATCH_LIST]
PARAMETER_TITLES = ['Alpha','Epsilon','Data per Training Set','Number of Training Sets','Initial Learning Rate', 'Batch Size']

color_dict = {'CO': 'green', 'NO': 'blue', 'C2H4': 'red'}
marker_dict = {'high': 'o', 'low': 's', '1': '>'}

ADSORBATE_COVERAGE = []
ADSORBATE_STRING = []
for i1, i2 in zip(ADSORBATE_LIST,COVERAGE_LIST):
    ADSORBATE_COVERAGE.append((i1,i2))
    ADSORBATE_STRING.append(i1+', '+i2)
ADSORBATE_COVERAGE = np.array(ADSORBATE_COVERAGE)
ADSORBATE_STRING = np.array(ADSORBATE_STRING)
for TARGET in ['GCN','binding_type','combine_hollow_sites']:
    unique_sets = np.unique(ADSORBATE_COVERAGE[TARGET_LIST==TARGET],axis=0)
    unique_string = np.unique(ADSORBATE_STRING[TARGET_LIST==TARGET])
    for count, parameter_title in enumerate(PARAMETER_TITLES):
        plt.figure()
        plt.title('Target: ' + TARGET + ', Parameter: '+ parameter_title)
        for pair in unique_sets:
            indices = np.all((TARGET_LIST==TARGET,ADSORBATE_LIST==pair[0],COVERAGE_LIST == pair[1]),axis=0)
            plt.plot(PARAMETER_SETS[count][indices],SCORE_LIST[indices], marker=marker_dict[pair[1]],color=color_dict[pair[0]],linewidth=0)
        plt.legend(unique_string)
        if parameter_title in ['Alpha', 'Epsilon']:
            plt.xscale('log')
            plt.xlabel('log('+parameter_title+')')
        else:
            plt.xlabel(parameter_title)
        plt.ylabel('score')
        plt.show()
        

            
#CV_class.plot_models(BEST_MODELS)
#CV_class.plot_models(BEST_MODELS,figure_directory=Downloads_folder)
#CV_class.plot_models(CV_class.CV_RESULTS,figure_directory=Downloads_folder,model_list=[6,7,9,10])
#CV_class.plot_parity_plots(figure_directory=Downloads_folder,model_list=[6,7,9,10])