# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:46:31 2017

@author: lansford
"""

from __future__ import absolute_import, division, print_function
from jl_spectra_2_structure.cross_validation import CROSS_VALIDATION
import numpy as np
#wasserstein_loss
#kl_div_loss
#Use 3350 max for C2H4, 2200 for CO, and 2000 for NO. Use 750 points for C2H4 and 500 for CO and 450 for NO.
#coverage is 'low', 'high' or a float <= 1
#assert TARGET in ['binding_type','GCN','combine_hollow_sites']
if __name__ == "__main__":
    CV_class = CROSS_VALIDATION(ADSORBATE='CO',POC=1)
    CV_SPLITS = 3
    CV_class.generate_test_cv_indices(CV_SPLITS=CV_SPLITS, NUM_GCN_LABELS=11\
                                      , GCN_ALL = True, test_fraction=0.25\
    ,random_state=0, read_file=False, write_file=True)
    properties_dictionary = {'batch_size':500, 'learning_rate_init':0.005\
                  , 'epsilon':10**-12,'hidden_layer_sizes':(50,50)
                  ,'alpha':0, 'epochs_per_training_set':200,'training_sets':1,'loss': 'wasserstein_loss'}
    for TARGET in ['binding_type']:
        print('TARGET: ' + str(TARGET))
        CV_class.set_model_parameters(TARGET=TARGET, COVERAGE=1, NN_PROPERTIES=properties_dictionary\
                         , NUM_TRAIN=1000, NUM_VAL=1000, NUM_TEST=1000\
                                 , MIN_GCN_PER_LABEL=0, NUM_GCN_LABELS=11,GCN_ALL = True\
                         ,LOW_FREQUENCY=200, HIGH_FREQUENCY=3350, ENERGY_POINTS=750)
    
        CV_class._set_pc_loadings(70,NUM_SAMPLES=10000)
        print('Total Explained Variance: ' + str(CV_class.TOTAL_EXPLAINED_VARIANCE))
        for alpha in np.linspace(0,0.1,10):
            properties_dictionary.update({'alpha':alpha})
            CV_class.set_nn_parameters(properties_dictionary)
            CV_class.run_CV_multiprocess(write_file=True, CV_RESULTS_FILE = None, num_procs=CV_SPLITS+1)
            #CV_class.run_CV(write_file=True, CV_RESULTS_FILE = None)