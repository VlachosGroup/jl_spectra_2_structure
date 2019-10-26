# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:46:31 2017

@author: lansford
"""

from __future__ import absolute_import, division, print_function

from jl_spectra_2_structure.cross_validation import CROSS_VALIDATION
CV_class = CROSS_VALIDATION(ADSORBATE='CO',POC=1)
CV_class.generate_test_cv_indices(CV_SPLITS=5, NUM_GCN_LABELS=11,test_fraction=0.2\
                                  ,random_state=0, read_file=False, write_file=True)

properties_dictionary = {'batch_size':600, 'learning_rate_init':0.005\
              , 'epsilon':10**-12,'hidden_layer_sizes':(50,50)
              ,'alpha':0.05, 'epochs_per_training_set':3,'training_sets':5,'loss': 'wasserstein_loss'}

CV_class.set_model_parameters(TARGET='binding_type', COVERAGE='low', NN_PROPERTIES=properties_dictionary\
                     , NUM_TRAIN=60000, NUM_VAL=600, NUM_TEST=600\
                             , MIN_GCN_PER_LABEL=0, NUM_GCN_LABELS=None, GCN_ALL=False)

CV_class.run_CV(write_file=True, CV_RESULTS_FILE = None)