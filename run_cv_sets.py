# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:46:31 2017

@author: lansford
"""

from __future__ import absolute_import, division, print_function

from jl_spectra_2_structure.cross_validation import CROSS_VALIDATION
CV_class = CROSS_VALIDATION(ADSORBATE='CO',poc=1,CV_SPLITS=5, NUM_GCN_LABELS=11)
CV_class.generate_TEST_CV_indices(TEST_FRACTION=0.20,RANDOM_STATE=0, write_file=True)

#CNCO CV


properties_dictionary = {'batch_size':600, 'learning_rate_init':0.005\
              , 'epsilon':10**-12,'hidden_layer_sizes':(50,50)
              ,'alpha':0.05, 'epochs':3,'iterations':5,'loss': 'wasserstein_loss'}


CV_class.run_CV(TARGET='binding_type', COVERAGE='low', PROPERTIES=properties_dictionary\
               ,num_train=6000, num_val=600, num_test=600\
               ,read_file=True, write_file=True, )

