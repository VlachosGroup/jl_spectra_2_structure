# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:46:31 2017

@author: lansford
"""

from __future__ import absolute_import, division, print_function
from jl_spectra_2_structure.cross_validation import CROSS_VALIDATION

CV_class = CROSS_VALIDATION(ADSORBATE='NO',POC=1)
if __name__ == "__main__":
    CV_class.generate_test_cv_indices(CV_SPLITS=5, NUM_GCN_LABELS=11,test_fraction=0.2\
                                  ,random_state=0, read_file=False, write_file=True)
else:
    print(__name__)
    CV_class.generate_test_cv_indices(CV_SPLITS=5, NUM_GCN_LABELS=11,test_fraction=0.2\
                                  ,random_state=0, read_file=True, write_file=False)

properties_dictionary = {'batch_size':600, 'learning_rate_init':0.005\
              , 'epsilon':10**-12,'hidden_layer_sizes':(50,50)
              ,'alpha':0.05, 'epochs_per_training_set':5,'training_sets':5,'loss': 'wasserstein_loss'}

CV_class.set_model_parameters(TARGET='binding_type', COVERAGE='low', NN_PROPERTIES=properties_dictionary\
                     , NUM_TRAIN=10000, NUM_VAL=10000, NUM_TEST=10000\
                             , MIN_GCN_PER_LABEL=0, NUM_GCN_LABELS=None, GCN_ALL=False)
if __name__ == "__main__":
    CV_class.run_CV_multiprocess(write_file=True, CV_RESULTS_FILE = None, num_procs=10)

#CV_class.run_CV(write_file=True, CV_RESULTS_FILE = None)
    