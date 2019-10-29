# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:46:31 2017

@author: lansford
"""

from __future__ import absolute_import, division, print_function
from jl_spectra_2_structure.cross_validation import CROSS_VALIDATION

def run_cross_validation(read_file,write_file):
    CV_class.generate_test_cv_indices(CV_SPLITS=CV_SPLITS, NUM_GCN_LABELS=11\
                                      , GCN_ALL = False, test_fraction=0.25\
    ,random_state=0, read_file=read_file, write_file=write_file)
CV_class = CROSS_VALIDATION(ADSORBATE='NO',POC=1)
CV_SPLITS = 3

if __name__ == "__main__":
    run_cross_validation(read_file=False, write_file=True)
else:
    run_cross_validation(read_file=True, write_file=False)

properties_dictionary = {'batch_size':50, 'learning_rate_init':0.005\
              , 'epsilon':10**-12,'hidden_layer_sizes':(50,50)
              ,'alpha':0, 'epochs_per_training_set':2000,'training_sets':1,'loss': 'wasserstein_loss'}
#wasserstein_loss
#kl_div_loss
#Use 3350 max for C2H4, 2200 for CO, and 2000 for NO. Use 750 points for C2H4 and 500 for CO and 450 for NO.
#coverage is 'low', 'high' or a float <= 1
#assert TARGET in ['binding_type','GCN','combine_hollow_sites']

CV_class.set_model_parameters(TARGET='GCN', COVERAGE=1, NN_PROPERTIES=properties_dictionary\
                     , NUM_TRAIN=1000, NUM_VAL=10000, NUM_TEST=10000\
                             , MIN_GCN_PER_LABEL=0, NUM_GCN_LABELS=11,GCN_ALL = False\
                     ,LOW_FREQUENCY=200, HIGH_FREQUENCY=2200, ENERGY_POINTS=450)

CV_class._set_pc_loadings(50,NUM_SAMPLES=1000)
if __name__ == "__main__":
    print('Total Explained Variance' + str(CV_class.TOTAL_EXPLAINED_VARIANCE))
    CV_class.run_CV_multiprocess(write_file=True, CV_RESULTS_FILE = None, num_procs=CV_SPLITS+1)
    #CV_class.run_CV(write_file=True, CV_RESULTS_FILE = None)