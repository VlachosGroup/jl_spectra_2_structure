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
    for ADSORBATE in ['CO','NO','C2H4']:
        if ADSORBATE == 'CO':
            INCLUDED_BINDING_TYPES=[1,2,3,4]
            MAX_COVERAGES = [1, 0.7, 0.2, 0.2]
            BINDING_TYPE_FOR_GCN=[1]
            BINDING_COVERAGE = [1,'low','high']
            GCN_COVERAGE = [1,'low','high']
            TARGET_SITES = ['binding_type']
            HIGH_FREQUENCY = 2200
            ENERGY_POINTS=500
        elif ADSORBATE == 'NO':
            INCLUDED_BINDING_TYPES=[1,2,3,4]
            MAX_COVERAGES = [1, 1, 1, 1]
            BINDING_TYPE_FOR_GCN=[1]
            BINDING_COVERAGE = ['low',1,'high']
            GCN_COVERAGE = ['low',1]
            TARGET_SITES = ['combine_hollow_sites','binding_type']
            HIGH_FREQUENCY = 2000
            ENERGY_POINTS=450
        elif ADSORBATE == 'C2H4':
            INCLUDED_BINDING_TYPES=[1,2]
            MAX_COVERAGES = [1, 1]
            BINDING_TYPE_FOR_GCN=[2]
            BINDING_COVERAGE = ['low']
            GCN_COVERAGE = ['low']
            TARGET_SITES = ['binding_type']
            HIGH_FREQUENCY = 3350
            ENERGY_POINTS=750
        CV_class = CROSS_VALIDATION(ADSORBATE=ADSORBATE,INCLUDED_BINDING_TYPES=INCLUDED_BINDING_TYPES)
        CV_SPLITS = 3
        CV_class.generate_test_cv_indices(CV_SPLITS=CV_SPLITS, BINDING_TYPE_FOR_GCN=BINDING_TYPE_FOR_GCN\
        , test_fraction=0.25, random_state=0, read_file=False, write_file=True)
        properties_dictionary = {'batch_size':500, 'learning_rate_init':0.005\
                      , 'epsilon':10**-12,'hidden_layer_sizes':(50,50)
                      ,'alpha':0, 'epochs_per_training_set':200,'training_sets':1,'loss': 'wasserstein_loss'}
        
        for COVERAGE in GCN_COVERAGE:   
            for GCN_ALL in [False,True]:
                print('ADSORBATE: '+ADSORBATE)
                print('TARGET: GCN')
                print('COVERAGE: ' + str(COVERAGE))
                CV_class.set_model_parameters(TARGET='GCN', COVERAGE=COVERAGE\
                , MAX_COVERAGES = [1, 0.7, 0.2, 0.2], NN_PROPERTIES=properties_dictionary\
                , NUM_TRAIN=1000, NUM_VAL=1000, NUM_TEST=1000\
                , MIN_GCN_PER_LABEL=0, NUM_GCN_LABELS=11, GCN_ALL = GCN_ALL\
                , LOW_FREQUENCY=200, HIGH_FREQUENCY=HIGH_FREQUENCY, ENERGY_POINTS=ENERGY_POINTS)
                CV_class.set_pc_loadings(70,NUM_SAMPLES=10000)
                print('Total Explained Variance: ' + str(CV_class.TOTAL_EXPLAINED_VARIANCE))
                for alpha in np.linspace(0, 1, 2, endpoint=True):
                    print('alpha: ' + str(alpha))
                    properties_dictionary.update({'alpha':alpha})
                    CV_class.set_nn_parameters(properties_dictionary)
                    CV_class.run_CV_multiprocess(write_file=True, CV_RESULTS_FILE = None, num_procs=CV_SPLITS+1)
           
        for COVERAGE in BINDING_COVERAGE:
            for TARGET in TARGET_SITES:
                print('ADSORBATE: '+ADSORBATE)
                print('TARGET: ' + str(TARGET))
                print('COVERAGE: ' + str(COVERAGE))
                CV_class.set_model_parameters(TARGET=TARGET, COVERAGE=COVERAGE\
                , MAX_COVERAGES = MAX_COVERAGES, NN_PROPERTIES=properties_dictionary\
                , NUM_TRAIN=1000, NUM_VAL=1000, NUM_TEST=1000\
                , MIN_GCN_PER_LABEL=0, NUM_GCN_LABELS=11, GCN_ALL = True\
                , LOW_FREQUENCY=200, HIGH_FREQUENCY=HIGH_FREQUENCY, ENERGY_POINTS=ENERGY_POINTS)
                CV_class.set_pc_loadings(70,NUM_SAMPLES=1000)
                print('Total Explained Variance: ' + str(CV_class.TOTAL_EXPLAINED_VARIANCE))
                for alpha in np.linspace(0, 1, 10, endpoint=True):
                    print('alpha: ' + str(alpha))
                    properties_dictionary.update({'alpha':alpha})
                    CV_class.set_nn_parameters(properties_dictionary)
                    CV_class.run_CV_multiprocess(write_file=True, CV_RESULTS_FILE = None, num_procs=CV_SPLITS+1)
                    #CV_class.run_CV(write_file=True, CV_RESULTS_FILE = None)
                    