#/usr/bin/env python

from __future__ import absolute_import, division, print_function
import os
from jl_spectra_2_structure.cross_validation import  CROSS_VALIDATION
import uuid
#x=1
#print('running this script')
#wasserstein_loss
#kl_div_loss
#Use 3350 max for C2H4, 2200 for CO, and 2000 for NO. Use 750 points for C2H4 and 500 for CO and 450 for NO.
#coverage is 'low', 'high' or a float <= 1
#assert TARGET in ['binding_type','GCN','combine_hollow_sites']
if __name__ == "__main__":
    run_number = str(uuid.uuid4())
    batch_size = 250
    learning_rate = 0.001
    epsilon = 10**-12
    alpha = 10**-4
    NUM_TRAIN = 500 #50000
    epochs=1
    training_sets = 2 #200
    hidden_layers = (100,100,100)
    
    for ADSORBATE in ['CO', 'NO', 'C2H4']:
        if ADSORBATE in ['CO', 'NO']:
            TARGETS = ['binding_type','combine_hollow_sites','GCN']
        else:
            TARGETS = ['binding_type', 'GCN']
        for TARGET in TARGETS:
            if ADSORBATE in ['CO','NO'] and TARGET in ['binding_type','combine_hollow_sites']:
                COVERAGES = ['low', 1, 'high']
            elif ADSORBATE == 'CO' and TARGET == 'GCN':
                COVERAGES = ['low', 1, 'high']
            elif ADSORBATE == 'NO' and TARGET == 'GCN':
                COVERAGES = ['low',1]
            else:
                COVERAGES = ['low']
            for COVERAGE in COVERAGES:
                for TRAINING_ERROR in ['gaussian',0.003,None]:
    
                    if ADSORBATE == 'CO':
                        INCLUDED_BINDING_TYPES=[1,2,3,4]
                        MAX_COVERAGES = [1, 0.7, 0.2, 0.2]
                        BINDING_TYPE_FOR_GCN=[1]
                        HIGH_FREQUENCY = 2200
                        ENERGY_POINTS=500
                        GCN_ALL = False
                    elif ADSORBATE == 'NO':
                        INCLUDED_BINDING_TYPES=[1,2,3,4]
                        MAX_COVERAGES = [1, 1, 1, 1]
                        BINDING_TYPE_FOR_GCN=[1]
                        HIGH_FREQUENCY = 2000
                        ENERGY_POINTS=450
                        GCN_ALL = False
                    elif ADSORBATE == 'C2H4':
                        INCLUDED_BINDING_TYPES=[1,2]
                        MAX_COVERAGES = [1, 1]
                        BINDING_TYPE_FOR_GCN=[2]
                        HIGH_FREQUENCY = 2000
                        ENERGY_POINTS=450
                        GCN_ALL = True
                        
                    print('ADSORBATE: ' + ADSORBATE)
                    print('TARGET: ' + TARGET)
                    print('COVERAGE: ' + str(COVERAGE))
                    print('TRAINING_ERROR: ' + str(TRAINING_ERROR))
                    print('GCN_ALL: ' + str(GCN_ALL))
                    
                    work_dir = os.getcwd()
                    cross_validation_path = os.path.join(work_dir,'cross_validation_'+ADSORBATE+'_'+TARGET+'_'+str(COVERAGE))
                    print(cross_validation_path) 
                    CV_class = CROSS_VALIDATION(ADSORBATE,INCLUDED_BINDING_TYPES\
                                                ,cross_validation_path, VERBOSE=True)
                    CV_SPLITS = 3
                    CV_class.generate_test_cv_indices(CV_SPLITS, BINDING_TYPE_FOR_GCN\
                        , test_fraction=0.25, random_state=0, read_file=False, write_file=False)
                    properties_dictionary = {'batch_size':batch_size, 'learning_rate_init':learning_rate\
                    , 'epsilon':epsilon,'hidden_layer_sizes':hidden_layers,'regularization':'L2'\
                    ,'alpha':alpha, 'epochs_per_training_set':epochs,'training_sets':training_sets,'loss': 'wasserstein_loss'}
                    CV_class.set_model_parameters(TARGET=TARGET, COVERAGE=COVERAGE\
                    , MAX_COVERAGES = MAX_COVERAGES, NN_PROPERTIES=properties_dictionary\
                    , NUM_TRAIN=NUM_TRAIN, NUM_VAL=100, NUM_TEST=100\
                    , MIN_GCN_PER_LABEL=12, NUM_GCN_LABELS=10, GCN_ALL = GCN_ALL, TRAINING_ERROR = TRAINING_ERROR\
                    , LOW_FREQUENCY=200, HIGH_FREQUENCY=HIGH_FREQUENCY, ENERGY_POINTS=ENERGY_POINTS)
                    CV_RESULTS_FILE = ADSORBATE+'_'+TARGET+'_'+str(COVERAGE)+'_'+ run_number
                    #CV_class.run_CV_multiprocess(write_file=True, CV_RESULTS_FILE = CV_RESULTS_FILE, num_procs=CV_SPLITS+1)
                    #CV_class.run_CV(write_file=True, CV_RESULTS_FILE = CV_RESULTS_FILE)
                    CV_class.get_test_results()
