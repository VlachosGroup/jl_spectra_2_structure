# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:46:31 2017

@author: lansford
"""

from __future__ import absolute_import, division, print_function
import os
from jl_spectra_2_structure.cross_validation import CROSS_VALIDATION
import numpy as np
import uuid

#wasserstein_loss
#kl_div_loss
#Use 3350 max for C2H4, 2200 for CO, and 2000 for NO. Use 750 points for C2H4 and 500 for CO and 450 for NO.
#coverage is 'low', 'high' or a float <= 1
#assert TARGET in ['binding_type','GCN','combine_hollow_sites']
if __name__ == "__main__":
    run_number = str(uuid.uuid4())
    print('run_number: ' + run_number)
    #try:
    #    run_number = sys.argv[1]
    #except:
    #    "requires an input argument"
    #    raise
    
    hidden_layer1 = np.random.randint(50,151)
    hidden_layer2 = np.random.randint(50,151)
    hidden_layer_3 = np.random.randint(50,151)
    num_layers = np.random.randint(2,3+1)
    if num_layers == 2:
        hidden_layers = (hidden_layer1, hidden_layer2)
    elif num_layers == 3:
        hidden_layers = (hidden_layer1, hidden_layer2, hidden_layer_3)
    
    random_numbers = np.random.random_sample(7)
    setup_list = [('CO','low','binding_type'),('CO','low','combine_hollow_sites')\
                                    ,('CO','low','GCN'),('CO','high','binding_type')\
                                    ,('CO','high','combine_hollow_sites'),('CO','high','GCN')\
                                    ,('CO',1,'binding_type'),('CO',1,'combine_hollow_sites'),('CO',1,'GCN')\
                                    ,('NO','low','binding_type'),('NO','low','combine_hollow_sites')\
                                    ,('NO','low','GCN'),('NO','high','binding_type')\
                                    ,('NO','high','combine_hollow_sites')\
                                    ,('NO',1,'binding_type'),('NO',1,'combine_hollow_sites'),('NO',1,'GCN')\
                                    ,('C2H4','low','binding_type'),('C2H4','low','GCN',False),('C2H4','low','GCN',True)]
    which_setup = setup_list[np.random.choice(np.arange(len(setup_list)))]
    

    batch_size = int(10**(3*random_numbers[0]+1))
    learning_rate = 10**(random_numbers[1]-3)
    epsilon = 10**(4*random_numbers[2]-14)
    alpha = 10**(5*random_numbers[3]-6)
    NUM_TRAIN = int(10**(random_numbers[4]+4))
    training_sets = int(2*10**(random_numbers[5]+1))
    epochs = int(10**(2*random_numbers[6]))
    if which_setup[1] == 'high' and which_setup[2] in ['binding_type','combine_hollow_sites']:
        NUM_TRAIN = 10000
    print('batch_size: '+str(batch_size))
    print('learning_rate: '+str(learning_rate))
    print('epsilon: '+str(epsilon))
    print('alpha: '+str(alpha))
    print('NUM_TRAIN: '+str(NUM_TRAIN))
    print('training_sets: '+str(training_sets))
    print('epochs: '+str(epochs))
    print('hidden layers: '+str(hidden_layers))
    #batch_size: 2297
    #learning_rate = 0.06651629336077657
    #epsilon = 0.02872678693613251
    #alpha = 0.001931666535492889
    #NUM_TRAIN = 100000
    #epochs=2
    #training_sets = 2
    ADSORBATE = which_setup[0]
    ADSORBATE = 'CO'
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
        if which_setup[2] == 'GCN':
            GCN_ALL = which_setup[3]
        else:
            GCN_ALL = False
    COVERAGE = which_setup[1]
    TARGET = which_setup[2]
    TARGET = 'GCN'
    COVERAGE = 'low'
    print('ADSORBATE: '+ADSORBATE)
    print('TARGET: ' + str(TARGET))
    print('COVERAGE: ' + str(COVERAGE))
    print('GCN_ALL: ' + str(GCN_ALL))
    
    work_dir = os.getcwd()
    cross_validation_path = os.path.join(work_dir,'cross_validation_'+ADSORBATE+'_'+TARGET+'_'+str(COVERAGE))
    CV_class = CROSS_VALIDATION(ADSORBATE=ADSORBATE,INCLUDED_BINDING_TYPES=INCLUDED_BINDING_TYPES\
                                ,cross_validation_path=cross_validation_path)
    CV_SPLITS = 3
    CV_class.generate_test_cv_indices(CV_SPLITS=CV_SPLITS, BINDING_TYPE_FOR_GCN=BINDING_TYPE_FOR_GCN\
        , test_fraction=0.25, random_state=0, read_file=True, write_file=False)
    properties_dictionary = {'batch_size':batch_size, 'learning_rate_init':learning_rate\
    , 'epsilon':epsilon,'hidden_layer_sizes':hidden_layers\
    ,'alpha':alpha, 'epochs_per_training_set':epochs,'training_sets':training_sets,'loss': 'wasserstein_loss'}
    CV_class.set_model_parameters(TARGET=TARGET, COVERAGE=COVERAGE\
    , MAX_COVERAGES = MAX_COVERAGES, NN_PROPERTIES=properties_dictionary\
    , NUM_TRAIN=NUM_TRAIN, NUM_VAL=10000, NUM_TEST=10000\
    , MIN_GCN_PER_LABEL=12, NUM_GCN_LABELS=10, GCN_ALL = GCN_ALL\
    , LOW_FREQUENCY=200, HIGH_FREQUENCY=HIGH_FREQUENCY, ENERGY_POINTS=ENERGY_POINTS)
    CV_class.set_pc_loadings(70,NUM_SAMPLES=10000)
    print('Total Explained Variance: ' + str(CV_class.TOTAL_EXPLAINED_VARIANCE))
    CV_class.run_CV_multiprocess(write_file=True, CV_RESULTS_FILE = ADSORBATE+'_'+TARGET+'_'+str(COVERAGE)+'_'+ run_number, num_procs=CV_SPLITS+1)
    #CV_class.run_CV(write_file=True, CV_RESULTS_FILE = None)
                    