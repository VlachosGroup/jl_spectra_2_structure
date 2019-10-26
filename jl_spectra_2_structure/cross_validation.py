# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:05:24 2017

@author: lansf
"""
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import json
import json_tricks
from timeit import default_timer as timer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from copy import deepcopy
from .neural_network import MLPRegressor
from .jl_spectra_2_structure import IR_GEN
from .jl_spectra_2_structure import get_defaults
from . import error_metrics

class CROSS_VALIDATION:
    """
    """
    def __init__(self, ADSORBATE='CO', POC=1\
                 ,cross_validation_path = None, nanoparticle_path=None\
                 ,high_coverage_path=None, coverage_scaling_path=None):
        """
        """
        assert type(ADSORBATE) == str, "ADSORBATE must be a String"
        nano_path, isotope_path, high_cov_path\
           , cross_val_path, cov_scale_path = get_defaults(ADSORBATE)
        
        if cross_validation_path is None:
            cross_validation_path = cross_val_path
        self.CV_PATH = cross_validation_path
        self.ADSORBATE = ADSORBATE
        self.POC = POC
        self.NANO_PATH = nanoparticle_path
        self.HIGH_COV_PATH = high_coverage_path
        self.COV_SCALE_PATH = coverage_scaling_path
        
    def generate_test_cv_indices(self, CV_SPLITS=5, NUM_GCN_LABELS=11\
                                 ,test_fraction=0.2,random_state=0, read_file=False, write_file=False):
        """
        """
        ADSORBATE = self.ADSORBATE
        POC = self.POC
        NANO_PATH = self.NANO_PATH
        HIGH_COV_PATH = self.HIGH_COV_PATH
        COV_SCALE_PATH = self.COV_SCALE_PATH
        CV_PATH = self.CV_PATH
        INDICES_FILE = os.path.join(CV_PATH,\
        'cross_validation_indices_'+str(CV_SPLITS)+'fold_'+ADSORBATE+'.json')
        if read_file == False:
            INDICES_DICTIONARY = {'BINDING_TYPE':{'train_indices':[], 'val_indices':[]\
                             ,'CV_indices':[],'TEST_indices':[]}\
                            ,'GCN_ALL':{'train_indices':[], 'val_indices':[]\
                             ,'CV_indices':[],'TEST_indices':[]}\
                            ,'GCN_ATOP':{'train_indices':[], 'val_indices':[]\
                             ,'CV_indices':[],'TEST_indices':[]}\
                            ,'BINDING_TYPE_4GCN':{'train_indices':[], 'val_indices':[]\
                             ,'CV_indices':[],'TEST_indices':[]}}
            GCNconv = IR_GEN(ADSORBATE, POC=POC, TARGET='GCN', NUM_TARGETS=NUM_GCN_LABELS\
                             ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                             , coverage_scaling_path=COV_SCALE_PATH)
            
            GCNconv.get_GCNlabels(Minimum=0,showfigures=False)
            #Get List of indices so we can split the data later
            #combined class to stratify data based on binding-type and GCN simultaneously
            combined_class = 100*GCNconv.BINDING_TYPES+GCNconv.GCNlabels
            #split data into cross validation and test set
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
            for CV_index, test_index in sss.split(combined_class,combined_class):
                CV_indices = CV_index
                TEST_indices = test_index
            INDICES_DICTIONARY['BINDING_TYPE'].update({'CV_indices':CV_indices.astype('int').tolist()})
            INDICES_DICTIONARY['BINDING_TYPE'].update({'TEST_indices':TEST_indices.astype('int').tolist()})
            INDICES_DICTIONARY['GCN_ATOP'].update({'CV_indices':CV_indices[GCNconv.BINDING_TYPES[CV_indices] == 1].astype('int').tolist()})
            INDICES_DICTIONARY['GCN_ATOP'].update({'TEST_indices':TEST_indices[GCNconv.BINDING_TYPES[TEST_indices] == 1].astype('int').tolist()})
            INDICES_DICTIONARY['BINDING_TYPE_4GCN'].update({'CV_indices':CV_indices[GCNconv.BINDING_TYPES[CV_indices] > 1].astype('int').tolist()})
            INDICES_DICTIONARY['BINDING_TYPE_4GCN'].update({'TEST_indices':TEST_indices[GCNconv.BINDING_TYPES[TEST_indices] > 1].astype('int').tolist()})
            #split data into training and validation sets
            skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle = True, random_state=random_state)
            for train_index, val_index in skf.split(combined_class[CV_indices], combined_class[CV_indices]):
                INDICES_DICTIONARY['BINDING_TYPE']['train_indices'].append(CV_indices[train_index].astype('int').tolist())
                INDICES_DICTIONARY['BINDING_TYPE']['val_indices'].append(CV_indices[val_index].astype('int').tolist())
                INDICES_DICTIONARY['GCN_ATOP']['train_indices'].append(CV_indices[train_index[GCNconv.BINDING_TYPES[CV_indices[train_index]] == 1]].astype('int').tolist())
                INDICES_DICTIONARY['GCN_ATOP']['val_indices'].append(CV_indices[val_index[GCNconv.BINDING_TYPES[CV_indices[val_index]] == 1]].astype('int').tolist())
                INDICES_DICTIONARY['BINDING_TYPE_4GCN']['train_indices'].append(CV_indices[train_index[GCNconv.BINDING_TYPES[CV_indices[train_index]] > 1]].astype('int').tolist())
                INDICES_DICTIONARY['BINDING_TYPE_4GCN']['val_indices'].append(CV_indices[val_index[GCNconv.BINDING_TYPES[CV_indices[val_index]] > 1]].astype('int').tolist())    
            if write_file==True:
                with open(INDICES_FILE, 'w') as outfile:
                    json.dump(INDICES_DICTIONARY, outfile, sort_keys=True, indent=4)
                    print('Generated CV indices and saved dictionary to file ' + INDICES_FILE)
        elif read_file == True:
            with open(INDICES_FILE, 'r') as outfile:
                INDICES_DICTIONARY = json.load(outfile)
        self.INDICES_DICTIONARY = INDICES_DICTIONARY
        self.CV_SPLITS = CV_SPLITS
        self.NUM_GCN_LABELS = NUM_GCN_LABELS
    
    def set_model_parameters(self, TARGET, COVERAGE, NN_PROPERTIES, NUM_TRAIN, NUM_VAL, NUM_TEST\
                             , MIN_GCN_PER_LABEL=0, NUM_GCN_LABELS=None, GCN_ALL=False):
        try:
            INDICES_DICTIONARY = self.INDICES_DICTIONARY
        except:
            print("generate_test_cv_indices must be run first")
            raise
        assert type(COVERAGE) == float or COVERAGE==1 or COVERAGE \
        in ['low', 'high'], "Coverage should be a float, 'low', or 'high'."
        assert TARGET in ['combine_hollow_sites','binding_type','GCN'], "incorrect TARGET variable given"
        ADSORBATE = self.ADSORBATE
        NANO_PATH = self.NANO_PATH
        HIGH_COV_PATH = self.HIGH_COV_PATH
        COV_SCALE_PATH = self.COV_SCALE_PATH
        POC = self.POC
        CV_SPLITS = self.CV_SPLITS
        if TARGET == 'combine_hollow_sites': 
            MAINconv = IR_GEN(ADSORBATE, POC=1, TARGET='combine_hollow_sites'\
                         ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                         , coverage_scaling_path=COV_SCALE_PATH)
        elif TARGET == 'binding_type':
            MAINconv = IR_GEN(ADSORBATE, POC=POC, TARGET='binding_type'\
                         ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                         , coverage_scaling_path=COV_SCALE_PATH)
        elif TARGET == 'GCN':
            MAINconv = IR_GEN(ADSORBATE, POC=POC, TARGET='GCN', NUM_TARGETS=NUM_GCN_LABELS\
                         ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                         , coverage_scaling_path=COV_SCALE_PATH)
            if GCN_ALL == True:
                INCLUDED_BINDING_TYPES = list(set(MAINconv.BINDING_TYPES))
            elif GCN_ALL == False:
                INCLUDED_BINDING_TYPES = [1]
            MAINconv.get_GCNlabels(Minimum=MIN_GCN_PER_LABEL, showfigures=False, INCLUDED_BINDING_TYPES=INCLUDED_BINDING_TYPES)
            OTHER_SITESconv = IR_GEN(ADSORBATE, POC=POC, TARGET='binding_type', exclude_atop=True\
                         ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                         , coverage_scaling_path=COV_SCALE_PATH)
        
        if TARGET == 'GCN' and GCN_ALL == False:
            INDICES_VAL = [(INDICES_DICTIONARY['GCN_ATOP']['val_indices'][CV_VAL]\
                           , INDICES_DICTIONARY['BINDING_TYPE_4GCN']['val_indices'][CV_VAL])\
                           for CV_VAL in range(CV_SPLITS)]
            INDICES_TRAIN = [(INDICES_DICTIONARY['GCN_ATOP']['train_indices'][CV_VAL]\
                             , INDICES_DICTIONARY['BINDING_TYPE_4GCN']['train_indices'][CV_VAL])\
                             for CV_VAL in range(CV_SPLITS)]
            INDICES_TEST = [INDICES_DICTIONARY['GCN_ATOP']['TEST_indices']\
                            , INDICES_DICTIONARY['BINDING_TYPE_4GCN']['TEST_indices']]
            INDICES_CV_ALL = [INDICES_DICTIONARY['GCN_ATOP']['CV_indices']\
                              , INDICES_DICTIONARY['BINDING_TYPE_4GCN']['CV_indices']]
        else:
            INDICES_VAL = INDICES_DICTIONARY['BINDING_TYPE']['val_indices']
            INDICES_TRAIN = INDICES_DICTIONARY['BINDING_TYPE']['train_indices']
            INDICES_TEST = INDICES_DICTIONARY['BINDING_TYPE']['TEST_indices']
            INDICES_CV_ALL = INDICES_DICTIONARY['BINDING_TYPE']['CV_indices']
            
        if NUM_GCN_LABELS is not None:
            self.NUM_GCN_LABELS = NUM_GCN_LABELS
        if TARGET == 'GCN' and GCN_ALL == False:
            self.OTHER_SITESconv = OTHER_SITESconv
        self.INDICES_VAL = INDICES_VAL
        self.INDICES_TRAIN = INDICES_TRAIN
        self.INDICES_TEST = INDICES_TEST
        self.INDICES_CV_ALL = INDICES_CV_ALL
        self.MAINconv = MAINconv
        self.TARGET = TARGET
        self.COVERAGE = COVERAGE
        self.NN_PROPERTIES = NN_PROPERTIES
        self.NUM_TRAIN = NUM_TRAIN
        self.NUM_VAL = NUM_VAL
        self.NUM_TEST = NUM_TEST
        self.MIN_GCN_PER_LABEL = MIN_GCN_PER_LABEL
        self.GCN_ALL = GCN_ALL
        
        
        
    def run_CV(self, write_file=False, CV_RESULTS_FILE = None):
        try:
            TARGET = self.TARGET
        except:
            print("set_model_parameters must be run first")
            raise
        CV_SPLITS = self.CV_SPLITS
        ADSORBATE = self.ADSORBATE
        _run_NN = self._run_NN
        get_secondary_data = self.get_secondary_data
        CV_PATH = self.CV_PATH
        COVERAGE = self.COVERAGE
        NN_PROPERTIES = self.NN_PROPERTIES
        NUM_TRAIN = self.NUM_TRAIN
        NUM_VAL = self.NUM_VAL
        NUM_TEST = self.NUM_TEST
        INDICES_TRAIN = self.INDICES_TRAIN
        INDICES_VAL = self.INDICES_VAL
        INDICES_TEST = self.INDICES_TEST
        INDICES_CV_ALL = self.INDICES_CV_ALL
        
        
        if CV_RESULTS_FILE is None:
            CV_RESULTS_FILE = os.path.join(CV_PATH,'/CV_results_'+TARGET+'_'+str(COVERAGE)\
            +'_'+str(CV_SPLITS)+'fold'+'_reg'+'{:.2E}'.format(NN_PROPERTIES['alpha'])\
            +'_'+NN_PROPERTIES['loss']+'_'+ADSORBATE+'.json')
            
        DictList = []
        #Cross Validation
        for CV_INDEX in range(CV_SPLITS):
            print('#########################################################')
            print('#########################################################')
            print('The CV number is '+str(CV_INDEX))
            #Get validation spectra
            X_compare, y_compare = get_secondary_data(NUM_SAMPLES=NUM_VAL\
                                , INDICES=INDICES_VAL[CV_INDEX],iterations=10)
            Dict =  _run_NN(NUM_TRAIN, INDICES_TRAIN[CV_INDEX], X_compare, y_compare, IS_TEST=False) 
            DictList.append(Dict)
            
        #Train model on all CV Data and Test agains Test Set
        #Get Test Spectra
        X_compare, y_compare = get_secondary_data(NUM_TEST, INDICES_TEST\
                                ,iterations=10)
        
        Dict =_run_NN(NUM_TRAIN, INDICES_CV_ALL, X_compare, y_compare, IS_TEST=True)
        DictList.append(Dict)
        
        if write_file == True:
            with open(CV_RESULTS_FILE, 'w') as outfile:
                json_tricks.dump(DictList, outfile, sort_keys=True, indent=4)
        self.CV_RESULTS_FILE = CV_RESULTS_FILE
        return DictList
    
    def get_secondary_data(self,NUM_SAMPLES, INDICES,iterations=1):
        try:
            TARGET = self.TARGET
        except:
            print("set_model_parameters must be run first")
            raise
        GCN_ALL = self.GCN_ALL
        COVERAGE = self.COVERAGE
        MAINconv = self.MAINconv
        start = timer()
        num_samples_original = NUM_SAMPLES
        NUM_SAMPLES = int(NUM_SAMPLES/iterations)
        ADDITIONAL_POINT = int(num_samples_original-NUM_SAMPLES*iterations)
        if TARGET == 'GCN' and GCN_ALL == False:
            OTHER_SITESconv = self.OTHER_SITESconv
            X1, y = MAINconv.get_synthetic_spectra(NUM_SAMPLES+ADDITIONAL_POINT, INDICES[0], COVERAGE=COVERAGE)
            X2, y2 = OTHER_SITESconv.get_synthetic_spectra(int(NUM_SAMPLES/5), INDICES[1], COVERAGE='low')
            X = MAINconv.add_noise(X1,X2)
            del X1; del X2; del y2
        else:
            X, y = MAINconv.get_synthetic_spectra(NUM_SAMPLES, INDICES, COVERAGE=COVERAGE)      
        stop = timer()
        print('Time to generate one batch of secondary data: ' + str(stop-start))
        #Add to the validation and test sets to get more coverage options
        #(each iteration has 10 different coverage combinations
        #when TARGET in ['binding_type','combine_hollow_sites'] and COVERAGE is not 'low')
        for _ in range(iterations-1):
            if TARGET == 'GCN' and GCN_ALL == False:
                X1_2, y_2 = MAINconv.get_more_spectra(NUM_SAMPLES, INDICES[0])
                X2_2, y2_2 = OTHER_SITESconv.get_more_spectra(int(NUM_SAMPLES/5), INDICES[1])
                X_2 = MAINconv.add_noise(X1_2, X2_2)
                del X1_2; del X2_2; del y2_2
            else:
                X_2, y_2 = MAINconv.get_more_spectra(NUM_SAMPLES, INDICES)
            X = np.append(X,X_2,axis=0)
            y = np.append(y,y_2,axis=0)
            del X_2; del y_2
        stop = timer()
        print('Time to generate val/test set: ' + str(stop-start))
        return X, y

    def _run_NN(self, NUM_SAMPLES, INDICES, X_compare, y_compare, IS_TEST):
        get_secondary_data = self.get_secondary_data
        NN_PROPERTIES = self.NN_PROPERTIES
        if IS_TEST == False:
            Dict = {'Wl2_Train':[], 'Score_Train':[]\
                ,'Wl2_Val':[], 'Score_Val':[]}
            Score_compare = Dict['Score_Val']
            Wl2_compare = Dict['Wl2_Val']
        else:
            Dict = {'NN_PROPERTIES':[]
            ,'Wl2_Train':[], 'Score_Train':[]
            ,'Wl2_Test':[], 'Score_Test': []
            ,'parameters': [],'__getstate__':[]
            ,'intercepts_': []}
            Score_compare = Dict['Score_Test']
            Wl2_compare = Dict['Wl2_Test']
        
        NN = MLPRegressor(hidden_layer_sizes=NN_PROPERTIES['hidden_layer_sizes'], activation='relu', solver='adam'
                              , tol=10**-9, alpha=NN_PROPERTIES['alpha'], verbose=False, batch_size=NN_PROPERTIES['batch_size']
                              , max_iter=1, epsilon= NN_PROPERTIES['epsilon'], early_stopping=False
                              ,warm_start=True,loss=NN_PROPERTIES['loss']
                              ,learning_rate_init=NN_PROPERTIES['learning_rate_init'],out_activation='softmax')
        
        #Using Fit (w/ coverages)
        X, y = get_secondary_data(NUM_SAMPLES, INDICES, iterations=1)
        NN.fit(X, y)
        y_predict = NN.predict(X)
        del X
        ycompare_predict = NN.predict(X_compare)
        Dict['Score_Train'].append(error_metrics.get_r2(y,y_predict))
        Dict['Wl2_Train'].append(error_metrics.get_wasserstein_loss(y,y_predict))
        Score_compare.append(error_metrics.get_r2(y_compare,ycompare_predict))
        Wl2_compare.append(error_metrics.get_wasserstein_loss(y_compare,ycompare_predict))
        
        for _ in range(NN_PROPERTIES['training_sets']):
            print(_)  
            X, y = get_secondary_data(NUM_SAMPLES, INDICES, iterations=1)
            indices = np.arange(y.shape[0])    
            for __ in range(NN_PROPERTIES['epochs_per_training_set']):
                np.random.shuffle(indices)
                X = X[indices]
                y = y[indices]    
                NN.partial_fit(X,y)
                y_predict = NN.predict(X)
                ycompare_predict = NN.predict(X_compare)
                Dict['Score_Train'].append(error_metrics.get_r2(y,y_predict))
                Dict['Wl2_Train'].append(error_metrics.get_wasserstein_loss(y,y_predict))
                Score_compare.append(error_metrics.get_r2(y_compare,ycompare_predict))
                Wl2_compare.append(error_metrics.get_wasserstein_loss(y_compare,ycompare_predict))
                print('Score val/test: ' + str(Score_compare[-1]))
            del X; del y
            print('Wl2_val/test: ' + str(Wl2_compare[-1]))
            print('Wl2_Train: ' + str(Dict['Wl2_Train'][-1]))
            print('Score val/test: ' + str(Score_compare[-1]))
            print('Score_Train: ' + str(Dict['Score_Train'][-1]))
            if IS_TEST==True:
                state = deepcopy(NN.__getstate__())
                #Below code is only necessary for saving data with list of lists.
                #for key in list(state.keys()):
                #    if type(state[key]) not in [str, float, int, tuple, bool, complex, type(None), list, type(np.array(0))]:
                #        del  state[key]
                #    elif type(state[key]) in [list, tuple, type(np.array(0))]:
                #        if type(state[key][0]) not in [str, float, int, tuple, bool, complex, type(None), list, type(np.array(0))]:
                #            del state[key]
                Dict.update({'NN_PROPERTIES':NN_PROPERTIES, 'parameters':NN.get_params()
                ,'__getstate__': state})
        self.NN = NN
        return Dict

    def get_test_secondary_data(self, read_file=True):
        try:
            NUM_TEST = self.NUM_TEST
        except:
            print("set_model_parameters must be run first")
            raise
        INDICES_TEST = self.INDICES_TEST
        #Get Test Spectra
        X_Test, y_Test = self.get_secondary_data(NUM_TEST, INDICES_TEST, iterations=10)
        return X_Test, y_Test
            
    def get_test_results(self, read_file=True):
        try:
            NUM_TRAIN = self.NUM_TRAIN
        except:
            print("set_model_parameters must be run first")
            raise
        get_test_secondary_data = self.get_test_secondary_data
        _run_NN = self._run_NN
        INDICES_CV_ALL = self.INDICES_CV_ALL
        X_Test, y_Test = get_test_secondary_data(read_file=True)
        Dict =_run_NN(NUM_TRAIN, INDICES_CV_ALL, X_Test, y_Test, IS_TEST=True)
        self.X_Test = X_Test
        self.Y_Test = y_Test
        return Dict