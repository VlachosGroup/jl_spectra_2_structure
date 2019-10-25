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
from . import error_metrics

class CROSS_VALIDATION:
    """
    """
    def __init__(self, ADSORBATE='CO', poc=1, CV_SPLITS=5, NUM_GCN_LABELS=11\
                 ,cross_validation_path = None, nanoparticle_path=None\
                 ,high_coverage_path=None, coverage_scaling_path=None):
        """
        """
        assert type(ADSORBATE) == str, "ADSORBATE must be a String"
        self.INDICES_FILE = os.path.join(cross_validation_path,\
        'cross_validation_indices_'+str(CV_SPLITS)+'fold_'+ADSORBATE+'.json')
        self.CV_SPLITS = CV_SPLITS
        self.NUM_GCN_LABELS=NUM_GCN_LABELS
        self.CV_PATH = cross_validation_path
        self.ADSORBATE = ADSORBATE
        self.POC = poc
        self.NANO_PATH = nanoparticle_path
        self.HIGH_COV_PATH = high_coverage_path
        self.COV_SCALE_PATH = coverage_scaling_path
        
    def generate_TEST_CV_indices(self,TEST_FRACTION=0.2,RANDOM_STATE=0, write_file=False):
        """
        """
        CV_SPLITS = self.CV_SPLITS
        NUM_GCN_LABELS = self.NUM_GCN_LABELS
        indices_file = self.INDICES_FILE
        ADSORBATE = self.ADSORBATE
        POC = self.POC
        NANO_PATH = self.NANO_PATH
        HIGH_COV_PATH = self.HIGH_COV_PATH
        COV_SCALE_PATH = self.COV_SCALE_PATH
        INDICES_DICTIONARY = {'binding_type':{'train_indices':[], 'val_indices':[]\
                         ,'CV_indices':[],'TEST_indices':[]}\
                        ,'GCN_ALL':{'train_indices':[], 'val_indices':[]\
                         ,'CV_indices':[],'TEST_indices':[]}\
                        ,'GCN_ATOP':{'train_indices':[], 'val_indices':[]\
                         ,'CV_indices':[],'TEST_indices':[]}\
                        ,'BINDING_TYPE_4GCN':{'train_indices':[], 'val_indices':[]\
                         ,'CV_indices':[],'TEST_indices':[]}}
        GCNconv = IR_GEN(ADSORBATE, poc=POC, TARGET='GCN', NUM_TARGETS=NUM_GCN_LABELS\
                         ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                         , coverage_scaling_path=COV_SCALE_PATH)
        
        GCNconv.get_GCNlabels(Minimum=0,showfigures=False)
        #Get List of indices so we can split the data later
        #combined class to stratify data based on binding-type and GCN simultaneously
        combined_class = 100*GCNconv.BINDING_TYPES+GCNconv.GCNLabel
        #split data into cross validation and test set
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=RANDOM_STATE)
        for CV_index, test_index in sss.split(combined_class,combined_class):
            CV_indices = CV_index
            TEST_indices = test_index
        INDICES_DICTIONARY['BINDING_TYPE'].update({'CV_indices':CV_indices.astype('int').tolist()})
        INDICES_DICTIONARY['BINDING_TYPE'].update({'TEST_indices':TEST_indices.astype('int').tolist()})
        INDICES_DICTIONARY['GCN_ALL'].update({'CV_indices':CV_indices.astype('int').tolist()})
        INDICES_DICTIONARY['GCN_ALL'].update({'TEST_indices':TEST_indices.astype('int').tolist()})
        INDICES_DICTIONARY['GCN_ATOP'].update({'CV_indices':CV_indices[GCNconv.BINDING_TYPES[CV_indices] == 1].astype('int').tolist()})
        INDICES_DICTIONARY['GCN_ATOP'].update({'TEST_indices':TEST_indices[GCNconv.BINDING_TYPES[TEST_indices] == 1].astype('int').tolist()})
        INDICES_DICTIONARY['BINDING_TYPE_4GCN'].update({'CV_indices':CV_indices[GCNconv.BINDING_TYPES[CV_indices] > 1].astype('int').tolist()})
        INDICES_DICTIONARY['BINDING_TYPE_4GCN'].update({'TEST_indices':TEST_indices[GCNconv.BINDING_TYPES[TEST_indices] > 1].astype('int').tolist()})
        #split data into training and validation sets
        skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle = True, random_state=RANDOM_STATE)
        for train_index, val_index in skf.split(combined_class[CV_indices], combined_class[CV_indices]):
            INDICES_DICTIONARY['BINDING_TYPE']['train_indices'].append(CV_indices[train_index].astype('int').tolist())
            INDICES_DICTIONARY['BINDING_TYPE']['val_indices'].append(CV_indices[val_index].astype('int').tolist())
            INDICES_DICTIONARY['GCN_ALL']['train_indices'].append(CV_indices[train_index].astype('int').tolist())
            INDICES_DICTIONARY['GCN_ALL']['val_indices'].append(CV_indices[val_index].astype('int').tolist())
            INDICES_DICTIONARY['GCN_ATOP']['train_indices'].append(CV_indices[train_index[GCNconv.BINDING_TYPES[CV_indices[train_index]] == 1]].astype('int').tolist())
            INDICES_DICTIONARY['GCN_ATOP']['val_indices'].append(CV_indices[val_index[GCNconv.BINDING_TYPES[CV_indices[val_index]] == 1]].astype('int').tolist())
            INDICES_DICTIONARY['BINDING_TYPE_4GCN']['train_indices'].append(CV_indices[train_index[GCNconv.BINDING_TYPES[CV_indices[train_index]] > 1]].astype('int').tolist())
            INDICES_DICTIONARY['BINDING_TYPE_4GCN']['val_indices'].append(CV_indices[val_index[GCNconv.BINDING_TYPES[CV_indices[val_index]] > 1]].astype('int').tolist())
        
        if write_file==True:
            with open(indices_file, 'w') as outfile:
                json.dump(INDICES_DICTIONARY, outfile, sort_keys=True, indent=4)
                print('Generated CV indices and saved dictionary to file ' + indices_file)
        self.INDICES_DICTIONARY = INDICES_DICTIONARY
    
    def run_CV(self, TARGET, COVERAGE, PROPERTIES, num_train, num_val, num_test\
               , read_file=True, write_file=False, MinGCNperLabel=0, GCN_ALL=False):
        assert type(COVERAGE) == float or COVERAGE==1 or COVERAGE \
        in ['low', 'high'], "Coverage should be a float, 'low', or 'high'."
        NUM_GCN_LABELS = self.NUM_GCN_LABELS
        CV_SPLITS = self.CV_SPLITS
        ADSORBATE = self.ADSORBATE
        _get_results = self._get_results
        _get_VAL_TEST = self._get_VAL_TEST
        INDICES_FILE = self.INDICES_FILE
        INDICES_DICTIONARY = self.INDICES_DICTIONARY
        CV_PATH = self.CV_PATH
        NANO_PATH = self.NANO_PATH
        HIGH_COV_PATH = self.HIGH_COV_PATH
        COV_SCALE_PATH = self.COV_SCALE_PATH
        POC = self.POC
        CV_RESULTS_FILE = os.path.join(CV_PATH,'/CV_results_'+TARGET+'_'+str(COVERAGE)\
        +'_'+str(CV_SPLITS)+'fold'+'_reg'+'{:.2E}'.format(PROPERTIES['alpha'])\
        +'_'+PROPERTIES['loss']+'_'+ADSORBATE+'.json')
        if TARGET == 'combine_hollow_sites': 
            MAINconv = IR_GEN(ADSORBATE, poc=1, TARGET='combine_hollow_sites'\
                         ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                         , coverage_scaling_path=COV_SCALE_PATH)
        elif TARGET == 'binding_type':
            MAINconv = IR_GEN(ADSORBATE, poc=POC, TARGET='binding_type'\
                         ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                         , coverage_scaling_path=COV_SCALE_PATH)
        elif TARGET == 'GCN':
            if GCN_ALL == True:
                INCLUDED_BINDING_TYPES = list(set(MAINconv.BINDING_TYPES))
            elif GCN_ALL == False:
                INCLUDED_BINDING_TYPES = [1]
            MAINconv.get_GCNlabels(Minimum=MinGCNperLabel, showfigures=False, INCLUDED_BINDING_TYPES=INCLUDED_BINDING_TYPES)
            SECONDARYconv = IR_GEN(ADSORBATE, poc=POC, TARGET='binding_type', exclude_atop=True\
                         ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                         , coverage_scaling_path=COV_SCALE_PATH)
        
        if read_file == True:
            with open(INDICES_FILE, 'r') as outfile:
                INDICES_DICTIONARY = json.load(outfile)
        
        if TARGET in ['binding_type','combine_hollow_sites']:
            indices_val = INDICES_DICTIONARY['BINDING_TYPE']['val_indices']
            indices_train = INDICES_DICTIONARY['BINDING_TYPE']['train_indices']
            indices_test = INDICES_DICTIONARY['BINDING_TYPE']['TEST_indices']
            indices_CV_all = INDICES_DICTIONARY['BINDING_TYPE']['CV_indices']
        elif TARGET == 'GCN' and GCN_ALL == False:
            indices_val = [(INDICES_DICTIONARY['GCN_ATOP']['val_indices'][CV_VAL]\
                           , INDICES_DICTIONARY['BINDING_TYPE_4GCN']['val_indices'][CV_VAL])\
                           for CV_VAL in range(CV_SPLITS)]
            indices_train = [(INDICES_DICTIONARY['GCN_ATOP']['train_indices'][CV_VAL]\
                             , INDICES_DICTIONARY['BINDING_TYPE_4GCN']['train_indices'][CV_VAL])\
                             for CV_VAL in range(CV_SPLITS)]
            indices_test = [INDICES_DICTIONARY['GCN_ATOP']['TEST_indices']\
                            , INDICES_DICTIONARY['BINDING_TYPE_4GCN']['TEST_indices']]
            indices_CV_all = [INDICES_DICTIONARY['GCN_ATOP']['CV_indices']\
                              , INDICES_DICTIONARY['BINDING_TYPE_4GCN']['CV_indices']]
        elif TARGET == 'GCN' and GCN_ALL == True:
            indices_val = INDICES_DICTIONARY['GCN_ALL']['val_indices']
            indices_train = INDICES_DICTIONARY['GCN_ALL']['train_indices']
            indices_test = INDICES_DICTIONARY['GCN_ALL']['TEST_indices']
            indices_CV_all = INDICES_DICTIONARY['GCN_ALL']['CV_indices']
            
        DictList = []
        #Cross Validation
        for CV_INDEX in range(CV_SPLITS):
            print('The CV number is '+str(CV_INDEX))
            #Get validation spectra
            _get_VAL_TEST(NUM_SAMPLES=num_val, INDICES=indices_val[CV_INDEX], GCN_ALL)
            Dict =  _get_results(num_train, indices_train[CV_INDEX]\
                                     , PROPERTIES, IS_TEST=False, GCN_ALL) 
            DictList.append(Dict)
            
        #Train model on all CV Data and Test agains Test Set
        #Get Test Spectra
        _get_VAL_TEST(num_test, indices_test, GCN_ALL)
        
        Dict =_get_results(num_train, indices_CV_all, PROPERTIES, IS_TEST=True, GCN_ALL)
        DictList.append(Dict)
        
        if write_file == True:
            with open(CV_RESULTS_FILE, 'w') as outfile:
                json_tricks.dump(DictList, outfile, sort_keys=True, indent=4)
        if TARGET == 'GCN' and GCN_ALL == False:
            self.SECONDARYconv = SECONDARYconv
        self.CV_RESULTS_FILE = CV_RESULTS_FILE
        self.MAINconv = MAINconv
        self.TARGET = TARGET
        self.COVERAGE = COVERAGE
        self.GCN_ALL = GCN_ALL
        return DictList
    
    def _get_VAL_TEST(self,NUM_SAMPLES, INDICES, GCN_ALL):
        COVERAGE = self.COVERAGE
        TARGET = self.TARGET
        MAINconv = self.MAINconv
        start = timer()
        if TARGET == 'GCN' and GCN_ALL == False:
            SECONDARYconv = self.SECONDARYconv
            X1, y = MAINconv.get_synthetic_spectra(NUM_SAMPLES, INDICES[0], COVERAGE=COVERAGE)
            X2, y2 = SECONDARYconv.get_synthetic_spectra(int(NUM_SAMPLES/5), INDICES[1], COVERAGE='low')
            X = MAINconv.add_noise(X1,X2)
            del X1; del X2; del y2
        elif TARGET in ['binding_type','combine_hollow_sites'] or GCN_ALL == True:
            X, y = MAINconv.get_synthetic_spectra(NUM_SAMPLES, INDICES, COVERAGE=COVERAGE)      
        stop = timer()
        print('Time to generate one batch of secondary data: ' + str(stop-start))
        #Add to the validation and test sets to get more coverage options (each iteration has 10 different coverages)
        for _ in range(9):
            if TARGET in ['binding_type','combine_hollow_sites'] or GCN_ALL == True:
                X_2, y_2 = MAINconv.get_more_spectra(NUM_SAMPLES, INDICES)
            elif TARGET == 'GCN':
                X1_2, y_2 = MAINconv.get_more_spectra(NUM_SAMPLES, INDICES[0])
                X2_2, y2_2 = SECONDARYconv.get_more_spectra(int(NUM_SAMPLES/5), INDICES[1])
                X_2 = MAINconv.add_noise(X1_2, X2_2)
                del X1_2; del X2_2; del y2_2
            X = np.append(X,X_2,axis=0)
            y = np.append(y,y_2,axis=0)
            del X_2; del y_2
        stop = timer()
        print('Time to generate val/test set: ' + str(stop-start))
        self.X_compare = X
        self.y_compare = y

    def _get_results(self, NUM_SAMPLES, INDICES, PROPERTIES, IS_TEST, GCN_ALL):
        X_compare = self.X_compare
        y_compare = self.y_compare
        COVERAGE = self.COVERAGE
        TARGET = self.TARGET
        MAINconv = self.MAINconv
        if TARGET == 'GCN' and GCN_ALL == False:
            SECONDARYconv = self.SECONDARYconv
        if IS_TEST == False:
            Dict = {'Wl2_Train':[], 'Score_Train':[]\
                ,'Wl2_Val':[], 'Score_Val':[]}
            Score_compare = Dict['Score_Val']
            Wl2_compare = Dict['Wl2_Val']
        else:
            Dict = {'properties':[]
            ,'Wl2_Train':[], 'Score_Train':[]
            ,'Wl2_Test':[], 'Score_Test': []
            ,'parameters': [],'__getstate__':[]
            ,'intercepts_': []}
            Score_compare = Dict['Score_Test']
            Wl2_compare = Dict['Wl2_Test']
        
        NN = MLPRegressor(hidden_layer_sizes=PROPERTIES['hidden_layer_sizes'], activation='relu', solver='adam'
                              , tol=10**-9, alpha=PROPERTIES['alpha'], verbose=False, batch_size=PROPERTIES['batch_size']
                              , max_iter=PROPERTIES['initial_epochs'], epsilon= PROPERTIES['epsilon'], early_stopping=False
                              ,warm_start=True,loss=PROPERTIES['loss']
                              ,learning_rate_init=PROPERTIES['learning_rate_init'],out_activation='softmax')
        
        #Using Fit (w/ coverages)
        if TARGET in ['binding_type','combine_hollow_sites'] or GCN_ALL == True:
            X, y = MAINconv.get_synthetic_spectra(NUM_SAMPLES, INDICES, COVERAGE=COVERAGE)
        elif TARGET == 'GCN':
            X1, y = MAINconv.get_synthetic_spectra(NUM_SAMPLES, INDICES[0], COVERAGE=COVERAGE)
            X2, y2 = SECONDARYconv.get_synthetic_spectra(int(NUM_SAMPLES/5), INDICES[1], COVERAGE='low')
            X = MAINconv.add_noise(X1,X2)
            del X1; del X2; del y2
        NN.fit(X, y)
        y_predict = NN.predict(X)
        del X
        ycompare_predict = NN.predict(X_compare)
        Dict['Score_Train'].append(error_metrics.get_r2(y,y_predict))
        Dict['Wl2_Train'].append(error_metrics.get_wasserstein_loss(y,y_predict))
        Score_compare.append(error_metrics.get_r2(y_compare,ycompare_predict))
        Wl2_compare.append(error_metrics.get_wasserstein_loss(y_compare,ycompare_predict))
        
        for _ in range(PROPERTIES['iterations']):
            print(_)  
            if TARGET in ['binding_type','combine_hollow_sites']  or GCN_ALL == True:
                X, y = MAINconv.get_synthetic_spectra(NUM_SAMPLES, INDICES, COVERAGE=COVERAGE)
            elif TARGET == 'GCN':
                X1, y = MAINconv.get_synthetic_spectra(NUM_SAMPLES, INDICES[0], COVERAGE=COVERAGE)
                X2, y2 = SECONDARYconv.get_synthetic_spectra(int(NUM_SAMPLES/5), INDICES[1], COVERAGE='low')
                X = MAINconv.add_noise(X1,X2)
                del X1; del X2; del y2
            indices = np.arange(y.shape[0])    
            for __ in range(PROPERTIES['epochs']):
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
                NoneType = type(None)
                array_type = type(np.array(0))
                for key in list(state.keys()):
                    if type(state[key]) not in [str, float, int, tuple, bool, complex, NoneType, list, array_type]:
                        del  state[key]
                    elif type(state[key]) in [list, tuple, array_type]:
                        if type(state[key][0]) not in [str, float, int, tuple, bool, complex, NoneType, list, array_type]:
                            del state[key]
                Dict.update({'properties':PROPERTIES, 'parameters':NN.get_params()
                ,'__getstate__': state})
        return Dict

    def get_Test(self, TARGET, COVERAGE, num_test, read_file=True, MinGCNperLabel=0, GCN_ALL=False):
            INDICES_FILE = self.INDICES_FILE
            NUM_GCN_LABELS = self.NUM_GCN_LABELS
            try:
                INDICES_DICTIONARY = self.INDICES_DICTIONARY
            except:
                "generate_TEST_CV_indices must be run before get_Test."
                raise
                
            if TARGET == 'combine_hollow_sites': 
                MAINconv = IR_GEN(ADSORBATE, poc=1, TARGET='combine_hollow_sites'\
                             ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                             , coverage_scaling_path=COV_SCALE_PATH)
            elif TARGET == 'binding_type':
                MAINconv = IR_GEN(ADSORBATE, poc=POC, TARGET='binding_type'\
                             ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                             , coverage_scaling_path=COV_SCALE_PATH)
            elif TARGET == 'GCN':
                MAINconv = IR_GEN(ADSORBATE, poc=POC, TARGET='GCN', NUM_TARGETS=NUM_GCN_LABELS\
                             ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                             , coverage_scaling_path=COV_SCALE_PATH)
                if GCN_ALL == True:
                    INCLUDED_BINDING_TYPES = list(set(MAINconv.BINDING_TYPES))
                elif GCN_ALL == False:
                    INCLUDED_BINDING_TYPES = [1]
                MAINconv.get_GCNlabels(Minimum=MinGCNperLabel, showfigures=False, INCLUDED_BINDING_TYPES=INCLUDED_BINDING_TYPES)
                SECONDARYconv = IR_GEN(ADSORBATE, poc=POC, TARGET='binding_type', exclude_atop=True\
                             ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                             , coverage_scaling_path=COV_SCALE_PATH)
            
            if read_file == True:
                with open(INDICES_FILE, 'r') as outfile:
                    INDICES_DICTIONARY = json.load(outfile)
            
            if TARGET in ['binding_type','combine_hollow_sites']:
                indices_test = INDICES_DICTIONARY['BINDING_TYPE']['TEST_indices']
            elif TARGET == 'GCN' and GCN_ALL == False:
                indices_test = [INDICES_DICTIONARY['GCN_ATOP']['TEST_indices']\
                                , INDICES_DICTIONARY['BINDING_TYPE_4GCN']['TEST_indices']]
            elif GCN_ALL == True:
                indices_test = INDICES_DICTIONARY['GCN_ALL']['TEST_indices']
            
            #Get Test Spectra
            if TARGET == 'GCN' and GCN_ALL == False:
                self.SECONDARYconv = SECONDARYconv
            self.TARGET = TARGET
            self.COVERAGE = COVERAGE
            self.MAINconv = MAINconv
            self._get_VAL_TEST(num_test, indices_test, GCN_ALL)