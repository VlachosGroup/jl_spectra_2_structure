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
import pkg_resources
from timeit import default_timer as timer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from copy import deepcopy
from .neural_network import MLPRegressor
from .jl_spectra_2_structure import IR_GEN

#default values
data_path = pkg_resources.resource_filename(__name__, 'data/')

class CROSS_VALIDATION:
    """
    """
    def __init__(self, cross_validation_path, CV_SPLITS=10, GCN_LABELS=11):
        """
        """
        self.CV_SPLITS = CV_SPLITS
        self.GCN_LABELS=GCN_LABELS
        self.INDICES_FILE = os.path.join(cross_validation_path +'/cross_validation_indices_'+str(CV_SPLITS)+'fold.json')
        self.CV_PATH = cross_validation_path
        
    def generate_TEST_CV_indices(self,TEST_FRACTION=0.2,RANDOM_STATE=0, write_file=False):
        """
        """
        CV_SPLITS = self.CV_SPLITS
        GCN_LABELS = self.GCN_LABELS
        indices_file = self.INDICES_FILE
        train_test_all = {'CNCO':{'train_indices':[], 'val_indices':[]\
                         ,'CV_indices':[],'TEST_indices':[]}\
                        ,'GCN':{'train_indices':[], 'val_indices':[]\
                         ,'CV_indices':[],'TEST_indices':[]}\
                        ,'CNCO4GCN':{'train_indices':[], 'val_indices':[]\
                         ,'CV_indices':[],'TEST_indices':[]}}
        #Remove points that do not represent local minima on the PES
        GCNconv = IR_GEN(TARGET='GCN', NUM_TARGETS=GCN_LABELS)
        GCNconv.get_GCNlabels(Minimum=0,showfigures=False)
        #Get List of indices so we can split the data later
        #combined class to stratify data based on binding-type and GCN simultaneously
        combined_class = 100*GCNconv.CNCOList+GCNconv.GCNLabel
        #split data into cross validation and test set
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=RANDOM_STATE)
        for CV_index, test_index in sss.split(combined_class,combined_class):
            CV_indices = CV_index
            TEST_indices = test_index
        train_test_all['CNCO'].update({'CV_indices':CV_indices.astype('int').tolist()})
        train_test_all['CNCO'].update({'TEST_indices':TEST_indices.astype('int').tolist()})
        train_test_all['GCN'].update({'CV_indices':CV_indices[GCNconv.CNCOList[CV_indices] == 1].astype('int').tolist()})
        train_test_all['GCN'].update({'TEST_indices':TEST_indices[GCNconv.CNCOList[TEST_indices] == 1].astype('int').tolist()})
        train_test_all['CNCO4GCN'].update({'CV_indices':CV_indices[GCNconv.CNCOList[CV_indices] > 1].astype('int').tolist()})
        train_test_all['CNCO4GCN'].update({'TEST_indices':TEST_indices[GCNconv.CNCOList[TEST_indices] > 1].astype('int').tolist()})
        #split data into training and validation sets
        skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle = True, random_state=RANDOM_STATE)
        for train_index, val_index in skf.split(combined_class[CV_indices], combined_class[CV_indices]):
            train_test_all['CNCO']['train_indices'].append(CV_indices[train_index].astype('int').tolist())
            train_test_all['CNCO']['val_indices'].append(CV_indices[val_index].astype('int').tolist())
            train_test_all['GCN']['train_indices'].append(CV_indices[train_index[GCNconv.CNCOList[CV_indices[train_index]] == 1]].astype('int').tolist())
            train_test_all['GCN']['val_indices'].append(CV_indices[val_index[GCNconv.CNCOList[CV_indices[val_index]] == 1]].astype('int').tolist())
            train_test_all['CNCO4GCN']['train_indices'].append(CV_indices[train_index[GCNconv.CNCOList[CV_indices[train_index]] > 1]].astype('int').tolist())
            train_test_all['CNCO4GCN']['val_indices'].append(CV_indices[val_index[GCNconv.CNCOList[CV_indices[val_index]] > 1]].astype('int').tolist())
        
        if write_file==True:
            with open(indices_file, 'w') as outfile:
                json.dump(train_test_all, outfile, sort_keys=True, indent=4)
                print('Generated CV indices and saved dictionary to file ' + indices_file)
        self.INDICES_DICTIONARY = train_test_all
    
    def run_CV(self, TARGET, COVERAGE, PROPERTIES\
               ,num_train, num_val, num_test, read_file=True, write_file=False
               ,MinGCNperLabel=0):
        self.TARGET = TARGET
        self.COVERAGE = COVERAGE
        GCN_LABELS = self.GCN_LABELS
        CV_SPLITS = self.CV_SPLITS
        self.CV_RESULTS_FILE = os.path.join(self.CV_PATH,'/CV_results_'+TARGET+'_'+str(COVERAGE)\
        +'_'+str(CV_SPLITS)+'fold'+'_reg'+'{:.2E}'.format(PROPERTIES['alpha'])+'_'+PROPERTIES['loss']+'.json')
        results_file = self.CV_RESULTS_FILE
        if TARGET == 'CNCO1HOLLOW': 
            MAINconv = IR_GEN(TARGET='CNCO1HOLLOW', NUM_TARGETS=3)
        elif TARGET == 'CNCO':
            MAINconv = IR_GEN(TARGET='CNCO', NUM_TARGETS=4)
        elif TARGET == 'GCN':
            MAINconv = IR_GEN(TARGET='GCN', NUM_TARGETS=GCN_LABELS)
            MAINconv.get_GCNlabels(Minimum=MinGCNperLabel, showfigures=False)
            SECONDARYconv = IR_GEN(TARGET='CNCO',NUM_TARGETS=3)
            self.SECONDARYconv = SECONDARYconv
        self.MAINconv = MAINconv
        
        if read_file == True:
            with open(self.INDICES_FILE, 'r') as outfile:
                indices_dictionary = json.load(outfile)
        else:
            indices_dictionary = self.INDICES_DICTIONARY
        
        if TARGET in ['CNCO','CNCO1HOLLOW']:
            indices_val = indices_dictionary['CNCO']['val_indices']
            indices_train = indices_dictionary['CNCO']['train_indices']
            indices_test = indices_dictionary['CNCO']['TEST_indices']
            indices_CV_all = indices_dictionary['CNCO']['CV_indices']
        elif TARGET == 'GCN':
            indices_val = [(indices_dictionary['GCN']['val_indices'][CV_VAL]\
                           , indices_dictionary['CNCO4GCN']['val_indices'][CV_VAL])\
                           for CV_VAL in range(CV_SPLITS)]
            indices_train = [(indices_dictionary['GCN']['train_indices'][CV_VAL]\
                             , indices_dictionary['CNCO4GCN']['train_indices'][CV_VAL])\
                             for CV_VAL in range(CV_SPLITS)]
            indices_test = [indices_dictionary['GCN']['TEST_indices']\
                            , indices_dictionary['CNCO4GCN']['TEST_indices']]
            indices_CV_all = [indices_dictionary['GCN']['CV_indices']\
                              , indices_dictionary['CNCO4GCN']['CV_indices']]
            
        DictList = []
        #Cross Validation
        for CV_INDEX in range(CV_SPLITS):
            print('The CV number is '+str(CV_INDEX))
            #Get validation spectra
            self._get_VAL_TEST(SAMPLES=num_val, INDICES=indices_val[CV_INDEX])
            Dict =  self._get_results(num_train, indices_train[CV_INDEX]\
                                     , PROPERTIES, IS_TEST=False) 
            DictList.append(Dict)
            
        #Train model on all CV Data and Test agains Test Set
        #Get Test Spectra
        self._get_VAL_TEST(num_test, indices_test)
        
        Dict =self._get_results(num_train, indices_CV_all, PROPERTIES, IS_TEST=True)
        DictList.append(Dict)
        
        if write_file == True:
            with open(results_file, 'w') as outfile:
                json_tricks.dump(DictList, outfile, sort_keys=True, indent=4)
        
        return DictList
    
    def _get_VAL_TEST(self,SAMPLES, INDICES):
        COVERAGE = self.COVERAGE
        TARGET = self.TARGET
        MAINconv = self.MAINconv
        if TARGET == 'GCN':
            SECONDARYconv = self.SECONDARYconv
        start = timer()
        if TARGET in ['CNCO','CNCO1HOLLOW']:
            X, y = MAINconv.get_synthetic_spectra(SAMPLES, INDICES, COVERAGE=COVERAGE)
        elif TARGET == 'GCN':
            X1, y = MAINconv.get_synthetic_spectra(SAMPLES, INDICES[0], COVERAGE=COVERAGE)
            X2, y2 = SECONDARYconv.get_synthetic_spectra(int(SAMPLES/5), INDICES[1], COVERAGE='low')
            X = MAINconv.add_noise(X1,X2)
            del X1; del X2; del y2
        stop = timer()
        print('Time to generate one batch of secondary data: ' + str(stop-start))
        #Add to the validation and test sets to get more coverage options (each iteration has 10 different coverages)
        for _ in range(9):
            if TARGET in ['CNCO','CNCO1HOLLOW']:
                X_2, y_2 = MAINconv.get_more_spectra(SAMPLES, INDICES)
            elif TARGET == 'GCN':
                X1_2, y_2 = MAINconv.get_more_spectra(SAMPLES, INDICES[0])
                X2_2, y2_2 = SECONDARYconv.get_more_spectra(int(SAMPLES/5), INDICES[1])
                X_2 = MAINconv.add_noise(X1_2, X2_2)
                del X1_2; del X2_2; del y2_2
            X = np.append(X,X_2,axis=0)
            y = np.append(y,y_2,axis=0)
            del X_2; del y_2
        stop = timer()
        print('Time to generate val/test set: ' + str(stop-start))
        self.X_compare = X
        self.y_compare = y

    def _get_results(self, SAMPLES, INDICES, PROPERTIES, IS_TEST):
        X_compare = self.X_compare
        y_compare = self.y_compare
        COVERAGE = self.COVERAGE
        TARGET = self.TARGET
        MAINconv = self.MAINconv
        if TARGET == 'GCN':
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
        if TARGET in ['CNCO','CNCO1HOLLOW']:
            X, y = MAINconv.get_synthetic_spectra(SAMPLES, INDICES, COVERAGE=COVERAGE)
        elif TARGET == 'GCN':
            X1, y = MAINconv.get_synthetic_spectra(SAMPLES, INDICES[0], COVERAGE=COVERAGE)
            X2, y2 = SECONDARYconv.get_synthetic_spectra(int(SAMPLES/5), INDICES[1], COVERAGE='low')
            X = MAINconv.add_noise(X1,X2)
            del X1; del X2; del y2
        NN.fit(X, y)
        y_predict = NN.predict(X)
        del X
        ycompare_predict = NN.predict(X_compare)
        Dict['Score_Train'].append(r2(y,y_predict))
        Dict['Wl2_Train'].append(wasserstein_loss(y,y_predict))
        Score_compare.append(r2(y_compare,ycompare_predict))
        Wl2_compare.append(wasserstein_loss(y_compare,ycompare_predict))
        
        for _ in range(PROPERTIES['iterations']):
            print(_)  
            if TARGET in ['CNCO','CNCO1HOLLOW']:
                X, y = MAINconv.get_synthetic_spectra(SAMPLES, INDICES, COVERAGE=COVERAGE)
            elif TARGET == 'GCN':
                X1, y = MAINconv.get_synthetic_spectra(SAMPLES, INDICES[0], COVERAGE=COVERAGE)
                X2, y2 = SECONDARYconv.get_synthetic_spectra(int(SAMPLES/5), INDICES[1], COVERAGE='low')
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
                Dict['Score_Train'].append(r2(y,y_predict))
                Dict['Wl2_Train'].append(wasserstein_loss(y,y_predict))
                Score_compare.append(r2(y_compare,ycompare_predict))
                Wl2_compare.append(wasserstein_loss(y_compare,ycompare_predict))
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

    def get_Test(self, TARGET, COVERAGE, num_test, read_file=True, MinGCNperLabel=0):
            self.TARGET = TARGET
            self.COVERAGE = COVERAGE
            GCN_LABELS = self.GCN_LABELS
            if TARGET == 'CNCO1HOLLOW': 
                MAINconv = IR_GEN(TARGET='CNCO1HOLLOW', NUM_TARGETS=3)
            elif TARGET == 'CNCO':
                MAINconv = IR_GEN(TARGET='CNCO', NUM_TARGETS=4)
            elif TARGET == 'GCN':
                MAINconv = IR_GEN(TARGET='GCN', NUM_TARGETS=GCN_LABELS)
                MAINconv.get_GCNlabels(Minimum=MinGCNperLabel, showfigures=False)
                SECONDARYconv = IR_GEN(TARGET='CNCO',NUM_TARGETS=3)
                self.SECONDARYconv = SECONDARYconv
            self.MAINconv = MAINconv
            
            if read_file == True:
                with open(self.INDICES_FILE, 'r') as outfile:
                    indices_dictionary = json.load(outfile)
            else:
                indices_dictionary = self.INDICES_DICTIONARY
            
            if TARGET in ['CNCO','CNCO1HOLLOW']:
                indices_test = indices_dictionary['CNCO']['TEST_indices']
            elif TARGET == 'GCN':
                indices_test = [indices_dictionary['GCN']['TEST_indices']\
                                , indices_dictionary['CNCO4GCN']['TEST_indices']]
                
            #Train model on all CV Data and Test agains Test Set
            #Get Test Spectra
            self._get_VAL_TEST(num_test, indices_test)