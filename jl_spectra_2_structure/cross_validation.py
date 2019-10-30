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
from .jl_spectra_2_structure import get_default_data_paths
from . import error_metrics
import multiprocessing
import psutil

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
           , cov_scale_path = get_default_data_paths(ADSORBATE)
        
        if cross_validation_path is None:
            cross_validation_path = self._get_default_cross_validation_path()
        self.CV_PATH = cross_validation_path
        self.ADSORBATE = ADSORBATE
        self.POC = POC
        self.NANO_PATH = nanoparticle_path
        self.HIGH_COV_PATH = high_coverage_path
        self.COV_SCALE_PATH = coverage_scaling_path
        
    def _get_default_cross_validation_path(self):
         work_dir = os.getcwd()
         cross_validation_path = os.path.join(work_dir,'cross_validation')
         already_exists = os.path.isdir(cross_validation_path)
         if already_exists == False:
             os.mkdir(cross_validation_path)
         return cross_validation_path

        
    def generate_test_cv_indices(self, CV_SPLITS=5, NUM_GCN_LABELS=11, GCN_ALL = False\
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
            if GCN_ALL == True:
                GCNconv.get_GCNlabels(Minimum=2,showfigures=False,INCLUDED_BINDING_TYPES='ALL')
            else:
                GCNconv.get_GCNlabels(Minimum=2,showfigures=False,INCLUDED_BINDING_TYPES=[1])
            #Get List of indices so we can split the data later
            #combined class to stratify data based on binding-type and GCN simultaneously
            GCNlabels = GCNconv.GCNlabels
            combined_class = GCNconv.BINDING_TYPES+10*GCNlabels
            classes_with_counts = np.unique(combined_class,return_counts=True)
            reduce_count=0
            num_classes = classes_with_counts[0].size
            for count in range(num_classes):
                if classes_with_counts[1][count-reduce_count] < 2:
                    BINDING_CLASS = classes_with_counts[0][count-reduce_count] - 10*GCNlabels[GCNlabels==count+1][0]
                    reduce_count+=1
                    print('combining classes to meet kfold constraints')
                    if count < num_classes-1:
                        GCNlabels[np.all((GCNconv.BINDING_TYPES==BINDING_CLASS,GCNlabels==count+1),axis=0)] += 1
                    else:
                        GCNlabels[np.all((GCNconv.BINDING_TYPES==BINDING_CLASS,GCNlabels==count+1),axis=0)] -= 1
                    combined_class = GCNconv.BINDING_TYPES + 10*GCNlabels
                    classes_with_counts = np.unique(combined_class,return_counts=True)            
            print('The class and number in each class for fold generation is '+str(classes_with_counts))
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
            with open(INDICES_FILE, 'r') as infile:
                INDICES_DICTIONARY = json.load(infile)
        self.INDICES_DICTIONARY = INDICES_DICTIONARY
        self.CV_SPLITS = CV_SPLITS
        self.NUM_GCN_LABELS = NUM_GCN_LABELS
    
    def set_model_parameters(self, TARGET, COVERAGE, NN_PROPERTIES, NUM_TRAIN, NUM_VAL, NUM_TEST\
                             , MIN_GCN_PER_LABEL=0, NUM_GCN_LABELS=None, GCN_ALL = False\
                             ,LOW_FREQUENCY=200, HIGH_FREQUENCY=2200, ENERGY_POINTS=501):
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
        if NUM_GCN_LABELS is None:
            NUM_GCN_LABELS = self.NUM_GCN_LABELS
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
            if GCN_ALL == False:
                INCLUDED_BINDING_TYPES=[1]
            else:
                INCLUDED_BINDING_TYPES='ALL'
            MAINconv.get_GCNlabels(Minimum=MIN_GCN_PER_LABEL, showfigures=False, INCLUDED_BINDING_TYPES=INCLUDED_BINDING_TYPES)
            OTHER_SITESconv = IR_GEN(ADSORBATE, POC=POC, TARGET='binding_type', EXCLUDE_ATOP=True\
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
        self.LOW_FREQUENCY = LOW_FREQUENCY
        self.HIGH_FREQUENCY = HIGH_FREQUENCY
        self.ENERGY_POINTS = ENERGY_POINTS
        self.GCN_ALL = GCN_ALL
        
    def set_nn_parameters(self, NN_PROPERTIES):
        self.NN_PROPERTIES = NN_PROPERTIES
        
    def _set_pc_loadings(self,NUM_PCs,NUM_SAMPLES = 100000):
        """
        Returns principal component loadings after performing SVD on the
        matrix of pure spectra where $pure-single_spectra = USV^T$
        
        Parameters
        ----------
        NUM_PCs : int
            The number of principal components of the spectra to keep.
            
        Attributes
        ----------
        TOTAL_EXPLAINED_VARIANCE : numpy.ndarray
            Total explained variance by the $n$ principal components where
            $n=NUM_PCs$
                  
        Returns
        -------
        PC_loadings : numpy.ndarray
            The first loadings of the first $N$ principal components where $N$
            is equal to NUM_PCs. $PC_loadings = V$ 
        
        """
        get_secondary_data = self.get_secondary_data
        INDICES_CV_ALL = self.INDICES_CV_ALL
        X_VAL, y_VAL = get_secondary_data(NUM_SAMPLES, INDICES_CV_ALL, iterations=10)
        FEATURE_MEANS = X_VAL.mean(axis=0,keepdims=True)
        X = (X_VAL - FEATURE_MEANS)
        U, S, V = np.linalg.svd(X, full_matrices=False)
        PC_loadings = V[:NUM_PCs]
        self.FEATURE_MEANS = FEATURE_MEANS
        self.TOTAL_EXPLAINED_VARIANCE = np.sum(S[:NUM_PCs]**2)/np.sum(S**2)
        self.EXPLAINED_VARIANCE = S[:NUM_PCs]**2/np.sum(S**2)
        self.PC_LOADINGS = PC_loadings
    
    def _transform_spectra(self,spectra):
        """
        Returns principal component loadings of the spectra as well as the
        matrix that multiplies the principal components of a given mixed
        spectra to return.
                  
        Parameters
        ----------
        NUM_PCs : int
            The number of principal components of the spectra to keep.
        
        Returns
        -------
        PC_loadings : numpy.ndarray
            The first loadings of the first $N$ principal components where $N$
            is equal to the number of pure-component species on which model is
            trained.
            
        PCs_2_concentrations : numpy.ndarray
            Regressed matrix to compute concentrations given the principal
            components of a mixed spectra.
        
        """
        FEATURE_MEANS = self.FEATURE_MEANS
        X = (spectra - FEATURE_MEANS)
        PC_LOADINGS = self.PC_LOADINGS
        PCs = np.dot(X,PC_LOADINGS.T)
        return PCs 
    
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
        
        #Decide if validation spectra can be created at once
        if ((COVERAGE == 'high' or type(COVERAGE) in [float, int]) and TARGET in ['binding_type', 'combine_hollow_sites'])\
        or (COVERAGE == 'high' and TARGET == 'GCN'):
            iterations = 10
        else:
            iterations = 1
        
        if CV_RESULTS_FILE is None:
            CV_RESULTS_FILE = os.path.join(CV_PATH,'CV_results_'+TARGET+'_'+str(COVERAGE)\
            +'_'+str(CV_SPLITS)+'fold'+'_reg'+'{:.2E}'.format(NN_PROPERTIES['alpha'])\
            +'_'+NN_PROPERTIES['loss']+'_'+ADSORBATE+'.json')
            
        DictList = [{} for _ in range(CV_SPLITS)]
        #Cross Validation
        start = timer()
        for CV_INDEX in range(CV_SPLITS):
            print('#########################################################')
            print('#########################################################')
            print('The CV number is '+str(CV_INDEX+1))
            #Get validation spectra
            X_compare, y_compare = get_secondary_data(NUM_SAMPLES=NUM_VAL\
                                , INDICES=INDICES_VAL[CV_INDEX],iterations=iterations)
            Dict =  _run_NN(NUM_TRAIN, INDICES_TRAIN[CV_INDEX], X_compare, y_compare, IS_TEST=False) 
            DictList[CV_INDEX] = Dict
        stop = timer()
        #Train model on all CV Data and Test agains Test Set
        #Get Test Spectra
        X_compare, y_compare = get_secondary_data(NUM_TEST, INDICES_TEST\
                                ,iterations=iterations)
        
        Dict =_run_NN(NUM_TRAIN, INDICES_CV_ALL, X_compare, y_compare, IS_TEST=True)
        DictList.append(Dict)
        stop = timer()
        print('#########################################################')
        print('#########################################################')
        print('Time to run the CV+Test: ' + str(stop-start))
        
        if write_file == True:
            with open(CV_RESULTS_FILE, 'w') as outfile:
                json_tricks.dump(DictList, outfile, sort_keys=True, indent=4)
        self.CV_RESULTS_FILE = CV_RESULTS_FILE
        return DictList
    
    def run_CV_multiprocess(self,write_file=False, CV_RESULTS_FILE = None, num_procs=None):
        try:
            TARGET = self.TARGET
        except:
            print("set_model_parameters must be run first")
            raise
        run_single_CV = self.run_single_CV
        CV_SPLITS = self.CV_SPLITS
        ADSORBATE = self.ADSORBATE
        CV_PATH = self.CV_PATH
        COVERAGE = self.COVERAGE
        NN_PROPERTIES = self.NN_PROPERTIES
        NUM_TRAIN = self.NUM_TRAIN
        NUM_VAL = self.NUM_VAL
        NUM_TEST = self.NUM_TEST
        if CV_RESULTS_FILE is None:
            CV_RESULTS_FILE = os.path.join(CV_PATH,'CV_results_'+TARGET+'_'+str(COVERAGE)\
            +'_'+str(CV_SPLITS)+'fold'+'_reg'+'{:.2E}'.format(NN_PROPERTIES['alpha'])\
            +'_'+NN_PROPERTIES['loss']+'_'+ADSORBATE+'.json')
            
        CV_INDEX_or_TEST = [i for i in range(CV_SPLITS)]
        CV_INDEX_or_TEST.append('TEST')
        #Cross Validation and Test Run
        start = timer()
        #memory requirment since spectra are arrays of 501 real numbers of float64
        memory_requriement_per_run = 8*501*(NUM_TRAIN+(max(NUM_VAL,NUM_TEST)))
        memory_available = psutil.virtual_memory()[0]
        max_runs_with_memory = int((0.95*memory_available-1)/memory_requriement_per_run)
        cpu_cores = multiprocessing.cpu_count()
        num_procs_given = num_procs
        if num_procs is None:
            num_procs=cpu_cores
        num_procs = min(max_runs_with_memory, cpu_cores,num_procs,CV_SPLITS+1)
        print('#########################################################')
        print('#########################################################')
        if num_procs_given is not None:
            if num_procs < num_procs_given:
                if num_procs == CV_SPLITS+1:
                    print('Resetting number of processes to '+str(num_procs)+', which is the necessary number of model calls.')
                elif num_procs == max_runs_with_memory:
                    print('Resetting number of processes to '+str(num_procs)+' due to memory limitations.')
                elif num_procs == cpu_cores:
                    print('Resetting number of processes to '+str(num_procs)+' which is the number of available cores.')
        else:
            if num_procs == CV_SPLITS+1:
                print('Setting number of processes to '+str(num_procs)+', which is the necessary number of model calls.')
            elif num_procs == max_runs_with_memory:
                print('Setting number of processes to '+str(num_procs)+' due to memory limitations.')
            elif num_procs == cpu_cores:
                print('Setting number of processes to '+str(num_procs)+' which is the number of available cores.')
                    
                
            
        pool = multiprocessing.Pool(processes=num_procs)
        DictList = pool.imap(run_single_CV,CV_INDEX_or_TEST)
        DictList = [Dict for Dict in DictList]
        stop = timer()
        print('#########################################################')
        print('#########################################################')
        print('Time to run the CV+Test: ' + str(stop-start))
        
        
        if write_file == True:
            with open(CV_RESULTS_FILE, 'w') as outfile:
                json_tricks.dump(DictList, outfile, sort_keys=True, indent=4)
        self.CV_RESULTS_FILE = CV_RESULTS_FILE
        return DictList
    
    def run_single_CV(self, CV_INDEX_or_TEST):
        try:
            NUM_TRAIN = self.NUM_TRAIN
        except:
            print("set_model_parameters must be run first")
            raise
        _run_NN = self._run_NN
        get_secondary_data = self.get_secondary_data
        NUM_VAL = self.NUM_VAL
        INDICES_TRAIN = self.INDICES_TRAIN
        INDICES_VAL = self.INDICES_VAL
        NUM_TEST = self.NUM_TEST
        INDICES_TEST = self.INDICES_TEST
        INDICES_CV_ALL = self.INDICES_CV_ALL
        TARGET = self.TARGET
        COVERAGE = self.COVERAGE
        
        #Decide if validation spectra can be created at once
        if ((COVERAGE == 'high' or type(COVERAGE) in [float, int]) and TARGET in ['binding_type', 'combine_hollow_sites'])\
        or (COVERAGE == 'high' and TARGET == 'GCN'):
            iterations = 10
        else:
            iterations = 1
        
        if CV_INDEX_or_TEST != 'TEST':
            print('#########################################################')
            print('#########################################################')
            print('The CV number is '+str(CV_INDEX_or_TEST+1))
            start = timer()
            X_compare, y_compare = get_secondary_data(NUM_SAMPLES=NUM_VAL\
                                , INDICES=INDICES_VAL[CV_INDEX_or_TEST],iterations=iterations)
            stop = timer()
            print('Time to generate one batch of secondary data is ' + str(stop-start))
            
            Dict =  _run_NN(NUM_TRAIN, INDICES_TRAIN[CV_INDEX_or_TEST], X_compare, y_compare, IS_TEST=False) 
        else:
            print('#########################################################')
            print('#########################################################')
            print('Training on whole CV population and Testing on test set')
                
            X_compare, y_compare = get_secondary_data(NUM_TEST, INDICES_TEST\
                                    ,iterations=iterations)
            
            Dict =_run_NN(NUM_TRAIN, INDICES_CV_ALL, X_compare, y_compare, IS_TEST=True)
        return Dict
    
    def get_secondary_data(self,NUM_SAMPLES, INDICES,iterations=1):
        try:
            TARGET = self.TARGET
        except:
            print("set_model_parameters must be run first")
            raise
        GCN_ALL = self.GCN_ALL
        COVERAGE = self.COVERAGE
        MAINconv = deepcopy(self.MAINconv)
        LOW_FREQUENCY = self.LOW_FREQUENCY
        HIGH_FREQUENCY = self.HIGH_FREQUENCY
        ENERGY_POINTS = self.ENERGY_POINTS
        num_samples_original = NUM_SAMPLES
        NUM_SAMPLES = int(NUM_SAMPLES/iterations)
        ADDITIONAL_POINT = int(num_samples_original-NUM_SAMPLES*iterations)
        if TARGET == 'GCN' and GCN_ALL == False:
            OTHER_SITESconv = deepcopy(self.OTHER_SITESconv)
            X1, y = MAINconv.get_synthetic_spectra(NUM_SAMPLES+ADDITIONAL_POINT, INDICES[0], COVERAGE=COVERAGE\
                                                   , LOW_FREQUENCY=LOW_FREQUENCY, HIGH_FREQUENCY=HIGH_FREQUENCY, ENERGY_POINTS=ENERGY_POINTS)
            X2, y2 = OTHER_SITESconv.get_synthetic_spectra(int(NUM_SAMPLES/5), INDICES[1], COVERAGE='low'\
                                                           , LOW_FREQUENCY=LOW_FREQUENCY, HIGH_FREQUENCY=HIGH_FREQUENCY, ENERGY_POINTS=ENERGY_POINTS)
            X = MAINconv.add_noise(X1,X2)
            del X1; del X2; del y2
        else:
            X, y = MAINconv.get_synthetic_spectra(NUM_SAMPLES, INDICES, COVERAGE=COVERAGE\
                                                  , LOW_FREQUENCY=LOW_FREQUENCY, HIGH_FREQUENCY=HIGH_FREQUENCY, ENERGY_POINTS=ENERGY_POINTS)
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
        return (X, y)

    def _run_NN(self, NUM_SAMPLES, INDICES, X_compare, y_compare, IS_TEST):
        _transform_spectra = self._transform_spectra
        get_secondary_data = self.get_secondary_data
        NN_PROPERTIES = self.NN_PROPERTIES
        X_compare = _transform_spectra(X_compare)
        if IS_TEST == False:
            Dict = {'Wl2_Train':[], 'Score_Train':[]\
                ,'Wl2_Val':[], 'Score_Val':[]}
            Score_compare = Dict['Score_Val']
            Wl2_compare = Dict['Wl2_Val']
        else:
            Dict = {'NN_PROPERTIES':[]
            ,'Wl2_Train':[], 'Score_Train':[]
            ,'Wl2_Test':[], 'Score_Test': []
            ,'parameters': [],'__getstate__':[]}
            Score_compare = Dict['Score_Test']
            Wl2_compare = Dict['Wl2_Test']
        
        NN = MLPRegressor(hidden_layer_sizes=NN_PROPERTIES['hidden_layer_sizes'], activation='relu', solver='adam'
                              , tol=10**-9, alpha=NN_PROPERTIES['alpha'], verbose=False, batch_size=NN_PROPERTIES['batch_size']
                              , max_iter=1, epsilon= NN_PROPERTIES['epsilon'], early_stopping=False
                              ,warm_start=True,loss=NN_PROPERTIES['loss']
                              ,learning_rate_init=NN_PROPERTIES['learning_rate_init'],out_activation='softmax')
        
        #Using Fit (w/ coverages)
        if IS_TEST == True:
            start = timer()
        X, y = get_secondary_data(NUM_SAMPLES, INDICES, iterations=1)
        if IS_TEST == True:
            stop = timer()
            print('Time to generate one batch of secondary data is ' + str(stop-start))
        X = _transform_spectra(X)
        NN.partial_fit(X, y)
        y_predict = NN.predict(X)
        ycompare_predict = NN.predict(X_compare)
        Dict['Score_Train'].append(error_metrics.get_r2(y,y_predict))
        Dict['Wl2_Train'].append(error_metrics.get_wasserstein_loss(y,y_predict))
        Score_compare.append(error_metrics.get_r2(y_compare,ycompare_predict))
        Wl2_compare.append(error_metrics.get_wasserstein_loss(y_compare,ycompare_predict))
        for i in range(NN_PROPERTIES['training_sets']):
            if IS_TEST == True:
                print('Training set number ' + str(i+1))
            if i > 0:
                X, y = get_secondary_data(NUM_SAMPLES, INDICES, iterations=1)
                X = _transform_spectra(X)
            indices = np.arange(y.shape[0])    
            for ii in range(NN_PROPERTIES['epochs_per_training_set']):
                if i > 0 or ii > 0:
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
            if IS_TEST == True:
                print('Wl2_val/test: ' + str(Wl2_compare[-1]))
                print('Wl2_Train: ' + str(Dict['Wl2_Train'][-1]))
                print('Score val/test: ' + str(Score_compare[-1]))
                print('Score_Train: ' + str(Dict['Score_Train'][-1]))
        if IS_TEST==True:
            state = deepcopy(NN.__getstate__())
            #Below code is only necessary for removing class instances like the random and _optimizer
            for key in list(state.keys()):
                if type(state[key]) not in [str, float, int, tuple, bool, complex, type(None), list, type(np.array(0))]:
                    del  state[key]
                elif type(state[key]) in [list, tuple, type(np.array(0))]:
                    if type(state[key][0]) not in [str, float, int, tuple, bool, complex, type(None), list, type(np.array(0))]:
                        del state[key]
            Dict.update({'NN_PROPERTIES':NN_PROPERTIES, 'parameters':NN.get_params()
            ,'__getstate__': state})
        if IS_TEST == True:
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
        return (X_Test, y_Test)
            
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
    
    def get_NN(dictionary):
        NN = MLPRegressor()
        NN.set_params(**dictionary['parameters'])
        #catches instances where coefficients and intercepts are saved via standard json package as list of lists
        dictionary['__getstate__']['coefs_'] = [np.asarray(coef_list) for coef_list in dictionary['__getstate__']['coefs_'].copy()]
        dictionary['__getstate__']['intercepts_'] = [np.asarray(coef_list) for coef_list in dictionary['__getstate__']['intercepts_'].copy()]
        NN.__setstate__(dictionary['__getstate__'])
        return NN