# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:05:24 2017

@author: lansf
"""
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import json_tricks
from timeit import default_timer as timer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import multiprocessing
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import rcParams
from .neural_network import MLPRegressor
from .jl_spectra_2_structure import IR_GEN
from .jl_spectra_2_structure import get_default_data_paths
from . import error_metrics

class CROSS_VALIDATION:
    """Class for running cross validation. Splits primary data into balanced
       folds and generates sets of secondary data.    
    """
    def __init__(self, ADSORBATE='CO', INCLUDED_BINDING_TYPES=[1,2,3,4]\
                 , cv_indices_path=None, cross_validation_path = None\
                 , nanoparticle_path=None,high_coverage_path=None\
                 , coverage_scaling_path=None,VERBOSE=False, SAVE_ALL_NN=True):
        """ 
        Parameters
        ----------
        ADSORBATE : str
            Adsorbate for which the spectra is to be generated.

        INCLUDED_BINDING_TYPES : list
            Binding types whose frequencies/intensiteis from the primary data
            set will be included in generating the complex spectra.
            
        cv_indices_path : str
            Folder path where indices for cross validation are saved or where
            they are to be created if they have not yet been created. 
            
        cross_validation_path : str
            Folder path where cross validation results are to be created.
            
        nanoparticle_path : str
            File path where nanoparticle or single adsorbate json data is saved.

        high_coverage_path : str
            File path where high coverage data for CO is saved saved.

        coverage_scaling_path : str
            File path where coverage scaling coefficients are saved.
            
        VERBOSE : bool
            Controls the printing of status statements.
        
        Attributes
        ----------
        CV_INDICES_PATH : str
            Folder path for cross validation indices.
            
        CV_PATH : str
            Folder path where cross validation results are to be created.
            
        NANO_PATH : str
            File path where nanoparticle or single adsorbate json data is saved.

        HIGH_COV_PATH : str
            File path where high coverage data for CO is saved saved.

        COV_SCALE_PATH : str
            File path where coverage scaling coefficients are saved.
            
        VERBOSE : bool
            Controls the printing of status statements.
            
        SAVE_ALL_NN : bool
            Controls whether a NN from each cross validation is saved or just
            from the test set.
            
        ADSORBATE : str
            Adsorbate for which the spectra is to be generated.

        INCLUDED_BINDING_TYPES : list
            Binding types whose frequencies/intensiteis from the primary data
            set will be included in generating the complex spectra.
        """
        assert type(ADSORBATE) == str, "ADSORBATE must be a string"
        nano_path, isotope_path, high_cov_path\
           , cov_scale_path = get_default_data_paths(ADSORBATE)
        
        if cross_validation_path is None:
            cross_validation_path = self._get_default_cross_validation_path()
        if cv_indices_path is None:
            cv_indices_path = self._get_default_cv_indices_path()
        if r'/' not in cross_validation_path and r'\\' not in cross_validation_path:
            cross_validation_path = os.path.join(os.getcwd(),cross_validation_path)
        already_exists = os.path.isdir(cross_validation_path)
        if already_exists == False:
            os.mkdir(cross_validation_path)
        self.CV_INDICES_PATH = cv_indices_path
        self.CV_PATH = cross_validation_path
        self.ADSORBATE = ADSORBATE
        self.INCLUDED_BINDING_TYPES = INCLUDED_BINDING_TYPES
        self.NANO_PATH = nanoparticle_path
        self.HIGH_COV_PATH = high_coverage_path
        self.COV_SCALE_PATH = coverage_scaling_path
        self.VERBOSE = VERBOSE
        self.SAVE_ALL_NN = SAVE_ALL_NN
        
        
    def _get_default_cross_validation_path(self):
        """Get cross validation path if not set by user.
        		  
        Returns
        -------
        cross_validation_path : str
            Folder path where cross validation results are to be created.
                
        """
        work_dir = os.getcwd()
        cross_validation_path = os.path.join(work_dir,'cross_validation')
        already_exists = os.path.isdir(cross_validation_path)
        if already_exists == False:
            os.mkdir(cross_validation_path)
        return cross_validation_path
     
    def _get_default_cv_indices_path(self):
        """Get cross validation indices path if not set by user.
        		  
        Returns
        -------
        cv_indices_path : str
            Folder path where cross validation indices are to be created.
                
        """
        work_dir = os.getcwd()
        cv_indices_path = os.path.join(work_dir,'cv_indices')
        already_exists = os.path.isdir(cv_indices_path)
        if already_exists == False:
            os.mkdir(cv_indices_path)
        return cv_indices_path
     
    def _get_state(self):
        """Get important state variables of the class
        		  
        Returns
        -------
        Dict : dict
            State variables necessary to recreate the cross validation run.
                
        """
        Dict = { 'ADSORBATE': self.ADSORBATE,'INCLUDED_BINDING_TYPES': self.INCLUDED_BINDING_TYPES\
                , 'NUM_GCN_LABELS': self.NUM_GCN_LABELS, 'MIN_GCN_PER_LABEL': self.MIN_GCN_PER_LABEL
                , 'GCN_ALL': self.GCN_ALL, 'CV_SPLITS': self.CV_SPLITS, 'BINDING_TYPE_FOR_GCN': self.BINDING_TYPE_FOR_GCN\
                , 'INDICES_VAL': self.INDICES_VAL, 'INDICES_TRAIN': self.INDICES_TRAIN\
                , 'INDICES_TEST': self.INDICES_TEST, 'INDICES_CV_ALL': self.INDICES_CV_ALL\
                , 'TARGET': self.TARGET, 'COVERAGE': self.COVERAGE, 'MAX_COVERAGES': self.MAX_COVERAGES\
                , 'NN_PROPERTIES': self.NN_PROPERTIES, 'TRAINING_ERROR': self.TRAINING_ERROR\
                , 'NUM_TRAIN': self.NUM_TRAIN, 'NUM_VAL': self.NUM_VAL, 'NUM_TEST': self.NUM_TEST\
                , 'LOW_FREQUENCY': self.LOW_FREQUENCY, 'HIGH_FREQUENCY': self.HIGH_FREQUENCY\
                , 'ENERGY_POINTS': self.ENERGY_POINTS}
        return Dict
        
    def generate_test_cv_indices(self, CV_SPLITS=3, BINDING_TYPE_FOR_GCN=[1]\
                                 ,test_fraction=0.25,random_state=0, read_file=False, write_file=False):
        """Function to generate the test and cross validation indices that
        will be used to split the primary data.

        Parameters
        ----------
        CV_SPLITS : int
        	The number of cross validation splits
            
        BINDING_TYPE_FOR_GCN : list or 'ALL'
        	The binding types to consider when tabulating the GCN values. 
            Primary data with other binding types is considered noise. If
            'ALL' is selected all binding types are used and no noise is added
            from other binding types as there are no other binding types.
            
        test_fraction : float
        	The percent of primary data to keep for testing.
            
        random_state : int
        	Can be set to None. Allows reproducibility.
            
        write_file : bool
        	Indicates whether the indices of the selected primary data for cross
            validation is written to a json file.
            
        read_file : bool
        	Indicates whether indices for selecting primary data during cross
            validation and testing is read from a file or directly from this
            function.
        	
        Attributes
        ----------
        CV_SPLITS : int
        	The number of cross validation splits
            
        BINDING_TYPE_FOR_GCN : list or 'ALL'
        	The binding types to consider when tabulating the GCN values. 
            Primary data with other binding types is considered noise. If
            'ALL' is selected all binding types are used and no noise is added
            from other binding types as there are no other binding types.
            
        INDICES_DICTIONARY : dict
        	Dictionary of indices used in cross validation for both the 
            binding type model and the GCN model.
        """
        ADSORBATE = self.ADSORBATE
        INCLUDED_BINDING_TYPES = self.INCLUDED_BINDING_TYPES
        NANO_PATH = self.NANO_PATH
        HIGH_COV_PATH = self.HIGH_COV_PATH
        COV_SCALE_PATH = self.COV_SCALE_PATH
        CV_INDICES_PATH = self.CV_INDICES_PATH
        VERBOSE = self.VERBOSE
        INDICES_FILE = os.path.join(CV_INDICES_PATH,\
        'cross_validation_indices_'+str(CV_SPLITS)+'fold_'+ADSORBATE+'.json')
        if read_file == False:
            INDICES_DICTIONARY = {'BINDING_TYPE':{'train_indices':[], 'val_indices':[]\
                             ,'CV_indices':[],'TEST_indices':[]}\
                            ,'GCN':{'train_indices':[], 'val_indices':[]\
                             ,'CV_indices':[],'TEST_indices':[]}\
                            ,'OTHER_BINDING_TYPE':{'train_indices':[], 'val_indices':[]\
                             ,'CV_indices':[],'TEST_indices':[]}}
            #recursive function that runs GCNconv until n_samples >= n_clusters
            def get_gcn_conv(NUM_GCN_LABELS, BINDING_TYPE_FOR_GCN):
                GCNconv = IR_GEN(ADSORBATE, INCLUDED_BINDING_TYPES=INCLUDED_BINDING_TYPES\
                                 , TARGET='GCN', NUM_TARGETS=NUM_GCN_LABELS\
                             ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                             , coverage_scaling_path=COV_SCALE_PATH, VERBOSE=VERBOSE)
                try:
                    GCNconv.set_GCNlabels(2*(CV_SPLITS+1), BINDING_TYPE_FOR_GCN,showfigures=False)
                except:
                    print('Attempting k-means clustering with fewer labels')
                    if NUM_GCN_LABELS >1:
                        GCNconv = get_gcn_conv(NUM_GCN_LABELS-1, BINDING_TYPE_FOR_GCN)
                    else:
                        GCNconv.set_GCNlabels(0, BINDING_TYPE_FOR_GCN, showfigures=False)
                return GCNconv
            
            GCNconv = get_gcn_conv(11, INCLUDED_BINDING_TYPES)
            BINDING_TYPES = GCNconv.BINDING_TYPES
            if BINDING_TYPE_FOR_GCN == 'ALL':
                BINDING_TYPE_FOR_GCN = list(set(BINDING_TYPES))
            GCNlabels = np.zeros_like(GCNconv.GCNlabels)
            for i in INCLUDED_BINDING_TYPES:
                MAX_GCN_LABELS = int(BINDING_TYPES[BINDING_TYPES == i].size/(2*(CV_SPLITS+1)))
                NUM_GCN_LABELS = min(MAX_GCN_LABELS,11)
                if NUM_GCN_LABELS > 1:
                    GCNconv = get_gcn_conv(NUM_GCN_LABELS,BINDING_TYPE_FOR_GCN=[i])
                    GCNlabels += GCNconv.GCNlabels
            #Get List of indices so we can split the data later
            #combined class to stratify data based on binding-type and GCN simultaneously
            combined_class = 1000*BINDING_TYPES+GCNlabels
            classes_with_counts = np.unique(combined_class,return_counts=True)
            reduce_count=0
            num_classes = classes_with_counts[0].size
            #Below for loop is not necessary if GCNlabels are generated separately
            #for each binding-type as it is now done.
            #ensure that each class as at least 2 members
            for count in range(num_classes):
                if classes_with_counts[1][count-reduce_count] < 2:
                    if VERBOSE == True:
                        print('combining classes to meet kfold constraints')
                    if count < num_classes-1:
                        combined_class[combined_class == classes_with_counts[0][count-reduce_count]] += 1
                    else:
                        combined_class[combined_class == classes_with_counts[0][count-reduce_count]] -= 1
                    classes_with_counts = np.unique(combined_class,return_counts=True)    
                    reduce_count+=1
            if VERBOSE == True:
                print('The class and number in each class for fold generation is '+str(classes_with_counts))
            #split data into cross validation and test set
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
            for CV_index, test_index in sss.split(combined_class,combined_class):
                CV_indices = CV_index
                TEST_indices = test_index
            INDICES_DICTIONARY['BINDING_TYPE'].update({'CV_indices':CV_indices.astype('int').tolist()})
            INDICES_DICTIONARY['BINDING_TYPE'].update({'TEST_indices':TEST_indices.astype('int').tolist()})
            INDICES_DICTIONARY['GCN'].update({'CV_indices':CV_indices[np.isin(BINDING_TYPES[CV_indices], BINDING_TYPE_FOR_GCN)].astype('int').tolist()})
            INDICES_DICTIONARY['GCN'].update({'TEST_indices':TEST_indices[np.isin(BINDING_TYPES[TEST_indices], BINDING_TYPE_FOR_GCN)].astype('int').tolist()})
            INDICES_DICTIONARY['OTHER_BINDING_TYPE'].update({'CV_indices':CV_indices[np.isin(BINDING_TYPES[CV_indices], BINDING_TYPE_FOR_GCN, invert=True)].astype('int').tolist()})
            INDICES_DICTIONARY['OTHER_BINDING_TYPE'].update({'TEST_indices':TEST_indices[np.isin(BINDING_TYPES[TEST_indices], BINDING_TYPE_FOR_GCN, invert=True)].astype('int').tolist()})
            #split data into training and validation sets
            skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle = True, random_state=random_state)
            for train_index, val_index in skf.split(combined_class[CV_indices], combined_class[CV_indices]):
                INDICES_DICTIONARY['BINDING_TYPE']['train_indices'].append(CV_indices[train_index].astype('int').tolist())
                INDICES_DICTIONARY['BINDING_TYPE']['val_indices'].append(CV_indices[val_index].astype('int').tolist())
                INDICES_DICTIONARY['GCN']['train_indices'].append(CV_indices[train_index[np.isin(BINDING_TYPES[CV_indices[train_index]], BINDING_TYPE_FOR_GCN)]].astype('int').tolist())
                INDICES_DICTIONARY['GCN']['val_indices'].append(CV_indices[val_index[np.isin(BINDING_TYPES[CV_indices[val_index]], BINDING_TYPE_FOR_GCN)]].astype('int').tolist())
                INDICES_DICTIONARY['OTHER_BINDING_TYPE']['train_indices'].append(CV_indices[train_index[np.isin(BINDING_TYPES[CV_indices[train_index]], BINDING_TYPE_FOR_GCN, invert=True)]].astype('int').tolist())
                INDICES_DICTIONARY['OTHER_BINDING_TYPE']['val_indices'].append(CV_indices[val_index[np.isin(BINDING_TYPES[CV_indices[val_index]], BINDING_TYPE_FOR_GCN, invert=True)]].astype('int').tolist())    
            if write_file==True:
                with open(INDICES_FILE, 'w') as outfile:
                    json_tricks.dump(INDICES_DICTIONARY, outfile, sort_keys=True, indent=4)
                    if VERBOSE == True:
                        print('Generated CV indices and saved dictionary to file ' + INDICES_FILE)
        elif read_file == True:
            with open(INDICES_FILE, 'r') as infile:
                INDICES_DICTIONARY = json_tricks.load(infile)
        self.INDICES_DICTIONARY = INDICES_DICTIONARY
        self.CV_SPLITS = CV_SPLITS
        self.BINDING_TYPE_FOR_GCN = BINDING_TYPE_FOR_GCN
    
    def set_model_parameters(self, TARGET, COVERAGE, MAX_COVERAGES, NN_PROPERTIES, NUM_TRAIN, NUM_VAL, NUM_TEST\
                             , MIN_GCN_PER_LABEL=0, NUM_GCN_LABELS=11, GCN_ALL = False, TRAINING_ERROR = None\
                             ,LOW_FREQUENCY=200, HIGH_FREQUENCY=2200, ENERGY_POINTS=501):
        """Sets model parameters both for generating the secondary data
        (complex spectra) and for running the neural network.

        Parameters
        ----------
        TARGET : str
            Geometric descriptor for which the target histogram is to be gnerated.
            Can be binding_type, GCN, or combine_hollow_sites. If it is
            combine_hollow_sites then 3-fold and 4-fold sites are grouped together.
            
        COVERAGE : str or float
            The coverage at which the synthetic spectra is generated. If high,
            spectra at various coverages is generated.
            
        MAX_COVERAGES : list
            Maximum coverages allowed for each binding-type if COVERAGE
            is set to 'high'
            
        NN_PROPERTIES : dict
        	Dictionary of properties for the neural network.
            
        NUM_TRAIN : int
        	Number of secondary data (complex spectra) in a single training set.
            
        NUM_VAL : int
        	Number of secondary data (complex spectra) in each validation set.
            
        NUM_TEST : int
        	Number of secondary data (complex spectra) in the test set.
            
        MIN_GCN_PER_LABEL : int
        	Minimum number of primary data points in each GCN group
            
        NUM_GCN_LABELS : int
        	The target number of GCN labels which to group the primary
            data points (simple spectra).
            
        GCN_ALL : bool
        	Whether or not to include all binding types in tabulating GCN values.
            If False, then only the selected binding types determined during the
            creation of the cross validation indices will be used.
            
        TRAINING_ERROR : str float or None
            Indicates the type of error induced in the training set. Can be
            'gaussian', a float, or None. If 'gaussian' then gaussian error
            is added to the scaling factor. If a float then uniform error is
            added after scaling of the primary DFT data.
            
        LOW_FREQUENCY : float
            The lowest frequency for which synthetic spectra is generated
            
        HIGH_FREQUENCY : float
            The high frequency for which synthetic spectra is generated
            
        ENERGY_POINTS : int
            The number of points the synthetic spectra is discretized into
        	
        Attributes
        ----------
        TARGET : str
            Geometric descriptor for which the target histogram is to be gnerated.
            Can be binding_type, GCN, or combine_hollow_sites. If it is
            combine_hollow_sites then 3-fold and 4-fold sites are grouped together.
            
        COVERAGE : str or float
            The coverage at which the synthetic spectra is generated. If high,
            spectra at various coverages is generated.
            
        MAX_COVERAGES : list
            Maximum coverages allowed for each binding-type if COVERAGE
            is set to 'high'
            
        NN_PROPERTIES : dict
        	Dictionary of properties for the neural network.
            
        NUM_TRAIN : int
        	Number of secondary data (complex spectra) in a single training set.
            
        NUM_VAL : int
        	Number of secondary data (complex spectra) in each validation set.
            
        NUM_TEST : int
        	Number of secondary data (complex spectra) in the test set.
            
        MIN_GCN_PER_LABEL : int
        	Minimum number of primary data points in each GCN group
            
        NUM_GCN_LABELS : int
        	The target number of GCN labels which to group the primary
            data points (simple spectra).
            
        GCN_ALL : bool
        	Whether or not to include all binding types in tabulating GCN values.
            If False, then only the selected binding types determined during the
            creation of the cross validation indices will be used.
            
        TRAINING_ERROR : str float or None
            Indicates the type of error induced in the training set. Can be
            'gaussian', a float, or None. If 'gaussian' then gaussian error
            is added to the scaling factor. If a float then uniform error is
            added after scaling of the primary DFT data.
            
        LOW_FREQUENCY : float
            The lowest frequency for which synthetic spectra is generated
            
        HIGH_FREQUENCY : float
            The high frequency for which synthetic spectra is generated
            
        ENERGY_POINTS : int
            The number of points the synthetic spectra is discretized into
        """
        try:
            INDICES_DICTIONARY = self.INDICES_DICTIONARY
        except:
            print("generate_test_cv_indices must be run first")
            raise
        assert type(COVERAGE) == float or COVERAGE==1 or COVERAGE \
        in ['low', 'high'], "Coverage should be a float, 'low', or 'high'."
        assert TARGET in ['combine_hollow_sites','binding_type','GCN'], "incorrect TARGET variable given"
        _set_ir_gen_class = self._set_ir_gen_class
        CV_SPLITS = self.CV_SPLITS

        
        if TARGET == 'GCN' and GCN_ALL == False:
            INDICES_VAL = [(INDICES_DICTIONARY['GCN']['val_indices'][CV_VAL]\
                           , INDICES_DICTIONARY['OTHER_BINDING_TYPE']['val_indices'][CV_VAL])\
                           for CV_VAL in range(CV_SPLITS)]
            INDICES_TRAIN = [(INDICES_DICTIONARY['GCN']['train_indices'][CV_VAL]\
                             , INDICES_DICTIONARY['OTHER_BINDING_TYPE']['train_indices'][CV_VAL])\
                             for CV_VAL in range(CV_SPLITS)]
            INDICES_TEST = [INDICES_DICTIONARY['GCN']['TEST_indices']\
                            , INDICES_DICTIONARY['OTHER_BINDING_TYPE']['TEST_indices']]
            INDICES_CV_ALL = [INDICES_DICTIONARY['GCN']['CV_indices']\
                              , INDICES_DICTIONARY['OTHER_BINDING_TYPE']['CV_indices']]
        else:
            INDICES_VAL = INDICES_DICTIONARY['BINDING_TYPE']['val_indices']
            INDICES_TRAIN = INDICES_DICTIONARY['BINDING_TYPE']['train_indices']
            INDICES_TEST = INDICES_DICTIONARY['BINDING_TYPE']['TEST_indices']
            INDICES_CV_ALL = INDICES_DICTIONARY['BINDING_TYPE']['CV_indices']
            
        self.NUM_GCN_LABELS = NUM_GCN_LABELS
        self.MIN_GCN_PER_LABEL = MIN_GCN_PER_LABEL
        self.GCN_ALL = GCN_ALL
        self.INDICES_VAL = INDICES_VAL
        self.INDICES_TRAIN = INDICES_TRAIN
        self.INDICES_TEST = INDICES_TEST
        self.INDICES_CV_ALL = INDICES_CV_ALL
        self.TARGET = TARGET
        self.COVERAGE = COVERAGE
        self.MAX_COVERAGES = MAX_COVERAGES
        self.NN_PROPERTIES = NN_PROPERTIES
        self.NUM_TRAIN = NUM_TRAIN
        self.NUM_VAL = NUM_VAL
        self.NUM_TEST = NUM_TEST
        self.LOW_FREQUENCY = LOW_FREQUENCY
        self.HIGH_FREQUENCY = HIGH_FREQUENCY
        self.ENERGY_POINTS = ENERGY_POINTS
        self.TRAINING_ERROR = TRAINING_ERROR
        _set_ir_gen_class()
        
    def _set_ir_gen_class(self):
        """Instantiates the class for generaitng complex spectra with
        parameters set by set_model_parameters()
        	
        Attributes
        ----------
        MAINconv : IR_GEN
        	An IR_GEN class that will generate spectra and histograms of
            of binding types or GCN groups
            
        OTHER_SITESconv : IR_GEN
        	An IR_GEN class that will generate spectra for binding types not
            considered by MAINconv. Only instantiated if TARGET == 'GCN' and
            GCN_ALL == False
            
        """
        ADSORBATE = self.ADSORBATE
        NANO_PATH = self.NANO_PATH
        HIGH_COV_PATH = self.HIGH_COV_PATH
        COV_SCALE_PATH = self.COV_SCALE_PATH
        INCLUDED_BINDING_TYPES = self.INCLUDED_BINDING_TYPES
        BINDING_TYPE_FOR_GCN = self.BINDING_TYPE_FOR_GCN
        VERBOSE = self.VERBOSE
        TARGET = self.TARGET
        NUM_GCN_LABELS = self.NUM_GCN_LABELS
        MIN_GCN_PER_LABEL = self.MIN_GCN_PER_LABEL
        GCN_ALL = self.GCN_ALL
        LOW_FREQUENCY = self.LOW_FREQUENCY
        HIGH_FREQUENCY = self.HIGH_FREQUENCY
        ENERGY_POINTS = self.ENERGY_POINTS
        COVERAGE = self.COVERAGE
        MAX_COVERAGES = self.MAX_COVERAGES
        if TARGET == 'combine_hollow_sites': 
            MAINconv = IR_GEN(ADSORBATE, INCLUDED_BINDING_TYPES, 'combine_hollow_sites'\
                         ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                         , coverage_scaling_path=COV_SCALE_PATH, VERBOSE=VERBOSE)
        elif TARGET == 'binding_type':
            MAINconv = IR_GEN(ADSORBATE, INCLUDED_BINDING_TYPES, 'binding_type'\
                         ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                         , coverage_scaling_path=COV_SCALE_PATH, VERBOSE=VERBOSE)
        elif TARGET == 'GCN':
            if GCN_ALL == True:
                MAINconv = IR_GEN(ADSORBATE, INCLUDED_BINDING_TYPES, 'GCN', NUM_TARGETS=NUM_GCN_LABELS\
                         ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                         , coverage_scaling_path=COV_SCALE_PATH, VERBOSE=VERBOSE)
            else:
                MAINconv = IR_GEN(ADSORBATE, BINDING_TYPE_FOR_GCN, 'GCN', NUM_TARGETS=NUM_GCN_LABELS\
                         ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                         , coverage_scaling_path=COV_SCALE_PATH, VERBOSE=VERBOSE)
                OTHER_BINDING_TYPES = np.array(INCLUDED_BINDING_TYPES)[np.isin(INCLUDED_BINDING_TYPES,BINDING_TYPE_FOR_GCN,invert=True)]
                if VERBOSE == True:
                    print('OTHER_BINDING_TYPES: ' + str(OTHER_BINDING_TYPES))
                OTHER_SITESconv = IR_GEN(ADSORBATE, OTHER_BINDING_TYPES, 'binding_type'\
                         ,nanoparticle_path=NANO_PATH, high_coverage_path=HIGH_COV_PATH\
                         , coverage_scaling_path=COV_SCALE_PATH, VERBOSE=VERBOSE)
                OTHER_SITESconv.set_spectra_properties('low', [1,1,1,1]\
                , LOW_FREQUENCY, HIGH_FREQUENCY, ENERGY_POINTS)
            MAINconv.set_GCNlabels(MIN_GCN_PER_LABEL, BINDING_TYPE_FOR_GCN, showfigures=False)
        MAINconv.set_spectra_properties(COVERAGE, MAX_COVERAGES\
        , LOW_FREQUENCY, HIGH_FREQUENCY, ENERGY_POINTS)
        if TARGET == 'GCN' and GCN_ALL == False:
            self.OTHER_SITESconv = OTHER_SITESconv
        self.MAINconv = MAINconv
    
    def set_nn_parameters(self, NN_PROPERTIES):
        """Set properties of the neural network so they can be changed after
        all other model properties have been set
        	
        Attributes
        ----------
        NN_PROPERTIES : dict
        	Dictionary of properties for the neural network.
        """
        self.NN_PROPERTIES = NN_PROPERTIES
    
    def run_CV(self, write_file=False, CV_RESULTS_FILE = None):
        """run cross validation on one cross validation set at a time

        Parameters
        ----------
        write_file : bool
        	Indicate whether cross validation results will be written out to
            a file.
            
        CV_RESULTS_FILE : str or None
        	File where cross validation results will be written as a json.
        	
        Attributes
        ----------
        CV_RESULTS_FILE : str or None
        	File where cross validation results will be written as a json.
        		  
        Returns
        -------
        DictList : list of dict
        	Dictionary of cross validation and test results
                
        """
        try:
            TARGET = self.TARGET
        except:
            print("set_model_parameters must be run first")
            raise
        _get_state = self._get_state
        _run_NN = self._run_NN
        get_secondary_data = self.get_secondary_data
        CV_SPLITS = self.CV_SPLITS
        ADSORBATE = self.ADSORBATE
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
        VERBOSE = self.VERBOSE
        
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
        elif r'/' not in CV_RESULTS_FILE and r'\\' not in CV_RESULTS_FILE:
            if '.json' in CV_RESULTS_FILE:
                CV_RESULTS_FILE = os.path.join(CV_PATH,CV_RESULTS_FILE)
            else:
                CV_RESULTS_FILE = os.path.join(CV_PATH,CV_RESULTS_FILE+'.json')
            
        DictList = [{} for _ in range(CV_SPLITS)]
        #Cross Validation
        start = timer()
        for CV_INDEX in range(CV_SPLITS):
            if VERBOSE == True:
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
        if VERBOSE == True:
            print('#########################################################')
            print('#########################################################')
            print('Time to run the CV+Test: ' + str(stop-start))
        Dict = _get_state()
        DictList.append(Dict)
        if write_file == True:
            with open(CV_RESULTS_FILE, 'w') as outfile:
                json_tricks.dump(DictList, outfile, sort_keys=True, indent=4)
        self.CV_RESULTS_FILE = CV_RESULTS_FILE
        return DictList
    
    def run_CV_multiprocess(self,write_file=False, CV_RESULTS_FILE = None, num_procs=None):
        """run multiple cross valdation sets and the test set simultaneously

        Parameters
        ----------
        write_file : bool
        	Indicate whether cross validation results will be written out to
            a file.
            
        CV_RESULTS_FILE : str or None
        	File where cross validation results will be written as a json.
            
        num_procs : int
        	Number of processes to run at any given time.
        	
        Attributes
        ----------
        CV_RESULTS_FILE : str or None
        	File where cross validation results will be written as a json.
        		  
        Returns
        -------
        DictList : list of dict
        	Dictionary of cross validation and test results
                
        """
        try:
            TARGET = self.TARGET
        except:
            print("set_model_parameters must be run first")
            raise
        _get_state = self._get_state
        run_single_CV = self.run_single_CV
        CV_SPLITS = self.CV_SPLITS
        ADSORBATE = self.ADSORBATE
        CV_PATH = self.CV_PATH
        COVERAGE = self.COVERAGE
        NN_PROPERTIES = self.NN_PROPERTIES
        VERBOSE = self.VERBOSE
        if CV_RESULTS_FILE is None:
            CV_RESULTS_FILE = os.path.join(CV_PATH,'CV_results_'+TARGET+'_'+str(COVERAGE)\
            +'_'+str(CV_SPLITS)+'fold'+'_reg'+'{:.2E}'.format(NN_PROPERTIES['alpha'])\
            +'_'+NN_PROPERTIES['loss']+'_'+ADSORBATE+'.json')
        elif r'/' not in CV_RESULTS_FILE and r'\\' not in CV_RESULTS_FILE:
            if '.json' in CV_RESULTS_FILE:
                CV_RESULTS_FILE = os.path.join(CV_PATH,CV_RESULTS_FILE)
            else:
                CV_RESULTS_FILE = os.path.join(CV_PATH,CV_RESULTS_FILE+'.json')
            
        CV_INDEX_or_TEST = [i for i in range(CV_SPLITS)]
        CV_INDEX_or_TEST.append('TEST')
        #Cross Validation and Test Run
        start = timer()
        #memory requirment since spectra are arrays of 501 real numbers of float64
        #memory_requriement_per_run = 8*501*(NUM_TRAIN+(max(NUM_VAL,NUM_TEST)))
        #memory_available = psutil.virtual_memory()[0]
        #max_runs_with_memory = int((0.95*memory_available-1)/memory_requriement_per_run)
        cpu_cores = multiprocessing.cpu_count()
        num_procs_given = num_procs
        if num_procs is None:
            num_procs=cpu_cores
        num_procs = min(cpu_cores,num_procs,CV_SPLITS+1)

        if num_procs_given is not None and VERBOSE == True:
            print('#########################################################')
            print('#########################################################')
            if num_procs < num_procs_given:
                if num_procs == CV_SPLITS+1:
                    print('Resetting number of processes to '+str(num_procs)+', which is the necessary number of model calls.')
                elif num_procs == cpu_cores:
                    print('Resetting number of processes to '+str(num_procs)+' which is the number of available cores.')
        else:
            if num_procs == CV_SPLITS+1:
                print('Setting number of processes to '+str(num_procs)+', which is the necessary number of model calls.')
            elif num_procs == cpu_cores:
                print('Setting number of processes to '+str(num_procs)+' which is the number of available cores.')          
            
        with multiprocessing.Pool(processes=num_procs) as pool:
            DictList = pool.imap(run_single_CV,CV_INDEX_or_TEST)
            DictList = [Dict for Dict in DictList]
        stop = timer()
        if VERBOSE == True:
            print('#########################################################')
            print('#########################################################')
            print('Time to run the CV+Test: ' + str(stop-start))
        Dict = _get_state()
        DictList.append(Dict)
        if write_file == True:
            with open(CV_RESULTS_FILE, 'w') as outfile:
                json_tricks.dump(DictList, outfile, sort_keys=True, indent=4)
        self.CV_RESULTS_FILE = CV_RESULTS_FILE
        return DictList
    
    def run_single_CV(self, CV_INDEX_or_TEST):
        """run a single cross validation process or the test set.

        Parameters
        ----------
        CV_INDEX_or_TEST : str or int
        	Indicates which cross validation set or if the test set is to be
            run.
        		  
        Returns
        -------
        Dict : dict
        	Dictionary of a single cross validation or test result
               
        """
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
        VERBOSE = self.VERBOSE
        iterations = 10
        if VERBOSE == True:
            print(str(multiprocessing.cpu_count())+ ' cores available to this instance.')
        if CV_INDEX_or_TEST != 'TEST':
            if VERBOSE == True:
                print('#########################################################')
                print('#########################################################')
                print('The CV number is '+str(CV_INDEX_or_TEST+1))
            start = timer()
            X_compare, y_compare = get_secondary_data(NUM_SAMPLES=NUM_VAL\
                                , INDICES=INDICES_VAL[CV_INDEX_or_TEST],iterations=iterations)
            stop = timer()
            if VERBOSE == True:
                print('Time to generate one batch of secondary data is ' + str(stop-start))
            
            Dict =  _run_NN(NUM_TRAIN, INDICES_TRAIN[CV_INDEX_or_TEST], X_compare, y_compare, IS_TEST=False) 
        else:
            if VERBOSE == True:
                print('#########################################################')
                print('#########################################################')
                print('Training on whole CV population and Testing on test set')
                
            X_compare, y_compare = get_secondary_data(NUM_TEST, INDICES_TEST\
                                    ,iterations=iterations)
            
            Dict =_run_NN(NUM_TRAIN, INDICES_CV_ALL, X_compare, y_compare, IS_TEST=True)
        return Dict
    
    def get_secondary_data(self,NUM_SAMPLES, INDICES, iterations=1, IS_TRAINING_SET=False):
        """Get secondary data (complex spectra)

        Parameters
        ----------
        NUM_SAMPLES : int
        	Number of complex spectra to generate
            
        INDICES : list of int
        	The indices of the primary data that will be selected.
            
        iterations : int
        	Number of times secondary data will be generated and strung
            together. Allows for a more diverse set of complex spectra for
            testing.
            
        IS_TRAINING_SET : bool
        	Indicates if the secondary set will be used for training or testing
        	
        Returns
        -------
        X : numpy.ndarray
        	Coverage shifted frequencies and intensities
            
        Y : The target variable histograms. Either binding-type or GCN label
                
        """
        try:
            TARGET = self.TARGET
        except:
            print("set_model_parameters must be run first")
            raise
        GCN_ALL = self.GCN_ALL
        MAINconv = deepcopy(self.MAINconv)
        TRAINING_ERROR = self.TRAINING_ERROR
        num_samples_original = NUM_SAMPLES
        NUM_SAMPLES = int(NUM_SAMPLES/iterations)
        ADDITIONAL_POINT = int(num_samples_original-NUM_SAMPLES*iterations)
        if TARGET == 'GCN' and GCN_ALL == False:
            OTHER_SITESconv = deepcopy(self.OTHER_SITESconv)
            X1, y = MAINconv.get_synthetic_spectra(NUM_SAMPLES+ADDITIONAL_POINT, INDICES[0], IS_TRAINING_SET, TRAINING_ERROR=TRAINING_ERROR)
            X2, y2 = OTHER_SITESconv.get_synthetic_spectra(int(NUM_SAMPLES/5), INDICES[1], IS_TRAINING_SET, TRAINING_ERROR=TRAINING_ERROR)
            X = MAINconv.add_noise(X1,X2)
            del X1; del X2; del y2
        else:
            X, y = MAINconv.get_synthetic_spectra(NUM_SAMPLES+ADDITIONAL_POINT, INDICES, IS_TRAINING_SET, TRAINING_ERROR=TRAINING_ERROR)
        #Add to the validation and test sets to get more coverage options
        #(each iteration has 10 different coverage combinations
        #when TARGET in ['binding_type','combine_hollow_sites'] and COVERAGE is not 'low')
        for _ in range(iterations-1):
            if TARGET == 'GCN' and GCN_ALL == False:
                X1_2, y_2 = MAINconv.get_synthetic_spectra(NUM_SAMPLES, INDICES[0], IS_TRAINING_SET, TRAINING_ERROR=TRAINING_ERROR)
                X2_2, y2_2 = OTHER_SITESconv.get_synthetic_spectra(int(NUM_SAMPLES/5), INDICES[1], IS_TRAINING_SET, TRAINING_ERROR=TRAINING_ERROR)
                X_2 = MAINconv.add_noise(X1_2, X2_2)
                del X1_2; del X2_2; del y2_2
            else:
                X_2, y_2 = MAINconv.get_synthetic_spectra(NUM_SAMPLES, INDICES, IS_TRAINING_SET, TRAINING_ERROR=TRAINING_ERROR)
            X = np.append(X,X_2,axis=0)
            y = np.append(y,y_2,axis=0)
            del X_2; del y_2
        return (X, y)

    def _run_NN(self, NUM_SAMPLES, INDICES, X_compare, y_compare, IS_TEST):
        """run the neural network and generated validation statistics

        Parameters
        ----------
        NUM_SAMPLES : int
        	Number of complex spectra to generate
            
        INDICES : list of int
        	The indices of the primary data that will be selected.
            
        X_compare : numpy.ndarray
        	Complex spectra on which to test the model.
            
        Y_compare : numpy.ndarray
            The target variable histogramsof the test or validation data.
            
        IS_TEST : bool
        	Indicates whether comparison set is the test set.
        	
        Attributes
        ----------
        NN : neural_network.MLPRegressor
        	Trained neural network. Only instantiated if IS_TEST == True
        		  
        Returns
        -------
        Dict : dict
        	Dictionary of validation or test results and statistics
                
        """
        get_secondary_data = self.get_secondary_data
        NN_PROPERTIES = self.NN_PROPERTIES
        VERBOSE = self.VERBOSE
        SAVE_ALL_NN = self.SAVE_ALL_NN
        if IS_TEST == False:
            Dict = {'Wl2_Train':[], 'Score_Train':[]\
                ,'Wl2_Val':[], 'Score_Val':[]}
            Score_compare = Dict['Score_Val']
            Wl2_compare = Dict['Wl2_Val']
            if SAVE_ALL_NN == True:
                WL2_score_best = 10**6
                Dict = {'NN_PROPERTIES':[], 'Wl2_Train':[], 'Score_Train':[]\
                ,'Wl2_Val':[], 'Score_Val':[]
                ,'parameters': [],'__getstate__':[]}
                Score_compare = Dict['Score_Val']
                Wl2_compare = Dict['Wl2_Val']
        else:
            WL2_score_best = 10**6
            Dict = {'NN_PROPERTIES':[]
            ,'Wl2_Train':[], 'Score_Train':[]
            ,'Wl2_Test':[], 'Score_Test': []
            ,'parameters': [],'__getstate__':[]}
            Score_compare = Dict['Score_Test']
            Wl2_compare = Dict['Wl2_Test']
        
        NN = MLPRegressor(hidden_layer_sizes=NN_PROPERTIES['hidden_layer_sizes'], activation='relu', solver='adam'
                              , tol=10**-9, alpha=NN_PROPERTIES['alpha'], verbose=False, batch_size=NN_PROPERTIES['batch_size']
                              , max_iter=1, epsilon= NN_PROPERTIES['epsilon'], early_stopping=False
                              ,warm_start=True,loss=NN_PROPERTIES['loss'], regularization=NN_PROPERTIES['regularization']
                              ,learning_rate_init=NN_PROPERTIES['learning_rate_init'],out_activation='softmax')
        
        #Using Fit (w/ coverages)
        start = timer()
        X, y = get_secondary_data(NUM_SAMPLES, INDICES, iterations=1, IS_TRAINING_SET=True)
        stop = timer()
        if IS_TEST == True and VERBOSE == True:
                print('Time to generate one batch of secondary data is ' + str(stop-start))
        NN.partial_fit(X, y)
        y_predict = NN.predict(X)
        ycompare_predict = NN.predict(X_compare)
        Dict['Score_Train'].append(error_metrics.get_r2(y,y_predict))
        Dict['Wl2_Train'].append(error_metrics.get_wasserstein_loss(y,y_predict))
        Score_compare.append(error_metrics.get_r2(y_compare,ycompare_predict))
        Wl2_compare.append(error_metrics.get_wasserstein_loss(y_compare,ycompare_predict))
        for i in range(NN_PROPERTIES['training_sets']):
            if IS_TEST == True and VERBOSE == True:
                print('Training set number ' + str(i+1))
            if i > 0:
                X, y = get_secondary_data(NUM_SAMPLES, INDICES, iterations=1, IS_TRAINING_SET=True)
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
                    #This updates the value in the dictionary as well.
                    Score_compare.append(error_metrics.get_r2(y_compare,ycompare_predict))
                    Wl2_compare.append(error_metrics.get_wasserstein_loss(y_compare,ycompare_predict))
                    if IS_TEST == True or SAVE_ALL_NN == True:
                        if WL2_score_best > Wl2_compare[-1]:
                            WL2_score_best = Wl2_compare[-1]
                            NN_BEST = deepcopy(NN)
            if IS_TEST == True and VERBOSE==True:
                print('Wl2_val/test: ' + str(Wl2_compare[-1]))
                print('Wl2_Train: ' + str(Dict['Wl2_Train'][-1]))
                print('Score val/test: ' + str(Score_compare[-1]))
                print('Score_Train: ' + str(Dict['Score_Train'][-1]))
        if IS_TEST == True or SAVE_ALL_NN == True:
            state = deepcopy(NN_BEST.__getstate__())
            #Below code is only necessary for removing class instances like the random and _optimizer
            for key in list(state.keys()):
                if type(state[key]) not in [str, float, int, tuple, bool, complex, type(None), list, type(np.array(0))]:
                    del  state[key]
                elif type(state[key]) in [list, tuple, type(np.array(0))]:
                    if type(state[key][0]) not in [str, float, int, tuple, bool, complex, type(None), list, type(np.array(0))]:
                        del state[key]
            Dict.update({'NN_PROPERTIES':NN_PROPERTIES, 'parameters':NN_BEST.get_params()
            ,'__getstate__': state})
        if SAVE_ALL_NN == True:
            try:
                self.NN_LIST.append(NN_BEST)
            except:
                self.NN_LIST = []
                self.NN_LIST.append(NN_BEST)
        if IS_TEST == True or SAVE_ALL_NN == True:
            self.NN = NN_BEST
        return Dict

    def get_test_secondary_data(self):
        """Get one batch of secondary data (comlex spectra) for the test indices
        		  
        Returns
        -------
        X_Test : numpy.ndarray
        	Complex spectra on which to test the model.
            
        Y_Test : numpy.ndarray
            The target variable histogramsof the test data.
             
        """
        try:
            NUM_TEST = self.NUM_TEST
        except:
            print("set_model_parameters must be run first")
            raise
        INDICES_TEST = self.INDICES_TEST
        #Get Test Spectra
        start = timer()
        X_Test, y_Test = self.get_secondary_data(NUM_TEST, INDICES_TEST, iterations=10)
        stop = timer()
        print('Time to generate Test set is ' + str(stop-start))
        return (X_Test, y_Test)
            
    def get_test_results(self):
        """Returns a dictionary of a trained neural network evaluated on the
        test results.
        	
        Attributes
        ----------
        X_Test : numpy.ndarray
        	Complex spectra on which to test the model.
            
        Y_Test : numpy.ndarray
            The target variable histogramsof the test data.
        		  
        Returns
        -------
        Dict : dict
        	Dictionary of test results and statistics
                
        """
        try:
            NUM_TRAIN = self.NUM_TRAIN
        except:
            print("set_model_parameters must be run first")
            raise
        get_test_secondary_data = self.get_test_secondary_data
        _run_NN = self._run_NN
        INDICES_CV_ALL = self.INDICES_CV_ALL
        X_Test, y_Test = get_test_secondary_data()
        Dict =_run_NN(NUM_TRAIN, INDICES_CV_ALL, X_Test, y_Test, IS_TEST=True)
        self.X_Test = X_Test
        self.Y_Test = y_Test
        return Dict
    
class LOAD_CROSS_VALIDATION(CROSS_VALIDATION):
    """Child class for loading cross validation results.
    """  
    
    def __init__(self, cv_indices_path=None, cross_validation_path=None):
        """
        Parameters
        ----------
        cv_indices_path : str
            Folder path where indices for cross validation are saved. 
            
        cross_validation_path : str
            Folder path where cross validation results are stored.
        	
        Attributes
        ----------
        CV_FILES : list of str
        	LIst of cross validation files.
            
        CV_INDICES_PATH_OLD : str
            Folder path where indices for cross validation are saved. 
            
        CV_PATH_OLD : str
            Folder path where cross validation results are stored.
        		  
        """
        _get_default_cross_validation_path = super()._get_default_cross_validation_path
        _get_default_cv_indices_path = super()._get_default_cv_indices_path
        if cross_validation_path is None:
            cross_validation_path = _get_default_cross_validation_path()
        if cv_indices_path is None:
            cv_indices_path = _get_default_cv_indices_path()
        if r'/' not in cross_validation_path and r'\\' not in cross_validation_path:
            cross_validation_path = os.path.join(os.getcwd(),cross_validation_path)
        already_exists = os.path.isdir(cross_validation_path)
        if already_exists == False:
            os.mkdir(cross_validation_path)
        is_directory = os.path.isdir(cross_validation_path)
        is_directory_indices = os.path.isdir(cv_indices_path)
        assert is_directory_indices == True and is_directory == True\
        , "input to LOAD_CROSS_VALIDATION path is not a directory"
        CV_FILES = [os.path.join(cross_validation_path,file) for file \
        in os.listdir(cross_validation_path) \
        if os.path.isfile(os.path.join(cross_validation_path,file)) == True]
        CV_FOLDERS = [os.path.join(cross_validation_path,folder) for folder \
        in os.listdir(cross_validation_path) \
        if os.path.isdir(os.path.join(cross_validation_path,folder)) == True]
        for folder in CV_FOLDERS:
            CV_FILE = [os.path.join(folder,file) for file \
                        in os.listdir(folder) \
                        if os.path.isfile(os.path.join(folder,file)) == True]
            CV_FILES += CV_FILE
        self.CV_FILES = CV_FILES
        self.CV_PATH_OLD = cross_validation_path
        self.CV_INDICES_PATH_OLD = cv_indices_path
        
    def get_NN(self,dictionary):
        """Loads a neural network from a dictionary containing properties
        such as number of nodes, acitavtion functions, their coefficients and
        their intercepts.

        Parameters
        ----------
        dictionary : dict
        	Dictionary containing neural network results and properties
        		  
        Returns
        -------
        NN : neural_network.MLPRegressor
        	Trained neural network.
                
        """
        NN = MLPRegressor(**dictionary['parameters'])
        #catches instances where coefficients and intercepts are saved via standard json package as list of lists
        dictionary['__getstate__']['coefs_'] = [np.asarray(coef_list) for coef_list in dictionary['__getstate__']['coefs_'].copy()]
        dictionary['__getstate__']['intercepts_'] = [np.asarray(coef_list) for coef_list in dictionary['__getstate__']['intercepts_'].copy()]
        NN.__setstate__(dictionary['__getstate__'])
        return NN
    
    def get_NN_ensemble(self,indices, use_all_cv_NN = False):
        """Loads a neural network from a dictionary containing properties
        such as number of nodes, acitavtion functions, their coefficients and
        their intercepts.

        Parameters
        ----------
        indices : list
        	Indicates which cross validation results to use in generation of a NN
          ensemble
          
         use_all_cv_NN : bool
          Indicates whether to use NN from all cross validation runs in
          addition to the one trained on all the cross validation data.
        		  
        Returns
        -------
        NN_ensemble : neural_network.MLPRegressor ensemble
        	Trained neural network ensemble.
                
        """
        get_NN = self.get_NN
        CV_FILES = self.CV_FILES
        NN_List = []
        for index in indices:
            file = CV_FILES[index]
            with open(file, 'r') as infile:
                CV_DICT_LIST = json_tricks.load(infile)
            if use_all_cv_NN == False:
                NN = get_NN(CV_DICT_LIST[-2])
                NN_List.append(NN)
            else:
                NN_List += [get_NN(i) for i in CV_DICT_LIST[0:-1]]
        return NN_ENSEMBLE(NN_List)
    
    def get_ensemble_cv(self):
        """Get cross validation error for an ensemble of models

        Parameters
        ----------
        	
        Attributes
        ----------
        ENSEMBLE_MODELS : dict
        	Dictionary of ensemble models
        		  
        Returns
        -------
        ENSEMBLE_MODELS_dict : dict
        	Dictionary of ensemble models
                
        """
        try:
            CV_RESULTS = self.CV_RESULTS
        except:
            self.load_all_CV_data()
            CV_RESULTS = self.CV_RESULTS
        ENSEMBLE_MODELS = NESTED_DICT()
        for ADSORBATE in CV_RESULTS.keys():
            for TARGET in CV_RESULTS[ADSORBATE].keys():
                for COVERAGE in CV_RESULTS[ADSORBATE][TARGET].keys():
                    NUM_EPOCHS = CV_RESULTS[ADSORBATE][TARGET][COVERAGE]['WL_TRAIN'][0].shape[1]
                    ENSEMBLE_MODELS[ADSORBATE][TARGET][COVERAGE]['WL_VAL_mean'] =\
                    np.reshape(CV_RESULTS[ADSORBATE][TARGET][COVERAGE]['WL_VAL']\
                               ,(-1,NUM_EPOCHS)).mean(axis=0)
                    ENSEMBLE_MODELS[ADSORBATE][TARGET][COVERAGE]['WL_VAL_std'] =\
                    np.reshape(CV_RESULTS[ADSORBATE][TARGET][COVERAGE]['WL_VAL']\
                               ,(-1,NUM_EPOCHS)).std(axis=0, ddof=1)
                    ENSEMBLE_MODELS[ADSORBATE][TARGET][COVERAGE]['WL_TRAIN_mean'] =\
                    np.reshape(CV_RESULTS[ADSORBATE][TARGET][COVERAGE]['WL_TRAIN']\
                               ,(-1,NUM_EPOCHS)).mean(axis=0)
                    ENSEMBLE_MODELS[ADSORBATE][TARGET][COVERAGE]['WL_TRAIN_std'] =\
                    np.reshape(CV_RESULTS[ADSORBATE][TARGET][COVERAGE]['WL_TRAIN']\
                               ,(-1,NUM_EPOCHS)).std(axis=0, ddof=1)
                    ENSEMBLE_MODELS[ADSORBATE][TARGET][COVERAGE]['WL_TEST_mean'] =\
                    np.mean(CV_RESULTS[ADSORBATE][TARGET][COVERAGE]['WL_TEST_TEST'],\
                                axis=0)
                    ENSEMBLE_MODELS[ADSORBATE][TARGET][COVERAGE]['WL_TEST_std'] =\
                    np.std(CV_RESULTS[ADSORBATE][TARGET][COVERAGE]['WL_TEST_TEST'],\
                                axis=0, ddof=1)
        ENSEMBLE_MODELS_dict = NESTED_DICT_to_DICT(ENSEMBLE_MODELS)
        self.ENSEMBLE_MODELS = ENSEMBLE_MODELS_dict
        return ENSEMBLE_MODELS_dict
    
    def load_CV_class(self,index, new_cv_indices_path = None, new_cross_validation_path=None):
        """Load all stored data from a single cross validation run.

        Parameters
        ----------
        index : int
        	Indicates which cross validation result to load
            
        new_cv_indices_path : str
         Folder path where indices for new cross validation are to be saved. 
            
        new_cross_validation_path : str
         Folder path where new cross validation results are to be saved.
        	
        Attributes
        ----------
        NN : neural_network.MLPRegressor
        	Trained neural network.
            
        CV_DICT_LIST : list of dict
        	List of dictionaries with cross validations results, statistics,
            and the neural network traine on all cross validation data.
                
        """
        __dict__ = self.__dict__
        get_NN = self.get_NN
        CV_FILES = self.CV_FILES
        CV_PATH_OLD = self.CV_PATH_OLD
        CV_INDICES_PATH_OLD = self.CV_INDICES_PATH_OLD
        if new_cross_validation_path is None:
            new_cross_validation_path = CV_PATH_OLD.rstrip('/ \ ') + r'_continuation/'
        if new_cv_indices_path is None:
            new_cv_indices_path = CV_INDICES_PATH_OLD.rstrip('/ \ ') + r'_continuation/'
        if os.path.isdir(new_cross_validation_path) == False:
            os.mkdir(new_cross_validation_path)
        if os.path.isdir(new_cv_indices_path) == False:
            os.mkdir(new_cv_indices_path)
        file = CV_FILES[index]
        with open(file, 'r') as infile:
            CV_DICT_LIST = json_tricks.load(infile)
        __dict__.update(CV_DICT_LIST[-1])
        self.NN = get_NN(CV_DICT_LIST[-2])
        try:
            self.NN_ENSEMBLE = NN_ENSEMBLE([get_NN(i) for i in CV_DICT_LIST[0:-1]])
        except:
            pass
        self.CV_DICT_LIST = CV_DICT_LIST
        super().__init__(ADSORBATE=self.ADSORBATE, INCLUDED_BINDING_TYPES=self.INCLUDED_BINDING_TYPES\
             , cv_indices_path=new_cv_indices_path, cross_validation_path=new_cross_validation_path)
        super()._set_ir_gen_class()
        
    def load_all_CV_data(self):
        """Load results from all cross validation runs in order to identify the
        best neural network for each category (CO, GCN, etc.)
        	
        Attributes
        ----------
        CV_RESULTS : dict
        	Dictionary of all cross validation results
        """
        CV_FILES = self.CV_FILES
        CV_RESULTS = NESTED_DICT()
        def deep_update(ADSORBATE,TARGET,COVERAGE,KEY,VALUE=None,create_list=False):
            if create_list == True:
                CV_RESULTS[ADSORBATE][TARGET][COVERAGE].update({KEY:[]})
            CV_RESULTS[ADSORBATE][TARGET][COVERAGE][KEY].append(VALUE)
        
        KEYS = ['INCLUDED_BINDING_TYPES','NUM_GCN_LABELS','WL_VAL','WL_TRAIN','WL_VAL_mean','WL_VAL_std','WL_TRAIN_mean'\
              ,'WL_TRAIN_std','WL_TEST_TEST','WL_TEST_TRAIN','CV_FILES_INDEX']
        for count, file in enumerate(CV_FILES):
            print('Loading data from model: '+str(count))
            with open(file, 'r') as infile:
                CV_DICT_LIST = json_tricks.load(infile)
            ADSORBATE = CV_DICT_LIST[-1]['ADSORBATE']
            INCLUDED_BINDING_TYPES = CV_DICT_LIST[-1]['INCLUDED_BINDING_TYPES']
            TARGET = CV_DICT_LIST[-1]['TARGET']
            COVERAGE = CV_DICT_LIST[-1]['COVERAGE']
            NUM_GCN_LABELS = CV_DICT_LIST[-1]['NUM_GCN_LABELS']
            WL_VAL = []
            WL_TRAIN= []
            for i in range(CV_DICT_LIST[-1]['CV_SPLITS']):
                WL_VAL.append(CV_DICT_LIST[i]['Wl2_Val'])
                WL_TRAIN.append(CV_DICT_LIST[i]['Wl2_Train'])
            WL_VAL_mean = np.mean(WL_VAL,axis=0)
            WL_VAL_std  = np.std(WL_VAL,axis=0,ddof=1)
            WL_TRAIN_mean = np.mean(WL_TRAIN,axis=0)
            WL_TRAIN_std  = np.std(WL_TRAIN,axis=0,ddof=1)
            WL_TEST_TEST = CV_DICT_LIST[-2]['Wl2_Test']
            WL_TEST_TRAIN = CV_DICT_LIST[-2]['Wl2_Train']
            VALUES = [INCLUDED_BINDING_TYPES,NUM_GCN_LABELS, np.array(WL_VAL), np.array(WL_TRAIN), WL_VAL_mean, WL_VAL_std, WL_TRAIN_mean\
                      , WL_TRAIN_std, WL_TEST_TEST, WL_TEST_TRAIN,count]
            try:
                for KEY, VALUE in zip(KEYS,VALUES):
                    deep_update(ADSORBATE,TARGET,COVERAGE,KEY,VALUE, create_list=False)
            except:
                for KEY, VALUE in zip(KEYS,VALUES):
                    deep_update(ADSORBATE,TARGET,COVERAGE,KEY,VALUE, create_list=True)
            
        CV_RESULTS_dict = NESTED_DICT_to_DICT(CV_RESULTS)
        self.CV_RESULTS = CV_RESULTS_dict
            
    def get_best_models(self, models_per_category, standard_deviations):
        """Identify the best neural networks for each category (CO, GCN, etc.)

        Parameters
        ----------
        models_per_category : int
        	Number of models to include for each 
            
        standard_deviations : float
        	Number of standard deviations to add to the mean cross validation
            loss in order select the best model. Optimizes but bias and
            variance
        	
        Attributes
        ----------
        BEST_MODELS : dict
        	Dictionary of just the best model results and their corresponding
            model index value
        		  
        Returns
        -------
        BEST_MODELS_dict : dict
        	Dictionary of just the best model results and their corresponding
            model index value
                
        """
        try:
            CV_RESULTS = self.CV_RESULTS
        except:
            self.load_all_CV_data()
            CV_RESULTS = self.CV_RESULTS
        BEST_MODELS = NESTED_DICT()
        for ADSORBATE in CV_RESULTS.keys():
            for TARGET in CV_RESULTS[ADSORBATE].keys():
                for COVERAGE in CV_RESULTS[ADSORBATE][TARGET].keys():
                    WL_VAL = np.array(CV_RESULTS[ADSORBATE][TARGET][COVERAGE]['WL_VAL_mean'])
                    WL_STD = np.array(CV_RESULTS[ADSORBATE][TARGET][COVERAGE]['WL_VAL_std'])
                    min_score = np.array([np.min(WL_VAL[cv_run] + standard_deviations * WL_STD[cv_run]) for cv_run in range(WL_VAL.shape[0])])
                    best_models = np.argsort(min_score)[0:models_per_category]
                    SCORES = min_score[best_models]
                    BEST_MODELS[ADSORBATE][TARGET][COVERAGE]['SCORES'] = SCORES
                    for KEY in CV_RESULTS[ADSORBATE][TARGET][COVERAGE].keys():
                        BEST_VALUES = np.array(CV_RESULTS[ADSORBATE][TARGET][COVERAGE][KEY])[best_models]
                        BEST_MODELS[ADSORBATE][TARGET][COVERAGE][KEY] = BEST_VALUES
        BEST_MODELS_dict = NESTED_DICT_to_DICT(BEST_MODELS)
        self.BEST_MODELS = BEST_MODELS_dict
        return BEST_MODELS_dict
    
    def get_keys(self,dictionary):
        """Get all keys in a dictionary of dictionaries for viewing

        Parameters
        ----------
        dictionary : dict
        	A dictionary of dictionaries.
        		  
        Returns
        -------
        dictionary_of_keys : dict
        	A dictionary of keys.
                
        """
        dictionary_of_keys = {}
        def recursive_items(dictionary,dictionary_of_keys):
            for key, value in dictionary.items():
                if type(value) is dict:
                    next_key = list(value.keys())[0]
                    next_value = value[next_key]
                    value2 = value.copy()
                    if type(next_value) is dict:
                        dictionary_of_keys.update({key:value2})
                        recursive_items(value,value2)
                    else:
                        dictionary_of_keys.update({key:list(value2.keys())})
        
        recursive_items(dictionary,dictionary_of_keys)
        return dictionary_of_keys
    
    def plot_models(self,dictionary,figure_directory='show',model_list = None\
                    ,xlim=[0, 200], ylim1=[0, 0.3], ylim2=[0, 0.3]):
        """plot model learning curves for all models in a given dictionary

        Parameters
        ----------
        dictionary : dict
        	A dictionary of dictionaries.
            
        figure_directory : str
        	Folder location to save the learning curves
            
        model_list : None or list of int
        	If None each learning curve is displayed independently. If list
            then the curves are compiled into one figure of two or four panels.
            
        xlim : list
        	The start and end number for the epochs to show
            
        ylim1 : list
        	The range of loss to show in the first figure if model_list is of
            length 2. If model_list is of length 4 it is the range on the first
            and third figures.
            
        ylim2 : list
        	The range of loss to show in the second figure if model_list is of
            length 2. If model_list is of length 4 it is the range on the
            second and fourth figures.
    
        """
        if model_list is not None:
            params = {'figure.autolayout': False,'axes.labelpad':2}
            rcParams.update(params)
            if len(model_list) == 4:
                if figure_directory == 'show':
                    fig = plt.figure()
                else:
                    fig = plt.figure(0,figsize=(7.2,4),dpi=400)
                axes = fig.subplots(nrows=2, ncols=2)
                axis_list = [axes[0, 0],axes[0, 1], axes[1, 0], axes[1, 1]]
                plt.gcf().subplots_adjust(bottom=0.09,top=0.98,left=0.08,right=0.97,wspace=0.05,hspace=0.05)
            elif len(model_list) == 2:
                if figure_directory == 'show':
                    fig = plt.figure()
                else:
                    fig = plt.figure(0,figsize=(3.5,4),dpi=400)
                axes = fig.subplots(nrows=2, ncols=1)
                axis_list = [axes[0],axes[1]]
                plt.gcf().subplots_adjust(bottom=0.09,top=0.98,left=0.14,right=0.97,wspace=0.05,hspace=0.05)
            abcd = ['(a)','(b)','(c)','(d)']
        for ADSORBATE in dictionary.keys():
            for TARGET in dictionary[ADSORBATE].keys():
                for COVERAGE in dictionary[ADSORBATE][TARGET].keys():
                    WL_VAL = np.asarray(dictionary[ADSORBATE][TARGET][COVERAGE]['WL_VAL_mean'])
                    #WL_VAL = WL_VAL.reshape(-1,WL_VAL.shape[-1])
                    WL_VAL_STD = np.asarray(dictionary[ADSORBATE][TARGET][COVERAGE]['WL_VAL_std'])
                    #WL_STD = WL_STD.reshape(-1,WL_STD.shape[-1])
                    WL_TRAIN = np.asarray(dictionary[ADSORBATE][TARGET][COVERAGE]['WL_TRAIN_mean'])
                    #WL_TRAIN = WL_TRAIN.reshape(-1,WL_TRAIN.shape[-1])
                    WL_TRAIN_STD = np.asarray(dictionary[ADSORBATE][TARGET][COVERAGE]['WL_TRAIN_std'])
                    #WL_TRAIN_STD = WL_TRAIN_STD.reshape(-1,WL_TRAIN_STD.shape[-1])
                    try:
                        WL_TEST = np.asarray(dictionary[ADSORBATE][TARGET][COVERAGE]['WL_TEST_TEST'])
                    except:
                        WL_TEST = np.asarray(dictionary[ADSORBATE][TARGET][COVERAGE]['WL_TEST_mean']).reshape(1,-1)
                        WL_TEST_STD = np.asarray(dictionary[ADSORBATE][TARGET][COVERAGE]['WL_TEST_std']).reshape(1,-1)
                        WL_VAL = WL_VAL.reshape(1,-1)
                        WL_VAL_STD = WL_VAL_STD.reshape(1,-1)
                        WL_TRAIN  = WL_TRAIN.reshape(1,-1)
                        WL_TRAIN_STD = WL_TRAIN_STD.reshape(1,-1)
                    #WL_TEST = WL_TEST.reshape(-1,WL_TEST.shape[-1])
                    try:
                        CV_FILES_INDEX = np.asarray(dictionary[ADSORBATE][TARGET][COVERAGE]['CV_FILES_INDEX'])
                    except:
                        CV_FILES_INDEX = [None]
                    for model_result in range(len(CV_FILES_INDEX)):
                        if model_list is None:
                            if figure_directory == 'show':
                                plt.figure(CV_FILES_INDEX[model_result])
                            else:
                                plt.figure(CV_FILES_INDEX[model_result], figsize=(3.5,2),dpi=400)
                            plt.title('ADSORBATE: '+ADSORBATE+', TARGET: '+TARGET+', COVERAGE: '+str(COVERAGE))
                            plt.plot(WL_VAL[model_result],'g')
                            plt.plot(WL_TRAIN[model_result],'b')
                            plt.plot(WL_TEST[model_result],'r')
                            plt.plot(WL_VAL[model_result]+WL_VAL_STD[model_result],'g:')
                            plt.plot(WL_VAL[model_result]-WL_VAL_STD[model_result],'g:')
                            plt.plot(WL_TRAIN[model_result]+WL_TRAIN_STD[model_result],'b:')
                            plt.plot(WL_TRAIN[model_result]-WL_TRAIN_STD[model_result],'b:')
                            try:
                                plt.plot(WL_TEST[model_result]+WL_TEST_STD[model_result],'r:')
                                plt.plot(WL_TEST[model_result]-WL_TEST_STD[model_result],'r:')
                            except:
                                pass
                            plt.legend(['Validation loss','Training loss','Test loss'])
                            if figure_directory == 'show':
                                plt.show()
                            else:
                                figure_path = os.path.join(figure_directory,str(CV_FILES_INDEX[model_result])+'.jpg')
                                plt.savefig(figure_path, format='jpg')
                                plt.close()
                        elif CV_FILES_INDEX[model_result] in model_list:
                            index_val = model_list.index(CV_FILES_INDEX[model_result])
                            axis_list[index_val].plot(WL_VAL[model_result],'g')
                            axis_list[index_val].plot(WL_TRAIN[model_result],'b')
                            axis_list[index_val].plot(WL_TEST[model_result],'r')
                            axis_list[index_val].plot(WL_VAL[model_result]+WL_VAL_STD[model_result],'g:')
                            axis_list[index_val].plot(WL_VAL[model_result]-WL_VAL_STD[model_result],'g:')
                            axis_list[index_val].plot(WL_TRAIN[model_result]+WL_TRAIN_STD[model_result],'b:')
                            axis_list[index_val].plot(WL_TRAIN[model_result]-WL_TRAIN_STD[model_result],'b:')
                            try:
                                axis_list[index_val].plot(WL_TEST[model_result]+WL_TEST_STD[model_result],'r:')
                                axis_list[index_val].plot(WL_TEST[model_result]-WL_TEST_STD[model_result],'r:')
                            except:
                                pass
                            axis_list[index_val].legend(['Validation loss','Training loss','Test loss'])
                            if len(model_list) == 4:
                                if index_val < 2:
                                    axis_list[index_val].set_xticks([])
                                    axis_list[index_val].set_ylim(ylim1)
                                else:
                                    axis_list[index_val].set_ylim(ylim2)
                                if index_val == 1 or index_val == 3:
                                    axis_list[index_val].set_yticks([])
                            elif len(model_list) == 2:
                                if index_val == 0:
                                    axis_list[index_val].set_xticks([])
                                    axis_list[index_val].set_ylim(ylim1)
                                else:
                                    axis_list[index_val].set_ylim(ylim2)
                            axis_list[index_val].set_xlim(xlim)
                            axis_list[index_val].text(0.01,0.93,abcd[index_val],transform=axis_list[index_val].transAxes)
        if model_list is not None:
            fig.text(0.01, 0.5, 'Wasserstein Loss', va='center', rotation='vertical')
            fig.text(0.5, 0.01, 'Epochs', ha='center')
            if figure_directory == 'show':
                plt.show()
            else:
                figure_path = os.path.join(figure_directory,'Learning_Cuves.jpg')
                plt.savefig(figure_path, format='jpg')
                plt.close()
        rcParams.update({'figure.autolayout': True})
        
    def plot_parity_plots(self,figure_directory='show',model_list=None, use_ensemble=False):
        """Plot parity plots for models whose index is in model_list

        Parameters
        ----------
        figure_directory : str
        	Folder location to save the the parity plots
            
        model_list : None or list of int
        	Can be None only if load_CV_class(index) has already been run.
            
        use_ensemble : bool
         Dictates whether an ensemble of CV models is used.
            
    """
        rcParams.update({'lines.markersize': 2.5})
        if model_list is None:
            try:
                if use_ensemble == False:
                    NN = self.NN
                else:
                    NN = self.NN_ENSEMBLE
                num_runs = 1
            except:
                print("LOAD_CROSS_VALIDATION.load_CV_class(index) must be run")
                raise
        else:
            assert type(model_list) == list, "model_list must be None or list"
            num_runs = len(model_list)
        
        for i in range(num_runs):
            if model_list is None:
                model_list = ['no_index']
            else:
                self.load_CV_class(model_list[i])
                if use_ensemble == False:
                    NN = self.NN
                else:
                    NN = self.NN_ENSEMBLE
            print('Model index: '+ str(model_list[i]))
            get_secondary_data = super().get_secondary_data
            INDICES_TEST = self.INDICES_TEST
            
            X_Test, Y_Test = get_secondary_data(200, INDICES_TEST,iterations=10)
            y_test_predict = NN.predict(X_Test)
            NUM_TARGETS = Y_Test.shape[1]
            if NUM_TARGETS < 5:
                if figure_directory == 'show':
                    plt.figure()
                else:
                    plt.figure(0,figsize=(3.5,2),dpi=400)
                marker = ['o','s','^','D']
                ax = plt.subplot()
                for ii in range(NUM_TARGETS):
                    ax.plot(Y_Test[:,ii],y_test_predict[:,ii],marker[ii],zorder=100-i)
                    print('R2: ' + str(error_metrics.get_r2(Y_Test[:,ii],y_test_predict[:,ii])))
                    print('RMSE: ' + str(error_metrics.get_rmse(Y_Test[:,ii],y_test_predict[:,ii])))
                print('WL: ' + str(error_metrics.get_wasserstein_loss(Y_Test,y_test_predict)))
                ax.plot([0,1],[0,1],'k')
                ax.set_xlabel('Actual Percent')
                ax.set_ylabel('Predicted Percent')
                #plt.gcf().subplots_adjust(bottom=0.12,top=0.98,left=0.09,right=0.97)
                ax.tick_params(axis='both', which='major')
                if figure_directory == 'show':
                    plt.show()
                else:
                    figure_path = os.path.join(figure_directory,'Parity_plot_'+str(model_list[i])+'.jpg')
                    plt.savefig(figure_path, format='jpg')
                    plt.close()
            else:
                print('WL: ' + str(error_metrics.get_wasserstein_loss(Y_Test,y_test_predict)))
        rcParams.update({'lines.markersize': 5})
        
class NESTED_DICT(dict):
    """Implementation of perl's autovivification feature.
    		  
    Returns
    -------
    dict : NESTED_DICT
    	Returns a dictionary that is value so that setting key, value pair at
        any key, value level will create all necessary keys/values if they
        don't exist up to that level
            
    """
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

def NESTED_DICT_to_DICT(nested_dict):
    """Converts a nested dict to a traditional python dictionary.
    
    Parameters
    ----------
    nested_dict : NESTED_DICT
        A dictionary of type NESTED_DICT
    		  
    Returns
    -------
    dictionary : dict
    	A python dictionary
            
    """
    dictionary = {}
    def recursive_items(nested_dict,dictionary):
        for key, value in nested_dict.items():
            if type(value) in [NESTED_DICT, dict]:
                value2 = {}
                value2.update(value.copy())
                dictionary.update({key:value2})
                recursive_items(value,value2) 
    recursive_items(nested_dict,dictionary)
    return dictionary

class NN_ENSEMBLE:
    """Class for generating an ensemble of neural networks.   
    """
    def __init__(self, NN_LIST):
        """ 
        Parameters
        ----------
        NN_LIST : list
            A list of type NN
        
        Attributes
        ----------
        NN_LIST : list
            A list of type NN
            
        """
        self.NN_LIST = NN_LIST
        
    def predict(self, X, create_predictions_list=False):
        """Predict value(s) using an ensemble
        
        Parameters
        ----------
        X : numpy.array
            A numpy 2D array of input variables
            
        Attributes
        ----------
        PREDICTIONS_LIST : list
            List of predictions if create_predictions_list=True
            
        STD : numpy.array
            A numpy 2D array of standard deviation in the predictions.
        		  
        Returns
        -------
        PREDICTIONS : numpy.array
            A numpy 2D array of average predictions
                
        """
        NN_LIST = self.NN_LIST
        predictions = []
        for NN in NN_LIST:
            predictions.append(NN.predict(X))
        PREDICTIONS = np.mean(predictions,axis=0)
        if create_predictions_list == True:
            self.PREDICTIONS_LIST = predictions
        self.PREDICTIONS = PREDICTIONS
        self.STD = np.std(predictions,axis=0)
        return PREDICTIONS
    
    
