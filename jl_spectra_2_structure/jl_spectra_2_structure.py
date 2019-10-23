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
from sklearn.cluster import KMeans
from imblearn.over_sampling import RandomOverSampler
from copy import deepcopy
from .neural_network import MLPRegressor

#default values
data_path = pkg_resources.resource_filename(__name__, 'data/')

def get_defaults(adsorbate):
    """
    Returns default frequencies to project intensities onto as well as default
    paths for locations of the pure and mixture spectroscopic data.
    
    Returns
    -------
    frequency_range: numpy.ndarray
        Frequencies over which to project the intensities.
    
    pure_data_path : str
        Directory location where pure-component spectra are stored.
        
    mixture_data_path : str
        Directory location where mixed-component spectra are stored.
    
    """
    nanoparticle_path = os.path.join(data_path, 'dft_nanoparticle/single_'+adsorbate+'.json')
    isotope_path = os.path.join(data_path, 'dft_surface/isotope_'+adsorbate+'.json')
    high_coverage_path = os.path.join(data_path, 'dft_surface/high_coverage_'+adsorbate+'.json')
    cross_validation_path = os.path.join(data_path, 'cross_validation')
    coverage_scaling_path = os.path.join(data_path,'coverage_scaling_params_'+adsorbate+'.json')
    return nanoparticle_path, isotope_path, high_coverage_path\
           , cross_validation_path, coverage_scaling_path

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

class IR_GEN:
    def __init__(self, nanoparticle_path, TARGET='CNCO', NUM_TARGETS=4):
        #number of target variablesa.
        coverage_scaling_path
        self.TARGET = TARGET
        self.NUM_TARGETS = NUM_TARGETS
        self.GCNLabel = None
        with open(nanoparticle_path, 'r') as infile:
            nanoparticle_data = json.load(infile)
        Xfreq_ALL = np.array(nanoparticle_data['FREQUENCIES'], dtype='float')
        is_local_minima = np.min(Xfreq_ALL, axis=1) > 0
        #CNCOs = np.array([i.CNCO for i in nanoparticle_data])
        #is_outlierBridge = np.all((np.max(Xfreq_ALL,axis=1)<1700,CNCOs==2),axis=0)
        #del CNCOs
        #select_files = np.all((is_local_minima,is_outlierBridge==False),axis=0)
        for key in nanoparticle_data.keys():
            nanoparticle_data[key] = np.array(nanoparticle_data[key])[is_local_minima]
        nanoparticle_data['CN_CO'][nanoparticle_data['CN_CO'] == 5] = 4
        nanoparticle_data['CN_CO'][nanoparticle_data['CN_CO'] == 0] = 4
        self.CNCOList_original = nanoparticle_data['CN_CO']
        nanoparticle_data['INTENSITIES'][nanoparticle_data['FREQUENCIES'] == 0] = 0
        if TARGET == 'CNCO1HOLLOW':
            nanoparticle_data['CN_CO'][nanoparticle_data['CN_CO'] == 4] = 3
            print('grouping hollow sites')
        self.X0cov = np.array([(nanoparticle_data['FREQUENCIES'][i], nanoparticle_data['INTENSITIES'][i])
                               for i in range(len(nanoparticle_data['FREQUENCIES']))])
        self.CNCOList = nanoparticle_data['CN_CO']
        self.GCNList = nanoparticle_data['GCN']

    def get_GCNlabels(self, Minimum=7, showfigures=False):
        print('Initial number of targets: '+str(self.NUM_TARGETS))
        NUM_TARGETS = self.NUM_TARGETS
        GCNList = self.GCNList
        CNCOList = self.CNCOList
        GCNListAtop = GCNList[CNCOList == 1]
        KC = KMeans(n_clusters=NUM_TARGETS, random_state=0).fit(GCNListAtop.reshape(-1, 1))
        KC_new = np.zeros(NUM_TARGETS, dtype='int')
        KC_new[0] = KC.labels_[np.argmin(GCNListAtop)]
        for i in range(NUM_TARGETS-1):
            KC_new[i+1] = KC.labels_[np.isin(KC.labels_, KC_new[0:i+1]) == False]\
            [np.argmin(GCNListAtop[np.isin(KC.labels_, KC_new[0:i+1]) == False])]
        KC2class = dict(zip(KC_new, np.arange(1, len(KC_new)+1, dtype='int')))
        KCclass = np.array([KC2class[i] for i in KC.labels_])
        for i in range(NUM_TARGETS-1)[::-1]:
            NUM_IN_CLASS = len(KCclass[KCclass == i+2])
            if NUM_IN_CLASS < Minimum:
                for ii in range(NUM_TARGETS)[::-1]:
                    if KC2class[ii] == i+2:
                        KC2class.update({ii:i+1})
                KCclass = np.array([KC2class[ii] for ii in KC.labels_])
        NUM_IN_CLASS = len(KCclass[KCclass == 1])
        if NUM_IN_CLASS < Minimum:
            for i in range(NUM_TARGETS):
                if KC2class[i] > 1:
                    KC2class.update({i:KC2class[i]-1})
        NUM_IN_CLASS = len(KCclass[KCclass == 2])
        if NUM_IN_CLASS < Minimum:
            for i in range(NUM_TARGETS):
                if KC2class[i] > 2:
                    KC2class.update({i:KC2class[i]-1})
        KCclass = np.array([KC2class[i] for i in KC.labels_])

        NUM_TARGETS = len(set(KCclass))

        GCNlabel = np.zeros(len(GCNList), dtype='int')
        GCNlabel[(CNCOList == 1)] = KCclass
        BreakPoints = np.linspace(0, 8.5, num=810)
        BreakLabels = KC.predict(BreakPoints.reshape(-1, 1))
        BreakLabels = [KC2class[i] for i in BreakLabels]
        BreakMin = np.array([np.min(BreakPoints[BreakLabels == i])
                             for i in np.arange(1, NUM_TARGETS+1)])
        BreakMax = np.array([np.max(BreakPoints[BreakLabels == i])
                             for i in np.arange(1, NUM_TARGETS+1)])
        if showfigures == True:
            import matplotlib.pyplot as plt
            plt.figure(0)
            plt.scatter(GCNListAtop, GCNListAtop, c=KC.labels_)
            for i in BreakMax:
                plt.plot([0, i, 2*i], [2*i, i, 0], 'k-')
            plt.xlabel('GCN')
            plt.ylabel('GCN')
            plt.xlim([0, 8.5])
            plt.ylim([0, 8.5])
            BreakString = zip(np.around(BreakMin, decimals=1), np.around(BreakMax, decimals=1))
            BreakString = [str(count+1)+': '+str(i[0])+'-'+str(i[1])
                           for count, i in enumerate(BreakString)]
            plt.figure(1)
            #ax = plt.subplot()
            plt.hist(GCNlabel[(CNCOList == 1)], bins=np.arange(0.5, NUM_TARGETS+1.5), rwidth=0.5)
            plt.xticks(range(1, NUM_TARGETS+1))
            #ax.set_xticklabels(greater_than)
            plt.xlabel('GCN Group')
            plt.ylabel('DFT Samples')
            plt.show()
            print(BreakString)
            from matplotlib import rcParams
            rcParams['lines.linewidth'] = 2
            rcParams['lines.markersize'] = 5
            params = {'figure.autolayout': True}
            rcParams.update(params)
            plt.figure(2, figsize=(3.5, 3), dpi=300)
            #ax = plt.subplot()
            plt.hist(GCNlabel[(CNCOList == 1)], bins=np.arange(0.5, NUM_TARGETS+1.5), rwidth=0.5)
            plt.xticks(range(1, NUM_TARGETS+1))
            #ax.set_xticklabels(greater_than)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.xlabel('GCN Group', fontsize=8)
            plt.ylabel('DFT Samples', fontsize=8)
            plt.savefig('../Figures/GCN_Clustering.png', format='png')
            plt.close()

        self.GCNLabel = GCNlabel
        self.NUM_TARGETS = NUM_TARGETS
        print('Final number of targets: '+str(self.NUM_TARGETS))

    def _get_probabilities(self, num_samples, NUM_TARGETS):
        #define np_shuffle to be locally in loop
        probabilities = np.zeros((num_samples, NUM_TARGETS))
        A = np.zeros(num_samples)
        for i in range(NUM_TARGETS-1):
            probs = (1-A)*np.random.random_sample(size=num_samples)
            A = A+probs
            probabilities[:, i] = probs
        probabilities[:, -1] = 1 - np.sum(probabilities, axis=1)
        #set any slightly negative probabilities to 0
        probabilities[probabilities[...] < 2**-500] = 0
        [np.random.shuffle(i) for i in probabilities]
        return probabilities

    def _perturb_spectra(self, perturbations, X, y, a=0.999, b=1.001, CNCO=None):
        X = X.copy(); y = y.copy()
        perturbed_values = (b-a)*np.random.random_sample((X.shape[0]*perturbations
                                                          , X.shape[1], X.shape[2]))+a
        Xperturbed = X.repeat(perturbations, axis=0)*perturbed_values
        yperturbed = y.repeat(perturbations, axis=0)
        if CNCO is not None:
            CNCO = CNCO.copy()
            CNCO_perturbed = CNCO.repeat(perturbations,axis=0)
            return(Xperturbed, yperturbed, CNCO_perturbed)
        else:
            return(Xperturbed, yperturbed)
   
    def _mixed_lineshape(self, FWHM, fL, ENERGY_POINTS, energy_spacing):
        """accepts numpy array of frequencies and intensities and FWHM
        returns spectrum
        Wertheim 1974"""
        numpoints = 2*ENERGY_POINTS-1
        x = energy_spacing*(np.arange(numpoints, dtype='int')-int((numpoints-1)/2))
        b = 0.5*np.sqrt(np.log(2))
        specL = 1.0/(1.0+4.0*(x/FWHM)**2)
        specG = np.exp(-(x/(b*FWHM))**2)
        transform = fL*specL+(1-fL)*specG
        return transform

    def _coverage_shift(self, X, CNCO, SELF_COVERAGE, TOTAL_COVERAGE):
        """
        COVERAGE: Relative spatial coverage of each binding-type
        TOTAL_COVERAGE: relative combined coverages of non-island
        """
        ones = np.ones(X.shape[0])
        Xfrequencies = X[:, 0].copy()
        Xintensities = X[:, 1].copy()
        if type(SELF_COVERAGE) == int or type(SELF_COVERAGE) == float:
            SELF_COVERAGE = ones*SELF_COVERAGE
        if type(TOTAL_COVERAGE) == int or type(TOTAL_COVERAGE) == float:
            TOTAL_COVERAGE = ones*TOTAL_COVERAGE
        with open(coverage_scaling_path, 'r') as infile:
            Coverage_Scaling = json.load(infile)
        #maximum coverage obtained from Norton et al. 1982
        ABS_COVERAGE = SELF_COVERAGE*0.096
        TOTAL_COVERAGE_ABS = TOTAL_COVERAGE*0.096
        #site-type: atop, bridge,threefold,fourfold
        CO_STRETCH_IDX = np.argmax(Xfrequencies, axis=1)
        #Copy frequencies and update CO stretch before updating all, then set CO stretch
        CO_frequencies = Xfrequencies.copy()[np.arange(len(Xfrequencies)), CO_STRETCH_IDX]
        CO_intensities = Xintensities.copy()[np.arange(len(Xintensities)), CO_STRETCH_IDX]
        for i in range(4):
            CO_frequencies[CNCO == i+1] = (CO_frequencies[CNCO == i+1]\
                                           *(Coverage_Scaling['CO_FREQ'][i]['SELF_CO_PER_A2']*ABS_COVERAGE[CNCO == i+1]\
                                             +Coverage_Scaling['CO_FREQ'][i]['CO_PER_A2']*TOTAL_COVERAGE_ABS[CNCO == i+1]\
                                             +Coverage_Scaling['CO_FREQ'][i]['const'])\
                                             /Coverage_Scaling['CO_FREQ'][i]['const'])
            CO_intensities[CNCO == i+1] = (CO_intensities[CNCO == i+1]\
                                         *np.exp(Coverage_Scaling['CO_INT_EXP']*TOTAL_COVERAGE_ABS[CNCO == i+1]))
            
            Xfrequencies[CNCO == i+1] = (Xfrequencies[CNCO == i+1]\
                                           *(Coverage_Scaling['PTCO_FREQ']['SELF_CO_PER_A2']*ABS_COVERAGE[CNCO == i+1]\
                                             +Coverage_Scaling['PTCO_FREQ']['CO_PER_A2']*TOTAL_COVERAGE_ABS[CNCO == i+1]\
                                             +Coverage_Scaling['PTCO_FREQ']['const']).reshape((-1, 1))\
                                             /Coverage_Scaling['PTCO_FREQ']['const'])
            Xintensities[CNCO == i+1] = (Xintensities[CNCO == i+1]\
                                         *np.exp(Coverage_Scaling['PTCO_INT_EXP']*TOTAL_COVERAGE_ABS[CNCO == i+1]).reshape((-1,1)))                          
        Xfrequencies[np.arange(X.shape[0]), CO_STRETCH_IDX] = CO_frequencies
        Xintensities[np.arange(X.shape[0]), CO_STRETCH_IDX] = CO_intensities
        Xcov = np.array([np.array((Xfrequencies[i], Xintensities[i]))
                         for i in range(X.shape[0])])
        Xcov[Xcov[...] < 2**-500] = 0
        return Xcov

    def _scaling_factor_shift(self, X):
        Xfrequencies = X[:, 0].copy()
        Xintensities = X[:, 1].copy()
        CO_STRETCH_IDX = Xfrequencies > 1000
        MC_IDX = Xfrequencies < 1000
        #Scaling Factor determined from comparing experiment to DFT
        #uncertainties are 9.3e-6 and 0.00182 respectively
        SFCO = 1.012111
        SFMC = 0.96851
        Xfrequencies[CO_STRETCH_IDX] = Xfrequencies[CO_STRETCH_IDX]*SFCO
        Xfrequencies[MC_IDX] = Xfrequencies[MC_IDX]*SFMC
        X = np.array([np.array((Xfrequencies[i], Xintensities[i]))
                      for i in range(len(Xfrequencies))])
        X[X[...] < 2**-500] = 0
        return X

    def _generate_spectra(self, Xfrequencies, Xintensities, energies2D, prefactor, sigma):
        ENERGY_POINTS = self.ENERGY_POINTS
        int_mesh = np.zeros((Xfrequencies.shape[0], ENERGY_POINTS))
        #mesh everything on energy grids of 4 cm-1 spacings with a FWHM of 2
        for i in range(Xfrequencies.shape[0]):
            freq2D = np.tile(Xfrequencies[i], (ENERGY_POINTS, 1))
            int2D = np.tile(Xintensities[i], (ENERGY_POINTS, 1))
            temp = int2D*prefactor*np.exp(-(freq2D-energies2D)**2/(2.0*sigma**2))
            int_mesh[i] = np.sum(temp, axis=1)
        int_mesh[abs(int_mesh[...])<2**-500] = 0
        return int_mesh
    
    def _xyconv(self, X_sample, Y_sample, probabilities, CNCO_sample):
        get_probabilities = self._get_probabilities
        mixed_lineshape = self._mixed_lineshape
        coverage_shift = self._coverage_shift
        generate_spectra = self._generate_spectra
        NUM_TARGETS = self.NUM_TARGETS
        LOW_FREQUENCY = self.LOW_FREQUENCY
        HIGH_FREQUENCY = self.HIGH_FREQUENCY
        ENERGY_POINTS = self.ENERGY_POINTS
        COVERAGE = self.COVERAGE
        TARGET = self.TARGET
        
        energies = np.linspace(LOW_FREQUENCY, HIGH_FREQUENCY, num=ENERGY_POINTS\
                               , endpoint=True)
        energy_spacing = energies[1]-energies[0]
        FWHM = 2*energy_spacing
        sigma = FWHM/(2.0 * np.sqrt(2.0 * np.log(2.)))
        prefactor = 1.0/(sigma * np.sqrt(2.0 * np.pi))
        energies2D = energies.reshape((-1, 1))
        MIN_Y = int(min(Y_sample))
        Xfrequencies = X_sample[:, 0].copy()
        Xintensities = X_sample[:, 1].copy()
        np.random.shuffle(probabilities)
        num_samples = len(probabilities)
        Nanos = np.random.randint(1, high=201, size=num_samples)
        fLs = np.random.sample(num_samples)
        FWHMs = np.random.uniform(low=2, high=75, size=num_samples)
        Xconv = np.zeros((num_samples, ENERGY_POINTS))
        yconv = np.zeros((num_samples, NUM_TARGETS))
        y_mesh = np.zeros((Y_sample.size, NUM_TARGETS))
        y_mesh[np.arange(Y_sample.size), Y_sample-MIN_Y] = 1
        sample_indices = np.arange(Y_sample.size)
        parray = np.zeros(Y_sample.size)
        coverage_parray = get_probabilities(num_samples, 11)
        coverage_totals = np.random.random_sample(size=[num_samples,10])
        if COVERAGE == 'low' or TARGET not in ['CNCO', 'CNCO1HOLLOW']:
            int_mesh = generate_spectra(Xfrequencies, Xintensities\
                                        ,energies2D, prefactor, sigma)
        for i in range(num_samples):
            for ii in range(NUM_TARGETS):
                parray[Y_sample == ii+MIN_Y] = probabilities[i, ii]
            parray /= np.sum(parray)
            indices_primary = np.random.choice(sample_indices, size=Nanos[i], replace=True, p=parray)
            if COVERAGE == 'low' or TARGET not in ['CNCO', 'CNCO1HOLLOW']:
                combined_mesh = np.sum(int_mesh[indices_primary], axis=0)
            else:
                #initialize coverages
                SELF_COVERAGE = np.random.random_sample(Nanos[i])
                TOTAL_COVERAGE = np.zeros_like(SELF_COVERAGE)
                COVERAGE_INDICES = np.random.choice([0,1,2,3,4,5,6,7,8,9,10], size=Nanos[i],replace=True, p=coverage_parray[i])
                #self coverage corresponding to index of 0 is island so it is single coverage and is skipped
                for ii in sorted(set(COVERAGE_INDICES.tolist()+[0]))[1:]:
                    TOTAL_COVERAGE[COVERAGE_INDICES == ii] = coverage_totals[i][ii-1]
                    #Set coverage of each spectra to be total coverage divided by the number of spectra being combined
                    SELF_COVERAGE[COVERAGE_INDICES == ii ] = coverage_totals[i][ii-1]/TOTAL_COVERAGE[COVERAGE_INDICES == ii].size
                    #update self coverage of indentical binding types to be the same (their sum)
                    for iii in [1,2,3,4]:
                        SELF_COVERAGE[np.all((COVERAGE_INDICES == ii,CNCO_sample[indices_primary]==iii),axis=0)] \
                        = np.sum(SELF_COVERAGE[np.all((COVERAGE_INDICES == ii,CNCO_sample[indices_primary]==iii),axis=0)])
                #decrease maximum coverage at less favorable sites to improve prediction score
                SELF_COVERAGE[CNCO_sample[indices_primary] == 2] *= 0.7
                SELF_COVERAGE[np.any((CNCO_sample[indices_primary] == 3,CNCO_sample[indices_primary] == 4), axis=0)] *= 0.2
                for ii in sorted(set(COVERAGE_INDICES.tolist()+[0]))[1:]:
                    TOTAL_COVERAGE[COVERAGE_INDICES == ii] *= ( \
                            SELF_COVERAGE[np.all((COVERAGE_INDICES == ii,CNCO_sample[indices_primary] == 1), axis=0)].size\
                          + 0.7 * SELF_COVERAGE[np.all((COVERAGE_INDICES == ii,CNCO_sample[indices_primary] == 2),axis=0)].size \
                          + 0.2 * SELF_COVERAGE[np.all((COVERAGE_INDICES == ii,CNCO_sample[indices_primary] == 3),axis=0)].size \
                          + 0.2 * SELF_COVERAGE[np.all((COVERAGE_INDICES == ii,CNCO_sample[indices_primary] == 4),axis=0)].size \
                          )/TOTAL_COVERAGE[COVERAGE_INDICES == ii].size        
                TOTAL_COVERAGE[COVERAGE_INDICES==0] = SELF_COVERAGE[COVERAGE_INDICES==0]
                Xcov = coverage_shift(X_sample[indices_primary], CNCO_sample[indices_primary], SELF_COVERAGE, TOTAL_COVERAGE)
                Xcovfrequencies = Xcov[:, 0].copy()
                Xcovintensities = Xcov[:, 1].copy()
                int_mesh = generate_spectra(Xcovfrequencies,Xcovintensities\
                                                  ,energies2D, prefactor, sigma)
                int_mesh[CNCO_sample[indices_primary] == 2] *= 0.7
                int_mesh[CNCO_sample[indices_primary] == 3] *= 0.2
                int_mesh[CNCO_sample[indices_primary] == 4] *= 0.2
                combined_mesh = np.sum(int_mesh, axis=0)
            yconv[i] = np.sum(y_mesh[indices_primary], axis=0, dtype='int')
            transform = mixed_lineshape(FWHMs[i], fLs[i], ENERGY_POINTS, energy_spacing)
            Xconv[i] = np.convolve(combined_mesh, transform, mode='valid')
        #This part is to adjust the tabulated contribution to match that of the adjusted spectra for max coverage of bridge, 3-fold and 4-fold
        if COVERAGE == 'high' and TARGET in ['CNCO', 'CNCO1HOLLOW']:
            yconv[:,1] *= 0.7
            yconv[:,2] *= 0.2
            if TARGET == 'CNCO':
                yconv[:,3] *= 0.2
        Xconv += 10**-3*np.max(Xconv, axis=1).reshape((-1, 1))*np.random.random_sample(Xconv.shape)
        Xconv[abs(Xconv[...])<2**-500] = 0
        yconv[abs(yconv[...])<2**-500] = 0
        Xconv /= np.max(Xconv, axis=1).reshape((-1, 1))
        yconv /= np.sum(yconv, axis=1).reshape((-1, 1))
        Xconv[abs(Xconv[...])<2**-500] = 0
        yconv[abs(yconv[...])<2**-500] = 0
        return Xconv, yconv

    def add_noise(self, Xconv_main, Xconv_noise, noise2signalmax=2.0/3.0):
        X_noisey = np.zeros((len(Xconv_main), len(Xconv_main[0])))
        noise_value = np.random.uniform(low=0, high=noise2signalmax, size=len(Xconv_main))
        noise_indices = np.arange(len(Xconv_noise))
        noise_sample = np.random.choice(noise_indices, size=len(Xconv_main), replace=True)
        for i in range(len(Xconv_main)):
            X_noisey[i] = Xconv_main[i] + Xconv_noise[noise_sample[i]]*noise_value[i]
        X_noisey[abs(X_noisey[...])<2**-500] = 0
        X_noisey = X_noisey/np.max(X_noisey, axis=1).reshape(-1, 1)
        X_noisey[abs(X_noisey[...])<2**-500] = 0
        return X_noisey

    def get_synthetic_spectra(self, NUM_SAMPLES, indices, COVERAGE=None
                              , LOW_FREQUENCY=200, HIGH_FREQUENCY=2200, ENERGY_POINTS=501):
        TARGET = self.TARGET
        CNCO = self.CNCOList
        CNCO_original = self.CNCOList_original
        self.COVERAGE = COVERAGE
        self.LOW_FREQUENCY = LOW_FREQUENCY
        self.HIGH_FREQUENCY = HIGH_FREQUENCY
        self.ENERGY_POINTS = ENERGY_POINTS
        #Assign the target variable Y to either GCN group or binding site
        if TARGET == 'GCN':
            if self.GCNLabel is None:
                #Assigns a class to ranges of GCN values
                self.get_GCNlabels(Minimum=0, showfigures=False)
            Y = self.GCNLabel
        else:
            Y = CNCO
        #correct self.NUM_TARGETS in case this method is run multiple times
        self.NUM_TARGETS = len(set(Y[indices]))
        print('NUM_TARGETS: '+str(self.NUM_TARGETS))
        X0cov = self._scaling_factor_shift(self.X0cov)
        #Adding Data for Extended Surfaces
        if COVERAGE == 'high' and self.TARGET == 'GCN':
            print('Adding high-coverage low index planes')
            with open(high_coverage_path, 'r') as infile:
                HC_Ext = json.load(infile)
            HC_CNPt = np.sort(np.array(list(set([np.min(i) for i in HC_Ext['CN_PT']]))))
            self.NUM_TARGETS += len(HC_CNPt)
            HC_frequencies = []
            HC_intensities = []
            HC_classes = []
            max_freqs = np.max([len(i) for i in HC_Ext['FREQUENCIES']])
            for counter, i in enumerate(HC_CNPt):
                for ii in range(len(HC_Ext['CN_PT'])):
                    if np.min(HC_Ext['CN_PT'][ii]) == i:
                        HC_classes.append(np.max(Y)+counter+1)
                        num_atop = len(np.array(HC_Ext['CN_CO'][ii]
                                               )[np.array(HC_Ext['CN_CO'][ii]) == 1])
                        offset = max_freqs-len(HC_Ext['FREQUENCIES'][ii])
                        HC_frequencies.append(np.pad(HC_Ext['FREQUENCIES'][ii], (0, offset)
                                                     , 'constant', constant_values=0))
                        HC_intensities.append(np.pad(HC_Ext['INTENSITIES'][ii], (0, offset)
                                                     , 'constant', constant_values=0)/num_atop)
            HC_classes = np.array(HC_classes)
            #Change all high-coverage classes to just max(y)+1
            #until more accurate experiments and DFT are available
            self.NUM_TARGETS = self.NUM_TARGETS - len(HC_CNPt) + 1
            HC_classes[...] = self.NUM_TARGETS
            #
            HC_frequencies = np.array(HC_frequencies)
            HC_intensities = np.array(HC_intensities)
            HC_X = np.array([np.array((HC_frequencies[i], HC_intensities[i]))
                             for i in range(len(HC_frequencies))])
            HC_X = self._scaling_factor_shift(HC_X)
            offset = max_freqs-len(X0cov[0][0])
            X = np.pad(X0cov, ((0, 0), (0, 0), (0, offset)), 'constant', constant_values=0)
        elif (COVERAGE == 'high' and (TARGET == 'CNCO' or TARGET == 'CNCO1HOLLOW')):
            print('testing all coverages')
            X = X0cov
        elif type(COVERAGE) == int or type(COVERAGE) == float:
            print('Relative coverage is ' + str(COVERAGE))
            X = self._coverage_shift(X0cov, CNCO, COVERAGE,COVERAGE)
        elif COVERAGE == 'low':
            print('Low Coverage')
            X = X0cov
        else:
            print('Error in Input')
            return
        self.X = X
        self.Y = Y
        if COVERAGE == 'high' and self.TARGET == 'GCN':
            self.HC_X = HC_X
            self.HC_classes = HC_classes
            num_single = Y.size
            #b = 2105/2095 #a = 1854/1865
            HC_X_expanded, HC_classes_expanded = self._perturb_spectra(5, HC_X, HC_classes
                                                                       , a=0.995, b=1.01)
            X = np.concatenate((X, HC_X_expanded), axis=0)
            Y = np.concatenate((Y, HC_classes_expanded), axis=0)
            indices = list(indices) + np.arange(Y.size)[num_single:].tolist()     
       
        indices_balanced, Y_balanced = RandomOverSampler().fit_sample(np.array(indices).reshape(-1,1),Y[indices].copy())
        X_balanced = X[indices_balanced.flatten()]
        
        
        CNCO_sample = None
        #adding perturbations for improved fitting
        if COVERAGE == 'high' and (TARGET == 'CNCO' or TARGET == 'CNCO1HOLLOW'):
            CNCO_balanced = CNCO_original[indices_balanced.flatten()]
            X_sample, Y_sample, CNCO_sample = self._perturb_spectra(5, X_balanced, Y_balanced
                                                                    , a=0.995, b=1.005,CNCO=CNCO_balanced)
        elif COVERAGE == 'high' and TARGET == 'GCN':
            X_sample, Y_sample = self._perturb_spectra(5, X_balanced, Y_balanced
                                                       , a=0.9975, b=1.0025)
        else:
            X_sample, Y_sample = self._perturb_spectra(5, X_balanced, Y_balanced, a=0.999, b=1.001)
        probabilities = self._get_probabilities(NUM_SAMPLES, self.NUM_TARGETS)
        Xconv, yconv = self._xyconv(X_sample, Y_sample, probabilities, CNCO_sample)
        return Xconv, yconv

    def get_more_spectra(self, NUM_SAMPLES, indices):
        TARGET = self.TARGET
        X = self.X
        Y = self.Y
        CNCO_original = self.CNCOList_original
        COVERAGE = self.COVERAGE
        if COVERAGE == 'high' and self.TARGET == 'GCN':
            HC_X = self.HC_X
            HC_classes = self.HC_classes
            num_single = Y.size
            #b = 2105/2095 #a = 1854/1865
            HC_X_expanded, HC_classes_expanded = self._perturb_spectra(5, HC_X, HC_classes
                                                                       , a=0.995, b=1.01)
            X = np.concatenate((X, HC_X_expanded), axis=0)
            Y = np.concatenate((Y, HC_classes_expanded), axis=0)
            indices = list(indices) + np.arange(Y.size)[num_single:].tolist()        
       
        indices_balanced, Y_balanced = RandomOverSampler().fit_sample(np.array(indices).reshape(-1,1),Y[indices].copy())
        X_balanced = X[indices_balanced.flatten()]
        
        CNCO_sample = None
        #adding perturbations for improved fitting
        if COVERAGE == 'high' and (TARGET == 'CNCO' or TARGET == 'CNCO1HOLLOW'):
            CNCO_balanced = CNCO_original[indices_balanced.flatten()]
            X_sample, Y_sample, CNCO_sample = self._perturb_spectra(5, X_balanced, Y_balanced
                                                                    , a=0.995, b=1.005,CNCO=CNCO_balanced)
        elif COVERAGE == 'high' and TARGET == 'GCN':
            X_sample, Y_sample = self._perturb_spectra(5, X_balanced, Y_balanced
                                                       , a=0.9975, b=1.0025)
        else:
            X_sample, Y_sample = self._perturb_spectra(5, X_balanced, Y_balanced, a=0.999, b=1.001)
        probabilities = self._get_probabilities(NUM_SAMPLES, self.NUM_TARGETS)
        Xconv, yconv = self._xyconv(X_sample, Y_sample, probabilities, CNCO_sample)
        return Xconv, yconv
    
def fold(frequencies, intensities, LOW_FREQUENCY, HIGH_FREQUENCY, ENERGY_POINTS,FWHM):
    energies = np.linspace(LOW_FREQUENCY, HIGH_FREQUENCY, num=ENERGY_POINTS, endpoint=True)
    energy_spacing = energies[1]-energies[0]
    if FWHM < 2*energy_spacing:
        raise ValueError('Function input FWHM must must be at least twice\
        the energy spacing to prevent information loss. It therefore must be\
        at least' + str(2*energy_spacing))
    sigma = FWHM/(2.0 * np.sqrt(2.0 * np.log(2.)))
    prefactor = 1.0/(sigma * np.sqrt(2.0 * np.pi))
    #Reshape array energies to be (ENERGY_POINTS,1) dimensional
    energies2D = energies.reshape((-1, 1))
    #Create array that is duplicate of frequencies
    freq2D = np.tile(frequencies, (ENERGY_POINTS, 1))
    int2D = np.tile(intensities, (ENERGY_POINTS, 1))
    #contribution of each frequency on each energy to total intensity
    int_matrix = int2D*prefactor*np.exp(-(freq2D-energies2D)**2/(2.0*sigma**2))
    spectrum = np.sum(int_matrix, axis=1)
    return spectrum

def HREEL_2_scaledIR(HREEL, frequency_range=np.linspace(200,2200,num=501,endpoint=True) ):
    PEAK_CONV = 2.7
    IR = np.interp(frequency_range, HREEL[0], HREEL[1]*HREEL[0]**PEAK_CONV, left=None, right=None, period=None)
    IR_scaled = IR/np.max(IR)
    return IR_scaled

def wasserstein_loss(y_true, y_pred,individual=False):
    """Compute the l2 wasserstein loss

    Parameters
    ----------
    y_true : array-like or label indicator matrix
    Ground truth (correct) values.

    y_pred : array-like or label indicator matrix
    Predicted values, as returned by a regression estimator.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    y_true = np.array(deepcopy(y_true))
    y_pred = np.array(deepcopy(y_pred))
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1,1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1,1)
    Tcum = np.cumsum(y_true, axis=-1)
    Pcum = np.cumsum(y_pred, axis=-1)
    w_loss = (1/float(y_true.shape[1])*np.sum((Pcum-Tcum)**2, axis=1))**0.5
    if individual == True:
        return w_loss
    else:
        return w_loss.mean()

def r2(y_true, y_pred):
    SStot = np.sum((y_true-y_true.mean())**2)
    SSres = np.sum((y_true-y_pred)**2)
    return 1 - SSres/SStot

def get_NN(dictionary):
    NN = MLPRegressor()
    NN.set_params(**dictionary['parameters'])
    #catches instances where coefficients and intercepts are saved via standard json package as list of lists
    dictionary['__getstate__']['coefs_'] = [np.array(coef_list) for coef_list in dictionary['__getstate__']['coefs_'].copy()]
    dictionary['__getstate__']['intercepts_'] = [np.array(coef_list) for coef_list in dictionary['__getstate__']['intercepts_'].copy()]
    NN.__setstate__(dictionary['__getstate__'])
    return NN