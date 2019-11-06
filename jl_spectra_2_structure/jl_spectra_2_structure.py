# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:05:24 2017

@author: lansf
"""
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import json
import pkg_resources
from sklearn.cluster import KMeans
from imblearn.over_sampling import RandomOverSampler

def get_default_data_paths(adsorbate):
    """
    Returns default frequencies to project intensities onto as well as default
    paths for locations of the pure and mixture spectroscopic data.
    
    Returns
    -------
    
    nanoparticle_path : str
        File path where nanoparticle or single adsorbate data will be saved.
        
    isotope_path : str
        File path where isotope data for CO and NO will be saved.

    high_coverage_path : str
        File path where high coverage data for CO will be saved.
        
    cross_validation_path : str
        Directory location cross validation data will be stored.

    coverage_scaling_path : str
        File path where coverage scaling coefficients will be saved.
    
    """
    data_path = pkg_resources.resource_filename(__name__, 'data/')
    nanoparticle_path = os.path.join(data_path, 'dft_nanoparticle/single_'+adsorbate+'.json')
    isotope_path = os.path.join(data_path, 'dft_surface/isotope_'+adsorbate+'.json')
    high_coverage_path = os.path.join(data_path, 'dft_surface/high_coverage_'+adsorbate+'.json')
    coverage_scaling_path = os.path.join(data_path,'coverage_scaling_params_'+adsorbate+'.json') 
    return (nanoparticle_path, isotope_path, high_coverage_path\
           , coverage_scaling_path)

class IR_GEN:
    def __init__(self, ADSORBATE='CO', INCLUDED_BINDING_TYPES=[1,2,3,4], TARGET='binding_type', NUM_TARGETS=None\
                 , nanoparticle_path=None, high_coverage_path=None, coverage_scaling_path=None):
        assert TARGET in ['binding_type','GCN','combine_hollow_sites'], "incorrect TARGET given"
        assert type(INCLUDED_BINDING_TYPES) in [list,tuple,np.ndarray], "Included Binding Types should be a list"
        #number of target variables.
        nano_path, isotope_path, high_cov_path\
           , cov_scale_path = get_default_data_paths(ADSORBATE)
        
        if nanoparticle_path is None:
            nanoparticle_path = nano_path
        if ADSORBATE in ['NO','CO']:
            max_coordination=4
            if coverage_scaling_path is None:
                coverage_scaling_path = cov_scale_path
            if ADSORBATE == 'CO':
                if high_coverage_path is None:
                    high_coverage_path = high_cov_path
        elif ADSORBATE in ['C2H4']:
            max_coordination = 2
        
        with open(nanoparticle_path, 'r') as infile:
            nanoparticle_data = json.load(infile)
        Xfreq_ALL = np.array(nanoparticle_data['FREQUENCIES'], dtype='float')
        is_local_minima = np.min(Xfreq_ALL, axis=1) > 0
        BINDING_TYPES_unfiltered = np.array(nanoparticle_data['CN_ADSORBATE'])
        max_forces = np.array(nanoparticle_data['MAX_FORCE'])
        is_adsorbed = BINDING_TYPES_unfiltered >0
        if TARGET in ['binding_type','combine_hollow_sites']:
            NUM_TARGETS = len(INCLUDED_BINDING_TYPES)
        correct_coordination = BINDING_TYPES_unfiltered <= max_coordination
        small_force = max_forces < 0.05
        select_files = np.all((is_local_minima,is_adsorbed,small_force,correct_coordination),axis=0)
        for key in nanoparticle_data.keys():
            nanoparticle_data[key] = np.array(nanoparticle_data[key])[select_files]
        nanoparticle_data['INTENSITIES'][nanoparticle_data['FREQUENCIES'] == 0] = 0
        BINDING_TYPES_with_4fold = nanoparticle_data['CN_ADSORBATE']
        if TARGET == 'combine_hollow_sites':
            nanoparticle_data['CN_ADSORBATE'][nanoparticle_data['CN_ADSORBATE'] == 4] = 3
            NUM_TARGETS -= 1
            print('grouping hollow sites')
        self.BINDING_TYPES_with_4fold = BINDING_TYPES_with_4fold
        self.TARGET = TARGET
        self.NUM_TARGETS = NUM_TARGETS
        self.GCNlabels = None
        self.X0cov = np.array([(nanoparticle_data['FREQUENCIES'][i], nanoparticle_data['INTENSITIES'][i])
                               for i in range(len(nanoparticle_data['FREQUENCIES']))])
        self.BINDING_TYPES = nanoparticle_data['CN_ADSORBATE']
        self.GCNList = nanoparticle_data['GCN']
        self.NANO_PATH = nanoparticle_path
        self.HIGH_COV_PATH = high_coverage_path
        self.COV_SCALE_PATH = coverage_scaling_path
        self.ADSORBATE = ADSORBATE
        self.INCLUDED_BINDING_TYPES = INCLUDED_BINDING_TYPES
        self.COVERAGE = None

    def get_GCNlabels(self, Minimum=0, showfigures=False, BINDING_TYPE_FOR_GCN=[1]):
        print('Initial number of targets: '+str(self.NUM_TARGETS))
        NUM_TARGETS = self.NUM_TARGETS
        GCNList = self.GCNList
        BINDING_TYPES = self.BINDING_TYPES
        GCNList_selected = GCNList[np.isin(BINDING_TYPES,BINDING_TYPE_FOR_GCN)]
        KC = KMeans(n_clusters=NUM_TARGETS, random_state=0).fit(GCNList_selected.reshape(-1, 1))
        KC_new = np.zeros(NUM_TARGETS, dtype='int')
        KC_new[0] = KC.labels_[np.argmin(GCNList_selected)]
        for i in range(NUM_TARGETS-1):
            KC_new[i+1] = KC.labels_[np.isin(KC.labels_, KC_new[0:i+1]) == False]\
            [np.argmin(GCNList_selected[np.isin(KC.labels_, KC_new[0:i+1]) == False])]
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

        GCNlabels = np.zeros(len(GCNList), dtype='int')
        GCNlabels[np.isin(BINDING_TYPES,BINDING_TYPE_FOR_GCN)] = KCclass
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
            plt.scatter(GCNList_selected, GCNList_selected, c=KC.labels_)
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
            plt.hist(GCNlabels[np.isin(BINDING_TYPES,BINDING_TYPE_FOR_GCN)], bins=np.arange(0.5, NUM_TARGETS+1.5), rwidth=0.5)
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
            plt.hist(GCNlabels[np.isin(BINDING_TYPES,BINDING_TYPE_FOR_GCN)], bins=np.arange(0.5, NUM_TARGETS+1.5), rwidth=0.5)
            plt.xticks(range(1, NUM_TARGETS+1))
            #ax.set_xticklabels(greater_than)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.xlabel('GCN Group', fontsize=8)
            plt.ylabel('DFT Samples', fontsize=8)
            plt.savefig('../Figures/GCN_Clustering.png', format='png')
            plt.close()

        self.GCNlabels = GCNlabels
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

    def _perturb_spectra(self, perturbations, X, y, a=0.999, b=1.001, BINDING_TYPES=None):
        X = X.copy(); y = y.copy()
        perturbed_values = (b-a)*np.random.random_sample((X.shape[0]*perturbations
                                                          , X.shape[1], X.shape[2]))+a
        Xperturbed = X.repeat(perturbations, axis=0)*perturbed_values
        yperturbed = y.repeat(perturbations, axis=0)
        if BINDING_TYPES is not None:
            BINDING_TYPES = BINDING_TYPES.copy()
            BINDING_TYPES_perturbed = BINDING_TYPES.repeat(perturbations,axis=0)
        else:
            BINDING_TYPES_perturbed = None
        return (Xperturbed, yperturbed, BINDING_TYPES_perturbed)
   
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

    def _coverage_shift(self, X, BINDING_TYPES, SELF_COVERAGE, TOTAL_COVERAGE):
        """
        COVERAGE: Relative spatial coverage of each binding-type
        TOTAL_COVERAGE: relative combined coverages of non-island
        """
        coverage_scaling_path = self.COV_SCALE_PATH
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
            CO_frequencies[BINDING_TYPES == i+1] = (CO_frequencies[BINDING_TYPES == i+1]\
                                           *(Coverage_Scaling['CO_FREQ'][i]['SELF_CO_PER_A2']*ABS_COVERAGE[BINDING_TYPES == i+1]\
                                             +Coverage_Scaling['CO_FREQ'][i]['CO_PER_A2']*TOTAL_COVERAGE_ABS[BINDING_TYPES == i+1]\
                                             +1))
            CO_intensities[BINDING_TYPES == i+1] = (CO_intensities[BINDING_TYPES == i+1]\
                                         *np.exp(Coverage_Scaling['CO_INT_EXP']*TOTAL_COVERAGE_ABS[BINDING_TYPES == i+1]))
            
            Xfrequencies[BINDING_TYPES == i+1] = (Xfrequencies[BINDING_TYPES == i+1]\
                                           *(Coverage_Scaling['PTCO_FREQ']['SELF_CO_PER_A2']*ABS_COVERAGE[BINDING_TYPES == i+1]\
                                             +Coverage_Scaling['PTCO_FREQ']['CO_PER_A2']*TOTAL_COVERAGE_ABS[BINDING_TYPES == i+1]\
                                             +1).reshape((-1, 1)))
            Xintensities[BINDING_TYPES == i+1] = (Xintensities[BINDING_TYPES == i+1]\
                                         *np.exp(Coverage_Scaling['PTCO_INT_EXP']*TOTAL_COVERAGE_ABS[BINDING_TYPES == i+1]).reshape((-1,1)))                          
        Xfrequencies[np.arange(X.shape[0]), CO_STRETCH_IDX] = CO_frequencies
        #Xfrequencies cannot be less than 0
        Xfrequencies[Xfrequencies[...]<2**-500] = 0
        Xintensities[np.arange(X.shape[0]), CO_STRETCH_IDX] = CO_intensities
        Xcov = np.stack((Xfrequencies,Xintensities),axis=1)
        return Xcov

    def scaling_factor_shift(self, X):
        Xfrequencies = X[:, 0].copy()
        Xintensities = X[:, 1].copy()
        CO_STRETCH_IDX = Xfrequencies > 1000
        MC_IDX = Xfrequencies < 1000
        #Scaling Factor determined from comparing experiment to DFT
        #uncertainties are 0.00097 and 0.00182 respectively
        
        SFCO = 1.0121
        SFMC = 0.969
        Xfrequencies[CO_STRETCH_IDX] = Xfrequencies[CO_STRETCH_IDX]*SFCO
        Xfrequencies[MC_IDX] = Xfrequencies[MC_IDX]*SFMC
        X = np.stack((Xfrequencies,Xintensities),axis=1)
        return X
    
    def _perturb_and_shift(self,perturbations, X, y, BINDING_TYPES=None):
        ADSORBATE = self.ADSORBATE
        X = X.copy()
        y = y.copy()
        Xperturbed = X.repeat(perturbations, axis=0)
        yperturbed = y.repeat(perturbations, axis=0)
        if BINDING_TYPES is not None:
            BINDING_TYPES = BINDING_TYPES.copy()
            BINDING_TYPES_perturbed = BINDING_TYPES.repeat(perturbations,axis=0)
        else:
            BINDING_TYPES_perturbed = None
            
        Xfrequencies = Xperturbed[:, 0].copy()
        Xintensities = Xperturbed[:, 1].copy()
        CO_STRETCH_IDX = Xfrequencies > 1000
        MC_IDX = Xfrequencies < 1000
        #Scaling Factor determined from comparing experiment to DFT
        #uncertainties are 0.00097 and 0.00182 respectively
        SFCO = 1.0121
        SFMC = 0.969
        if ADSORBATE == 'CO':
            perturb_CO_freq = 0.00097044 * np.random.standard_normal((Xfrequencies[CO_STRETCH_IDX].shape)) + SFCO
            perturb_MC_freq = 0.00183104 * np.random.standard_normal((Xfrequencies[MC_IDX].shape)) + SFMC
        else:
            perturb_CO_freq = 0.00097044 * np.random.standard_normal((Xfrequencies[CO_STRETCH_IDX].shape)) + 1
            perturb_MC_freq = 0.00183104 * np.random.standard_normal((Xfrequencies[MC_IDX].shape)) + 1
        
        perturb_CO_int = 0.00097044 * np.random.standard_normal((Xintensities[CO_STRETCH_IDX].shape)) + 1
        perturb_MC_int = 0.00183104 * np.random.standard_normal((Xintensities[MC_IDX].shape)) + 1
        
        Xfrequencies[CO_STRETCH_IDX] *= perturb_CO_freq
        Xfrequencies[MC_IDX] *= perturb_MC_freq
        
        Xintensities[CO_STRETCH_IDX] *= perturb_CO_int
        Xintensities[MC_IDX] *= perturb_MC_int
        
        Xperturbed_and_shifted = np.stack((Xfrequencies,Xintensities),axis=1)
        Xperturbed_and_shifted[Xperturbed_and_shifted[...]<2**-500] = 0
        return (Xperturbed_and_shifted, yperturbed, BINDING_TYPES_perturbed)

    def _generate_spectra(self, Xfrequencies, Xintensities, energies2D, prefactor, sigma):
        ENERGY_POINTS = self.ENERGY_POINTS
        int_mesh = np.zeros((Xfrequencies.shape[0], ENERGY_POINTS))
        #mesh everything on energy grids of 4 cm-1 spacings with a FWHM of 2
        for i in range(Xfrequencies.shape[0]):
            freq2D = np.tile(Xfrequencies[i], (ENERGY_POINTS, 1))
            int2D = np.tile(Xintensities[i], (ENERGY_POINTS, 1))
            temp = int2D*prefactor*np.exp(-(freq2D-energies2D)**2/(2.0*sigma**2))
            int_mesh[i] = np.sum(temp, axis=1)
        return int_mesh
    
    def _xyconv(self, X_sample, Y_sample, probabilities, BINDING_TYPES_sample):
        get_probabilities = self._get_probabilities
        mixed_lineshape = self._mixed_lineshape
        _coverage_shift = self._coverage_shift
        generate_spectra = self._generate_spectra
        NUM_TARGETS = self.NUM_TARGETS
        LOW_FREQUENCY = self.LOW_FREQUENCY
        HIGH_FREQUENCY = self.HIGH_FREQUENCY
        ENERGY_POINTS = self.ENERGY_POINTS
        COVERAGE = self.COVERAGE
        TARGET = self.TARGET
        GCNlabels = self.GCNlabels
        INCLUDED_BINDING_TYPES = self.INCLUDED_BINDING_TYPES
        MAX_COVERAGES = self.MAX_COVERAGES
        
        energies = np.linspace(LOW_FREQUENCY, HIGH_FREQUENCY, num=ENERGY_POINTS\
                               , endpoint=True)
        energy_spacing = energies[1]-energies[0]
        FWHM = 2*energy_spacing
        sigma = FWHM/(2.0 * np.sqrt(2.0 * np.log(2.)))
        prefactor = 1.0/(sigma * np.sqrt(2.0 * np.pi))
        energies2D = energies.reshape((-1, 1))
        Xfrequencies = X_sample[:, 0].copy()
        Xintensities = X_sample[:, 1].copy()
        np.random.shuffle(probabilities)
        num_samples = len(probabilities)
        num_simple_spectra = np.random.randint(1, high=201, size=num_samples)
        fLs = np.random.sample(num_samples)
        FWHMs = np.random.uniform(low=2, high=75, size=num_samples)
        Xconv = np.zeros((num_samples, ENERGY_POINTS))
        yconv = np.zeros((num_samples, NUM_TARGETS))
        y_mesh = np.zeros((Y_sample.size, NUM_TARGETS))
        #subtracting MIN_Y ensures that the number of indices in
        #ymin is equal to the labels, even if the labels don't start at 0
        if TARGET == 'GCN':
            MIN_Y = min(GCNlabels[GCNlabels>0])
        else:
            MIN_Y = min(INCLUDED_BINDING_TYPES)
        y_mesh[np.arange(Y_sample.size), Y_sample-MIN_Y] = 1
        sample_indices = np.arange(Y_sample.size)
        parray = np.zeros(Y_sample.size)
        shift_vector = np.zeros(Y_sample.size)                
        if TARGET == 'GCN' or COVERAGE == 'low' or type(COVERAGE) in [float,int]:
            int_mesh = generate_spectra(Xfrequencies, Xintensities\
                                        ,energies2D, prefactor, sigma)
        else:
            coverage_parray = get_probabilities(num_samples, 11)
            coverage_totals = np.random.random_sample(size=[num_samples,10])
            
        for i in range(num_samples):
            shift_vector[...]=0
            non_zeros = np.random.randint(low=1, high=Y_sample.size)
            indices_to_shift = np.random.choice(sample_indices,size=non_zeros,replace=False)
            shift_vector[indices_to_shift] = np.random.random_sample(size=non_zeros)
            for ii in range(NUM_TARGETS):
                ii_MIN_Y = ii+MIN_Y
                shift_vector_sum = shift_vector[Y_sample == ii_MIN_Y].sum()
                if shift_vector_sum >0:
                    shift_vector[Y_sample == ii_MIN_Y] /= shift_vector_sum
                    parray[Y_sample == ii_MIN_Y] = probabilities[i, ii] * shift_vector[Y_sample==ii_MIN_Y]       
                else:
                    parray[Y_sample == ii_MIN_Y] = probabilities[i, ii]
            parray /= np.sum(parray)
            indices_primary = np.random.choice(sample_indices, size=num_simple_spectra[i], replace=True, p=parray)
            if TARGET == 'GCN' or COVERAGE == 'low' or type(COVERAGE) in [float,int]:
                combined_mesh = np.sum(int_mesh[indices_primary], axis=0)
            else:
                #initialize coverages
                SELF_COVERAGE = np.random.random_sample(num_simple_spectra[i])
                TOTAL_COVERAGE = np.zeros_like(SELF_COVERAGE)
                COVERAGE_INDICES = np.random.choice([0,1,2,3,4,5,6,7,8,9,10], size=num_simple_spectra[i],replace=True, p=coverage_parray[i])
                #self coverage corresponding to index of 0 is island so it is single coverage and is skipped
                for ii in sorted(set(COVERAGE_INDICES.tolist()+[0]))[1:]:
                    TOTAL_COVERAGE[COVERAGE_INDICES == ii] = coverage_totals[i][ii-1]
                    #Set coverage of each spectra to be total coverage divided by the number of spectra being combined
                    SELF_COVERAGE[COVERAGE_INDICES == ii ] = coverage_totals[i][ii-1]/TOTAL_COVERAGE[COVERAGE_INDICES == ii].size
                    #update self coverage of indentical binding types to be the same (their sum)
                    for iii in INCLUDED_BINDING_TYPES:
                        SELF_COVERAGE[np.all((COVERAGE_INDICES == ii,BINDING_TYPES_sample[indices_primary]==iii),axis=0)] \
                        = np.sum(SELF_COVERAGE[np.all((COVERAGE_INDICES == ii,BINDING_TYPES_sample[indices_primary]==iii),axis=0)])
                #decrease maximum coverage at less favorable sites to improve prediction score
                for count, max_coverage in enumerate(MAX_COVERAGES):
                    SELF_COVERAGE[BINDING_TYPES_sample[indices_primary] == INCLUDED_BINDING_TYPES[count]] *= max_coverage
                #Ensure that Total coverage is compatible with self_coverage
                for ii in sorted(set(COVERAGE_INDICES.tolist()+[0]))[1:]:
                    coverage_factor = 0
                    for count, max_coverage in enumerate(MAX_COVERAGES):
                        coverage_factor += max_coverage * \
                        SELF_COVERAGE[np.all((COVERAGE_INDICES == ii,BINDING_TYPES_sample[indices_primary] == INCLUDED_BINDING_TYPES[count]), axis=0)].size
                    TOTAL_COVERAGE[COVERAGE_INDICES == ii] *= coverage_factor/TOTAL_COVERAGE[COVERAGE_INDICES == ii].size        
                TOTAL_COVERAGE[COVERAGE_INDICES==0] = SELF_COVERAGE[COVERAGE_INDICES==0]
                Xcov = _coverage_shift(X_sample[indices_primary], BINDING_TYPES_sample[indices_primary], SELF_COVERAGE, TOTAL_COVERAGE)
                Xcovfrequencies = Xcov[:, 0].copy()
                Xcovintensities = Xcov[:, 1].copy()
                int_mesh = generate_spectra(Xcovfrequencies,Xcovintensities\
                                                  ,energies2D, prefactor, sigma)
                for count, max_coverage in enumerate(MAX_COVERAGES):
                    int_mesh[BINDING_TYPES_sample[indices_primary] == INCLUDED_BINDING_TYPES[count]] *= max_coverage
                combined_mesh = np.sum(int_mesh, axis=0)
            yconv[i] = np.sum(y_mesh[indices_primary], axis=0, dtype='int')
            transform = mixed_lineshape(FWHMs[i], fLs[i], ENERGY_POINTS, energy_spacing)
            Xconv[i] = np.convolve(combined_mesh, transform, mode='valid')
        #This part is to adjust the tabulated contribution to match that of the adjusted spectra for max coverage of bridge, 3-fold and 4-fold
        if COVERAGE == 'high':
            if TARGET == 'binding_type':
                for count, max_coverage in enumerate(MAX_COVERAGES):
                    yconv[:,count] *= max_coverage
            if TARGET == 'combine_hollow_sites':
                for count, max_coverage in enumerate(MAX_COVERAGES[0:len(MAX_COVERAGES)-1]):
                    yconv[:,count] *= max_coverage
        #10**-4 accounts for noise in experimental spectra
        Xconv += 10**-4*np.max(Xconv, axis=1).reshape((-1, 1))*np.random.random_sample(Xconv.shape)
        #normalize so max X is 1 and make y a set of fractions that sum to 1
        Xconv /= np.max(Xconv, axis=1).reshape((-1, 1))
        yconv /= np.sum(yconv, axis=1).reshape((-1, 1))
        return (Xconv, yconv)

    def add_noise(self, Xconv_main, Xconv_noise, noise2signalmax=0.67):
        X_noisey = np.zeros((len(Xconv_main), len(Xconv_main[0])))
        noise_value = np.random.uniform(low=0, high=noise2signalmax, size=len(Xconv_main))
        noise_indices = np.arange(len(Xconv_noise))
        noise_sample = np.random.choice(noise_indices, size=len(Xconv_main), replace=True)
        for i in range(len(Xconv_main)):
            X_noisey[i] = Xconv_main[i] + Xconv_noise[noise_sample[i]]*noise_value[i]
        X_noisey = X_noisey/np.max(X_noisey, axis=1).reshape(-1, 1)
        return X_noisey

    def get_synthetic_spectra(self, NUM_SAMPLES, indices, COVERAGE=None, MAX_COVERAGES = [1,1,1,1]\
                              , LOW_FREQUENCY=200, HIGH_FREQUENCY=2200, ENERGY_POINTS=501):
        assert self.COVERAGE is None, "get_synthetic_spectra is intended \
        to be run only once. Please run get_more_spectra."
        assert type(COVERAGE) == float or COVERAGE==1 or COVERAGE \
        in ['low', 'high'], "Coverage should be a float, 'low', or 'high'."
        _coverage_shift = self._coverage_shift
        _get_probabilities = self._get_probabilities
        _perturb_spectra = self._perturb_spectra
        _perturb_and_shift = self._perturb_and_shift
        _xyconv = self. _xyconv
        high_coverage_path = self.HIGH_COV_PATH
        TARGET = self.TARGET
        NUM_TARGETS = self.NUM_TARGETS
        BINDING_TYPES = self.BINDING_TYPES
        BINDING_TYPES_with_4fold = self.BINDING_TYPES_with_4fold
        X0cov = self.X0cov
        GCNlabels = self.GCNlabels
        #Assign the target variable Y to either GCN group or binding site
        if TARGET == 'GCN':
            assert GCNlabels is not None, "get_GCNlabels must be executed before spectra can be generated"
            Y = GCNlabels
        else:
            Y = BINDING_TYPES
        #Adding Data for Extended Surfaces
        if COVERAGE == 'high' and TARGET == 'GCN':
            print('Adding high-coverage low index planes')
            print('Initial number of targets: '+str(NUM_TARGETS))
            with open(high_coverage_path, 'r') as infile:
                HC_Ext = json.load(infile)
            HC_CNPt = np.sort(np.array(list(set([np.min(i) for i in HC_Ext['CN_METAL']]))))
            NUM_TARGETS += len(HC_CNPt)
            HC_frequencies = []
            HC_intensities = []
            HC_classes = []
            max_freqs = np.max([len(i) for i in HC_Ext['FREQUENCIES']])
            for counter, i in enumerate(HC_CNPt):
                for ii in range(len(HC_Ext['CN_METAL'])):
                    if np.min(HC_Ext['CN_METAL'][ii]) == i:
                        HC_classes.append(np.max(Y)+counter+1)
                        num_atop = len(np.array(HC_Ext['CN_ADSORBATE'][ii]
                                               )[np.array(HC_Ext['CN_ADSORBATE'][ii]) == 1])
                        offset = max_freqs-len(HC_Ext['FREQUENCIES'][ii])
                        HC_frequencies.append(np.pad(HC_Ext['FREQUENCIES'][ii], (0, offset)
                                                     , 'constant', constant_values=0))
                        HC_intensities.append(np.pad(HC_Ext['INTENSITIES'][ii], (0, offset)
                                                     , 'constant', constant_values=0)/num_atop)
            HC_classes = np.array(HC_classes)
            #Change all high-coverage classes to just max(y)+1
            #until more accurate experiments and DFT are available
            NUM_TARGETS = NUM_TARGETS - len(HC_CNPt) + 1
            HC_classes[...] = NUM_TARGETS
            #
            HC_frequencies = np.array(HC_frequencies)
            HC_intensities = np.array(HC_intensities)
            HC_X = np.array([np.array((HC_frequencies[i], HC_intensities[i]))
                             for i in range(len(HC_frequencies))])
            offset = max_freqs-len(X0cov[0][0])
            X = np.pad(X0cov, ((0, 0), (0, 0), (0, offset)), 'constant', constant_values=0)
            print('Final number of targets: '+str(NUM_TARGETS))
        elif COVERAGE == 'high' and TARGET in ['binding_type', 'combine_hollow_sites']:
            print('testing all coverages')
            X = X0cov
        elif type(COVERAGE) == int or type(COVERAGE) == float:
            print('Relative coverage is ' + str(COVERAGE))
            X = _coverage_shift(X0cov, BINDING_TYPES_with_4fold, COVERAGE,COVERAGE)
        elif COVERAGE == 'low':
            X = X0cov
        X_new = X.copy()
        Y_new = Y.copy()
        if COVERAGE == 'high' and TARGET == 'GCN':
            num_single = Y_new.size
            #b = 2105/2095 #a = 1854/1865 - difference between experiments and DFT
            HC_X_expanded, HC_classes_expanded, _ = _perturb_spectra(5, HC_X, HC_classes
                                                                       , a=0.997, b=1.007)
            X_new = np.concatenate((X_new, HC_X_expanded), axis=0)
            Y_new = np.concatenate((Y_new, HC_classes_expanded), axis=0)
            indices = list(indices) + np.arange(Y_new.size)[num_single:].tolist()
        if len(set(Y_new[indices])) >1:
            indices_balanced, Y_balanced = RandomOverSampler().fit_sample(np.array(indices).reshape(-1,1),Y_new[indices].copy())
        else:
            indices_balanced = np.array(indices)
            Y_balanced = Y_new[indices]
        X_balanced = X_new[indices_balanced.flatten()]
        #BINDING_TYPES_balanced used only if COVERAGE == 'high' and (TARGET == 'binding_type' or TARGET == 'combine_hollow_sites')
        if COVERAGE == 'high' and TARGET in ['binding_type', 'combine_hollow_sites']:
            BINDING_TYPES_balanced = BINDING_TYPES_with_4fold[indices_balanced.flatten()]
        else:
            BINDING_TYPES_balanced = None
        #adding perturbations for improved fitting by account for frequency and intensity errors from DFT
        X_sample, Y_sample, BINDING_TYPES_sample = _perturb_and_shift(5, X_balanced, Y_balanced, BINDING_TYPES_balanced)
        probabilities = _get_probabilities(NUM_SAMPLES, NUM_TARGETS)
        if COVERAGE == 'high' and TARGET == 'GCN':
            self.HC_X = HC_X
            self.HC_classes = HC_classes
        self.X = X
        self.Y = Y
        self.NUM_TARGETS = NUM_TARGETS
        self.X0cov = X0cov
        self.COVERAGE = COVERAGE
        self.LOW_FREQUENCY = LOW_FREQUENCY
        self.HIGH_FREQUENCY = HIGH_FREQUENCY
        self.ENERGY_POINTS = ENERGY_POINTS
        self.MAX_COVERAGES = MAX_COVERAGES
        Xconv, yconv = _xyconv(X_sample, Y_sample, probabilities, BINDING_TYPES_sample)
        #set numbers that may form denormals to zero to improve numerics
        Xconv[Xconv[...]<2**-500] = 0
        yconv[yconv[...]<2**-500] = 0
        return (Xconv, yconv)

    def get_more_spectra(self, NUM_SAMPLES, indices):
        _perturb_spectra = self._perturb_spectra
        _perturb_and_shift = self._perturb_and_shift
        _get_probabilities = self._get_probabilities
        _xyconv = self. _xyconv
        NUM_TARGETS = self.NUM_TARGETS
        X_new = self.X.copy()
        Y_new = self.Y.copy()
        TARGET = self.TARGET
        BINDING_TYPES_with_4fold = self.BINDING_TYPES_with_4fold
        COVERAGE = self.COVERAGE
        if COVERAGE == 'high' and TARGET == 'GCN':
            HC_X = self.HC_X
            HC_classes = self.HC_classes
            num_single = Y_new.size
            #b = 2105/2095 #a = 1854/1865 difference between experiments and DFT
            HC_X_expanded, HC_classes_expanded, _ = _perturb_spectra(5, HC_X, HC_classes
                                                                       , a=0.997, b=1.07)
            X_new = np.concatenate((X_new, HC_X_expanded), axis=0)
            Y_new = np.concatenate((Y_new, HC_classes_expanded), axis=0)
            indices = list(indices) + np.arange(Y_new.size)[num_single:].tolist() 
       
        if len(set(Y_new[indices])) >1:
            indices_balanced, Y_balanced = RandomOverSampler().fit_sample(np.array(indices).reshape(-1,1),Y_new[indices].copy())
        else:
            indices_balanced = np.array(indices)
            Y_balanced = Y_new[indices]
        X_balanced = X_new[indices_balanced.flatten()]
        #only use BINDING_TYPES_balanced if COVERAGE == 'high' and (TARGET == 'binding_type' or TARGET == 'combine_hollow_sites')
        if COVERAGE == 'high' and TARGET in ['binding_type', 'combine_hollow_sites']:
            BINDING_TYPES_balanced = BINDING_TYPES_with_4fold[indices_balanced.flatten()]
        else:
            BINDING_TYPES_balanced = None
        #adding perturbations for improved fitting by account for frequency and intensity errors from DFT
        X_sample, Y_sample, BINDING_TYPES_sample = _perturb_and_shift(5, X_balanced, Y_balanced, BINDING_TYPES_balanced)
        probabilities = _get_probabilities(NUM_SAMPLES, NUM_TARGETS)
        Xconv, yconv = _xyconv(X_sample, Y_sample, probabilities, BINDING_TYPES_sample)
        #set numbers that may form denormals to zero to improve numerics
        Xconv[Xconv[...]<2**-500] = 0
        yconv[yconv[...]<2**-500] = 0
        return (Xconv, yconv)
    
def fold(frequencies, intensities, LOW_FREQUENCY, HIGH_FREQUENCY, ENERGY_POINTS,FWHM):
    energies = np.linspace(LOW_FREQUENCY, HIGH_FREQUENCY, num=ENERGY_POINTS, endpoint=True)
    energy_spacing = energies[1]-energies[0]
    if FWHM < 2*energy_spacing:
        raise ValueError('Function input FWHM must must be at least twice\
        the energy spacing to prevent information loss. It therefore must be\
        at least' + str(2*energy_spacing))
    assert HIGH_FREQUENCY > LOW_FREQUENCY, "The high frequency must be greater than the low frequenc"
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

def HREEL_2_scaledIR(HREEL, frequency_range=None ):
    if frequency_range is None:
        frequency_range = np.linspace(200,2200,num=501,endpoint=True)
    PEAK_CONV = 2.7
    IR = np.interp(frequency_range, HREEL[0], HREEL[1]*HREEL[0]**PEAK_CONV, left=None, right=None, period=None)
    IR_scaled = IR/np.max(IR)
    return IR_scaled