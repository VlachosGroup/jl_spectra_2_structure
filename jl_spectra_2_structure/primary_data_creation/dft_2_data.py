# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:14:47 2017

@author: lansford
"""

from __future__ import absolute_import, division, print_function
import os
from copy import deepcopy
import itertools
import json
import numpy as np
import pandas as pd
from pandas import read_csv
from ase.io import read
import statsmodels.api as sm
from scipy.optimize import curve_fit
from matplotlib import rcParams
from matplotlib import pyplot as plt
from ..vibrations import Infrared
from ..error_metrics import get_r2
from .coordination import Coordination
from .file_parser import VASP_PARSER
from .file_parser import explode


class Primary_DATA:
    """
    Class that generates primary data (frequencies, intensities, GCN values)
    """
    def __init__(self,metal_atoms=['Pt'], adsorbate_atoms=['C','O']\
                 , create_new_vasp_files=False, delta=0.025):
        """
        Parameters
        ----------
        metal_atoms : list of str
        	List of atom types considered part of the surface, bulk
            nanoparticle
            
        adsorbate_atoms : list of str
        	List of atom types onsidered adsorbates. The first atom type
            listed is considered in contact with the surface.
            
        create_new_vasp_files : bool
        	If True, charg2.extxyzstripped and vasprun2.xmlstripped files
            are generated which compile concatenated DDEC6 charge and
            vasprun.xml files into a format readable by PANDAS or ASE
            
        delta : float
        	Angstroms atoms were displaced in finite difference calculation
            for parameterizing the Hessian
        	
        Attributes
        ----------
        METAL_ATOMS : list of str
        	List of atom types considered part of the surface, bulk
            nanoparticle
            
        ADSORBATE_ATOMS : list of str
        	List of atom types onsidered adsorbates. The first atom type
            listed is considered in contact with the surface.
            
        CREATE_NEW_VASP_FILES : bool
        	If True, charg2.extxyzstripped and vasprun2.xmlstripped files
            are generated which compile concatenated DDEC6 charge and
            vasprun.xml files into a format readable by PANDAS or ASE
            
        DELTA : float
        	Angstroms atoms were displaced in finite difference calculation
            for parameterizing the Hessian
        """
        self.METAL_ATOMS = metal_atoms
        self.ADSORBATE_ATOMS = adsorbate_atoms
        self.DELTA = delta
        self.CREATE_NEW_VASP_FILES = create_new_vasp_files
    def generate_primary_data(self, vasp_directory, output_path\
                                ,data_type='nanoparticle'\
                                , num_adsorbates='single', poc=1):
        """Generate primary data

        Parameters
        ----------
        vasp_directory : str
        	Directory where vasp files are stored.
            
        output_path : str
        	Output file location of json that will stroe primary data
            
        data_type : str
        	Indicates the kind of DFT data that is being used. Can be
            'nanoparticle' or 'surface'.
            
        num_adsorbates : str
        	Indicates the number of adsorbates that were considered in the
            simulation. Can be 'single' or 'multiple'
            
        poc : int
        	The number of points of contact each adsorbate has with the surface.
        	
        Attributes
        ----------
        OUTPUT_DICTIONARY : dict
        	Dictionary of primary data
            
        FREQ_FILES : list of str
            List of processed concatenated vasprun.xml files
            
        CHARGE_FILES : list of str
            list of processed concatenated DECC6 charge files
        
        INDICES_USED : numpy.ndarray
            indices of CHARGE_FILES and FREQ_FILES whose data is used to generate primary
            data written to the OUTPUT_DICTIONARY
            
        MOLECULES : list of Atoms
            List of ASE atoms object that represent the local minima used as
            the initial configuration for each normal mode analysis
                
        """
        metal_atoms = self.METAL_ATOMS
        adsorbate_atoms = self.ADSORBATE_ATOMS
        delta = self.DELTA
        create_new_vasp_files = self.CREATE_NEW_VASP_FILES
        assert data_type in ['nanoparticle','surface'], "data type not 'nanoparticle' or 'slab'"
        assert num_adsorbates in ['single','multiple'], "data type not 'nanoparticle' or 'slab'"
        VASP_FILES = VASP_PARSER(vasp_directory, create_new_vasp_files)
        #Get files that store forces (freq_files) and charges (charge_files)
        freq_files = VASP_FILES.get_freq_files()
        charge_files = VASP_FILES.get_charge_files()
        
        #Initialize dictionary that will be saved as json
        VIB_DICT = {'FREQUENCIES' : [], 'IMAGINARY' : [], 'INTENSITIES' : []\
                    ,'MAX_FORCE' : [], 'CN_ADSORBATE' : [], 'GCN' : []\
                    ,'CN_METAL' : [], 'NUM_METAL' : [], 'COVERAGE':[]\
                    , 'ENERGY' : []}
        #countlist is used only for testing purposes
        count_list = []
        molecule_list = []
        #iterate over each force file and charge file simultaneously
        for count, (freq_file, charge_file) in enumerate(zip(freq_files, charge_files)):
            print(count)
            #read in vasp force calculations as ASE atoms object
            molecule_images = read(freq_file,':')
            #Get the unit cell dimensions
            unit_cell_length = np.diagonal(molecule_images[0].get_cell())
            #if the unit cell is not rectangular then wrap-around in one direction can cause wrap-around in another
            unit_cell = molecule_images[0].get_cell()
            #iterate over all frequency calculations not including the equilibrium
            for molecule in molecule_images[1:]:
                #initialize positions that have wrap-around removed
                molecule.unwrapped_positions = molecule.get_positions()
                if data_type == 'surface':
                    #This part of the code undoes wrap-around so that proper changes in dipole moments of the unit cell can be calculated
                    #wrap-around involves movment of an atom back inside the unit cell if it goes beyond the periodic boundaries
                    #repreat 3 times which can be necessary for parallelpipeds
                    for _ in range(3):
                        displacement = molecule.unwrapped_positions - molecule_images[0].get_positions()
                        wrapped_positive = displacement > 4/5*unit_cell_length
                        wrapped_negative = displacement < -4/5*unit_cell_length
                        #iteratire over all atoms
                        for i in range(len(molecule_images[0])):
                            #iterate over each carteian direction
                            for ii in range(3):
                                #if new position of the atom is much greater than the old position it was wrapped from the left to right
                                if wrapped_positive[i][ii] == True:    
                                    molecule.unwrapped_positions[i] -= unit_cell[ii]
                                #if new position of the atom is much less than the old position it was wrapped from right to left
                                elif wrapped_negative[i][ii] == True:
                                    molecule.unwrapped_positions[i] += unit_cell[ii]
            #set unwrapped positions of the equilibrium structure to the actual positions
            molecule_images[0].unwrapped_positions = molecule_images[0].get_positions()
            #get number of displacments (6n)
            num_disp = len(molecule_images)-1
            #Get carteisan coordinate (flattend positions)  of atom that moved
            moved_positions = np.array([np.argmax(abs(i.unwrapped_positions\
            -molecule_images[0].get_positions())) for i in molecule_images[1:]])
            #get the amount each atom was displaced
            amount_moved = np.round([molecule_images[i+1].unwrapped_positions.flatten()\
            [moved_positions[i]]-molecule_images[0].get_positions().flatten()\
            [moved_positions[i]] for i in np.arange(0, num_disp)],decimals=3)
            #check to see if the amount moved was delta A
            moved_25 =  np.isclose(np.abs(amount_moved),delta) == True
            #check to see that each atom was displaced once in each direction
            moved_unique = len(amount_moved*moved_positions) == len(set(amount_moved*moved_positions))
            #get the atom that was moved in each case
            atom_positions = np.array(np.floor(moved_positions/3),dtype='int')
            #ensure that the atoms moved where those desired
            atom_symbols = np.array([molecule_images[i+1][atom_positions[i]].symbol in adsorbate_atoms for i in range(num_disp)])
            #Get indices of carbon atoms
            Catoms = [atom.index for atom in molecule_images[0] if atom.symbol == adsorbate_atoms[0]]
            adsorbate_indices = [atom.index for atom in molecule_images[0] if atom.symbol in adsorbate_atoms]
            num_adsorbate_atoms = len(adsorbate_indices)
            #Get indices of oxygen atoms
            #Oatoms = [atom.index for atom in molecule_images[0] if atom.symbol == adsorbate_atoms[1]]
            if num_adsorbates == 'single':
                adsorbates_match = len(molecule_images) == num_adsorbate_atoms*6+1
            else:
                adsorbates_match=True
            #Get all forces on the carbon and oxygen atoms and ensure they are less than 0.05 eV/A
            All_forces = np.max(np.sum((molecule_images[0].get_forces()[adsorbate_indices])**2,axis=-1)**0.5)
            max_force = np.max(All_forces)
            #Ensure a valid frequency calculation was done before moving on
            if np.all([moved_25.all(),moved_unique,adsorbates_match,atom_symbols.all()]) == True:
                #read in chargemole file to get patial charges and atomic dipoles as a dataframe
                chargemol = read_csv(charge_file,header=0,sep=',',index_col=False,usecols=range(0,15))
                #Get number of atoms in the chargemol file
                numcatoms = max(chargemol.atomnumber)
                #Get number of atoms in the VASP file
                numatoms = len(molecule_images[0])
                #Ensure number of atoms in the chargemol file equals that in the VASP file
                if numatoms == numcatoms:
                    #Get the number of total atoms in the charge file (should be (num_dsip+1)*numcatoms)
                    chargelen = len(chargemol.atomnumber)
                    #Generate a list of dataframes where each dataframe is the charges of a VASP atoms object (corresponds to single point calculations)
                    chargelist = [chargemol[i*numatoms:(i+1)*numatoms] for i in range(0,int(chargelen/numatoms))]
                    #Generate a copy of the original VASP image file
                    catoms = deepcopy(molecule_images[0])
                    #set positions of chargemole atoms obect to those of the chargemol dataframe
                    catoms.set_positions(np.array(chargelist[0][['x','y','z']]),apply_constraint=False)
                    #get the largest difference between positions in the VASP file and the chargemol (related to VASP precision)
                    maxdif = np.max(abs(molecule_images[0].get_distances(0,range(0,numatoms)\
                    ,mic=True,vector=False) - catoms.get_distances(0,range(0,numatoms),mic=True,vector=False)))
                    #only continue if they match
                    if maxdif < 10**-4:
                        #conversion factor from atomic units to electron Angstroms
                        atomicU2eA =  8.47836*10**-30/(1.602188*10**-19)*10**10
                        for i in range(0,num_disp+1):
                            #initialize dipole moments of each atom
                            dipolemoments = np.zeros((numatoms,3))
                            #get atomic dipoles
                            atomicdipoles = chargelist[i][['dipole_x','dipole_y','dipole_z']]
                            #combine atomic dipoles with partial charges
                            dipolemoments = atomicdipoles*atomicU2eA + molecule_images[i].unwrapped_positions\
                            *np.array([chargelist[i]['net_charge']]).T
                            #get dipole moment of th unit cell
                            molecule_images[i].dipole = np.array(np.sum(dipolemoments,axis=0))
                            #set charges of the atoms
                            molecule_images[i].charges = np.array(chargelist[i]['net_charge'])
                        
                        if data_type == 'surface':
                            Pt_indices = np.isin(molecule_images[0].get_chemical_symbols(), metal_atoms)
                            Z_distances = np.array([molecule.z for molecule in molecule_images[0]])
                            is_Surface = np.all((Z_distances>=np.max(Z_distances[Pt_indices])-1,Pt_indices),axis=0)
                            surface_atoms = len(Pt_indices[is_Surface])
                        #If the dataset does not have enouhgh surface atoms expand it
                        if data_type == 'surface' and surface_atoms <16:
                            repeated_images = molecule_images[0].repeat((2,2,1))
                        else:
                            repeated_images = molecule_images[0]
                        #remove adsorbate atoms from including in generalized coordination number calculations
                        exclude=[atom.index for atom in repeated_images if atom.symbol not in metal_atoms]
                        #Initialize Coordination object. Repeat is necessary so it doesn't count itself
                        CN = Coordination(repeated_images,exclude=exclude,cutoff=1.25)
                        #Get coordination numbers
                        CN.get_coordination_numbers()
                        #The Coordination number of the site is TotalNN
                        TotalNN = []
                        #Get list of Carbon atom coordination
                        CN_CO = [CN.cn[C] for C in Catoms]
                        #Get list of GCN values for adsorption site
                        GCN = [CN.get_gcn(CN.bonded[i]) for i in Catoms]
                        for C in Catoms:
                            TotalC = []
                            for iii in CN.bonded[C]:
                                TotalC = TotalC + CN.bonded[iii]
                            TotalNN.append(len(set(TotalC)))
                        #Get number of Pt atoms
                        NumPt = 0
                        for atom in molecule_images[0]:
                            if atom.symbol in metal_atoms:
                                NumPt += 1
                        
                        #Get Pt atoms on the surface
                        if data_type == 'nanoparticle':
                            surface_atoms = len([i for i in CN.cn if i <12]) - num_adsorbate_atoms
                        NUM_CO = len(Catoms)
                        #set values of dictionary
                        #Generate an infrared object
                        if data_type == 'surface':
                            infrared = Infrared(molecule_images,delta=delta, nfree=2, directions=2)
                        else:
                            infrared = Infrared(molecule_images,delta=delta, nfree=2)
                        #Get frequencies and remove imaginary parts
                        FREQUENCIES = infrared.get_frequencies().real
                        #Get intensities of corresponding to frequenices
                        VIB_DICT['INTENSITIES'].append(infrared.intensities)
                        #Set intensities of imaginary frequencies to 0
                        VIB_DICT['INTENSITIES'][-1][FREQUENCIES == 0] = 0
                        #convert arrays to lists to store in json
                        VIB_DICT['INTENSITIES'][-1] = VIB_DICT['INTENSITIES'][-1].tolist()
                        VIB_DICT['FREQUENCIES'].append(FREQUENCIES.real.tolist())
                        VIB_DICT['IMAGINARY'].append(FREQUENCIES.imag.tolist())
                        VIB_DICT['NUM_METAL'].append(NumPt)
                        VIB_DICT['ENERGY'].append(molecule_images[0].get_potential_energy())
                        VIB_DICT['COVERAGE'].append(NUM_CO/surface_atoms)
                        VIB_DICT['MAX_FORCE'].append(max_force)
                        if num_adsorbates == 'single' and poc==1:
                            VIB_DICT['GCN'].append(GCN[0])
                            VIB_DICT['CN_ADSORBATE'].append(CN_CO[0])
                            VIB_DICT['CN_METAL'].append(TotalNN[0])
                        else:
                            VIB_DICT['GCN'].append(GCN)
                            VIB_DICT['CN_ADSORBATE'].append(CN_CO)
                            VIB_DICT['CN_METAL'].append(TotalNN)
                        if poc > 1:
                            VIB_DICT['CN_ADSORBATE'][-1] = np.unique([CN.bonded[C] for C in Catoms]).size
                            VIB_DICT['GCN'][-1] = np.mean(GCN)
                            VIB_DICT['CN_METAL'][-1] = np.mean(TotalNN)
                        count_list.append(count)
                        molecule_list.append(molecule_images[0])
        
        with open(output_path, 'w') as outfile:
            json.dump(VIB_DICT, outfile, sort_keys=True, indent=1)
        self.OUTPUT_DICTIONARY = VIB_DICT
        self.FREQ_FILES = freq_files
        self.CHARGE_FILES = charge_files
        self.INDICES_USED = np.array(count_list)
        self.MOLECULES = molecule_list
            
    def generate_isotope_data(self, vasp_directory, output_file\
                                       ,masses1=[12,16], masses2=[24,32]):
        
        """Generate isotope data used in frequency scaling

        Parameters
        ----------
        vasp_directory : str
        	Directory where vasp files are stored.
            
        output_path : str
        	Output file location of json that will stroe isotope studies
            
        masses1 : list of int
        	List of masses for 'adsorbate_atoms' for one isotope set
            
        masses2 : list of int
        	List of masses for 'adsorbate_atoms' for the other isotope set
        	
        Attributes
        ----------
        OUTPUT_DICTIONARY : dict
        	Dictionary of primary data
            
        FREQ_FILES : list of str
            List of preprocessed concatenated vasprun.xml files
            
        CHARGE_FILES : list of str
            list of preprocessed concatenated DECC6 charge files
        
        INDICES_USED : numpy.ndarray
            indices of CHARGE_FILES and FREQ_FILES whose data is used to generate primary
            data written to the OUTPUT_DICTIONARY
            
        MOLECULES : list of Atoms
            List of ASE atoms object that represent the local minima used as
            the initial configuration for each normal mode analysis
                
        Notes
        -----
        The following data is written for the isotopic analysis used in
        determining coverage scaling.
        
        FREQUENCIES: frequencies of normal modes
        INTENSITIES: intensities of normal modes
        CO_CN_CO: number of CO to which each CO is coordinated
        MIN_SEPARATION: smallest distance (A) between each CO
        CN_CO: number of Pt to which each CO is coordinated
        GCN: generalized coordination number of each adsorption site
        CN_PT: coordination of each adsorption site
        NUM_PT: Number of Pt atoms in each slab
        SURFACE_AREA: surface area in each slab (A2)
        CO_PER_A2: CO per surface area (A2)
        NUM_CO: number of CO on the surface
        SURFACE_PT: Number of Pt atoms on the adsorbate surface
        NUM_C12O16: Number of C12O16 on the surface
        COVERAGE: CO molecules/surface Pt atoms
        SELF_COVERAGE: major contributing CO molecules of the normal mode per surface Pt atom
        NUM_C12O32: number of C12O32 molecules
        NUM_C24O16: number of C24O16 molecules
        NUM_C24O32: number of C24O32 molecules
        SELF_CO_PER_A2: major contributing CO moldecules of the normal mode per A2
        MODE_ID: ID of each mode
        REDUCED_MASS: simple reduced mass of each molecule
        REDUCED_MASS_NORMED: normalized reduced mass of each normal mode
        MIXED_FACTOR: contribution to each normal from secondary contributing CO isotopes
        """
        metal_atoms = self.METAL_ATOMS
        adsorbate_atoms = self.ADSORBATE_ATOMS
        delta = self.DELTA
        create_new_vasp_files = self.CREATE_NEW_VASP_FILES
        #convert these strings to numbers
        C12 = masses1[0]
        C24 = masses2[0]
        O16 = masses1[1]
        O32 = masses2[1]
        
        #Get all directories that contain results of VASP vibrational calculations
        VASP_FILES = VASP_PARSER(vasp_directory, create_new_vasp_files)
        #Get files that store forces (freq_files) and charges (charge_files)
        freq_files = VASP_FILES.get_freq_files()
        charge_files = VASP_FILES.get_charge_files()
        #Initialize dictionary that will be saved as json
        VIB_DICT = {'FREQUENCIES':[],'INTENSITIES':[],'CO_CN_CO':[],'MIN_SEPARATION':[]\
                ,'CN_CO':[],'GCN':[],'CN_PT':[],'NUM_PT':[],'SURFACE_AREA':[],'CO_PER_A2':[]\
                ,'NUM_CO':[],'SURFACE_PT':[], 'NUM_C12O16':[],'COVERAGE':[],'SELF_COVERAGE':[]\
                ,'NUM_C12O32':[],'NUM_C24O16':[],'NUM_C24O32':[],'SELF_CO_PER_A2':[]\
                ,'MODE_ID':[],'REDUCED_MASS':[],'REDUCED_MASS_NORMED':[],'MIXED_FACTOR':[]\
                   }
        #countlist is used only for testing purposes
        count_list = []
        #iterate over each force file and charge file simultaneously
        for count, (freq_file, charge_file) in enumerate(zip(freq_files, charge_files)):
            print(count)
            #read in vasp force calculations as ASE atoms object
            molecule_images = read(freq_file,':')
            #This part of the code undoes wrap-around so that proper dipole moments of the unit cell can be calculated
            #wrap-around involves movment of an atom back inside the unit cell if it goes beyond the periodic boundaries
            #Get the unit cell dimensions
            unit_cell_diag = np.diagonal(molecule_images[0].get_cell())
            #if the unit cell is not rectangular then wrap-around in one direction can cause wrap-around in another
            unit_cell = molecule_images[0].get_cell()
            #iterate over all frequency calculations not including the equilibrium
            for molecule in molecule_images[1:]:
                #initialize positions that have wrap-around removed
                molecule.unwrapped_positions = molecule.get_positions()
                #repreat 3 times which can be necessary for parallelpipeds
                for _ in range(3):
                    displacement = molecule.unwrapped_positions - molecule_images[0].get_positions()
                    wrapped_positive = displacement > 4/5*unit_cell_diag
                    wrapped_negative = displacement < -4/5*unit_cell_diag
                    #iteratire overa all atoms
                    for i in range(len(molecule_images[0])):
                        #iterate over each carteian direction
                        for ii in range(3):
                            #if new position of the atom is much greater than the old position it was wrapped from the left to right
                            if wrapped_positive[i][ii] == True:    
                                molecule.unwrapped_positions[i] -= unit_cell[ii]
                            #if new position of the atom is much less than the old position it was wrapped from right to left
                            elif wrapped_negative[i][ii] == True:
                                molecule.unwrapped_positions[i] += unit_cell[ii]
            #set unwrapped positions of the equilibrium structure to the actual positions
            molecule_images[0].unwrapped_positions = molecule_images[0].get_positions()
            #get number of displacments (6n)
            num_disp = len(molecule_images)-1
            #Get carteisan coordinate (flattend positions)  of atom that moved
            moved_positions = np.array([np.argmax(abs(i.unwrapped_positions-molecule_images[0].get_positions())) for i in molecule_images[1:]])
            #get the amount each atom was displaced
            amount_moved = np.round([molecule_images[i+1].unwrapped_positions.flatten()[moved_positions[i]]-molecule_images[0].get_positions().flatten()[moved_positions[i]] for i in np.arange(0, num_disp)],decimals=3)
            #check to see if the amount moved was delta A
            moved_25 =  np.isclose(np.abs(amount_moved),delta) == True
            #check to see that each atom was displaced once in each direction
            moved_unique = len(amount_moved*moved_positions) == len(set(amount_moved*moved_positions))
            #get the atom that was moved in each case
            atom_positions = np.array(np.floor(moved_positions/3),dtype='int')
            #ensure that the atoms moved where those desired
            atom_symbols = np.array([molecule_images[i+1][atom_positions[i]].symbol in adsorbate_atoms for i in range(num_disp)])
            #Get indices of carbon atoms
            Catoms = [atom.index for atom in molecule_images[0] if atom.symbol in adsorbate_atoms[0]]
            #Get indices of oxygen atoms
            Oatoms = [atom.index for atom in molecule_images[0] if atom.symbol in adsorbate_atoms[1]]
            #Get all forces on the carbon and oxygen atoms and ensure they are less than 0.05 eV/A
            Cforce = np.max(np.sum((molecule_images[0].get_forces()[Catoms])**2,axis=-1)**0.5)
            Oforce = np.max(np.sum((molecule_images[0].get_forces()[Oatoms])**2,axis=-1)**0.5)
            COforce_small = np.max((Cforce,Oforce)) < 0.05
            #Ensure a valid frequency calculation was done before moving on
            if np.all([moved_25.all(),moved_unique,atom_symbols.all(),COforce_small]) ==True:
                #read in chargemole file to get patial charges and atomic dipoles as a dataframe
                chargemol = read_csv(charge_file,header=0,sep=',',index_col=False,usecols=range(0,15))
                #Get number of atoms in the chargemol file
                numcatoms = max(chargemol.atomnumber)
                #Get number of atoms in the VASP file
                numatoms = len(molecule_images[0])
                #Ensure number of atoms in the chargemol file equals that in the VASP file
                if numatoms == numcatoms:
                    #Get the number of total atoms in the charge file (should be (num_dsip+1)*numcatoms)
                    chargelen = len(chargemol.atomnumber)
                    #Generate a list of dataframes where each dataframe is the charges of a VASP atoms object (corresponds to single point calculations)
                    chargelist = [chargemol[i*numatoms:(i+1)*numatoms] for i in range(0,int(chargelen/numatoms))]
                    #Generate a copy of the original VASP image file
                    catoms = deepcopy(molecule_images[0])
                    #set positions of chargemole atoms obect to those of the chargemol dataframe
                    catoms.set_positions(np.array(chargelist[0][['x','y','z']]),apply_constraint=False)
                    #get the largest difference between positions in the VASP file and the chargemol (related to VASP precision)
                    maxdif = np.max(abs(molecule_images[0].get_distances(0,range(0,numatoms),mic=True,vector=False) - catoms.get_distances(0,range(0,numatoms),mic=True,vector=False)))
                    #only continue if they match
                    if maxdif < 10**-4:
                        #conversion factor from atomic units to electron Angstroms
                        atomicU2eA =  8.47836*10**-30/(1.602188*10**-19)*10**10
                        for i in range(0,num_disp+1):
                            #initialize dipole moments of each atom
                            dipolemoments = np.zeros((numatoms,3))
                            #get atomic dipoles
                            atomicdipoles = chargelist[i][['dipole_x','dipole_y','dipole_z']]
                            #combine atomic dipoles with partial charges
                            dipolemoments = atomicdipoles*atomicU2eA + molecule_images[i].unwrapped_positions*np.array([chargelist[i]['net_charge']]).T
                            #get dipole moment of th unit cell
                            molecule_images[i].dipole = np.array(np.sum(dipolemoments,axis=0))
                            #set charges of the atoms
                            molecule_images[i].charges = np.array(chargelist[i]['net_charge'])
                        
                        #Get Pt atoms on the surface
                        Pt_indices = np.isin(molecule_images[0].get_chemical_symbols(), metal_atoms)
                        Z_distances = np.array([molecule.z for molecule in molecule_images[0]])
                        is_Surface = np.all((Z_distances>=np.max(Z_distances[Pt_indices])-1,Pt_indices),axis=0)
                        surface_atoms = len(Pt_indices[is_Surface])
                        NUM_CO = len(Catoms)
                        #If the dataset does not have enouhgh surface atoms expand it
                        if surface_atoms <16:
                            repeated_images = molecule_images[0].repeat((2,2,1))
                        else:
                            #This must be repeated so we can get the minimum separation distance for the case of one adsorbate
                            repeated_images = molecule_images[0].repeat((2,1,1))
                        #remove adsorbate atoms from including in generalized coordination number calculations
                        exclude=[atom.index for atom in repeated_images if atom.symbol not in metal_atoms]
                        #Initialize Coordination object. Repeat is necessary so it doesn't count itself
                        CN = Coordination(repeated_images,exclude=exclude,cutoff=1.25)
                        #Get coordination numbers
                        CN.get_coordination_numbers()
                        #Get list of Carbon atom coordination
                        CN_CO = [CN.cn[C] for C in Catoms]
                        #Get list of GCN values for adsorption site
                        GCN = [CN.get_gcn(CN.bonded[i]) for i in Catoms]
                        #filter out data where more than one binding-type is occupied
                        if len(set(GCN)) == 1 and len(set(CN_CO)) == 1:
                            #exlude all non-carbon atoms
                            exclude_notC=[atom.index for atom in repeated_images if atom.symbol not in adsorbate_atoms[0]]
                            #Get Distances
                            Catoms2 = [atom.index for atom in repeated_images if atom.symbol in adsorbate_atoms[0]]
                            C_Distances = []
                            for C in Catoms:
                                distance_list = [atom for atom in Catoms2 if atom != C]
                                C_Distances.append(repeated_images.get_distances(C,distance_list,mic=True))
                            #Get minimum separation between CO molecule
                            MIN_SEPARATION = np.min(C_Distances)
                            #Use minimum separation as cutoff radius to get nearest neighbor CO molecules
                            CO_CN = Coordination(repeated_images,exclude=exclude_notC,cutoff=1.1*MIN_SEPARATION,cutoff_type = 'absolute')
                            #Get nearest neighbor CO molcules to ensure ordered overlayers
                            CO_CN.get_coordination_numbers()
                            CO_CN_CO = [CO_CN.cn[C] for C in Catoms]            
                            #Check to see that all CO have an equal number of nearest neighbors
                            if len(set(CO_CN_CO)) == 1:
                                #The Coordination number of the site is TotalNN
                                TotalNN = []
                                for C in Catoms:
                                    TotalC = []
                                    for iii in CN.bonded[C]:
                                        TotalC = TotalC + CN.bonded[iii]
                                    TotalNN.append(len(set(TotalC)))
                                #Get number of Pt atoms
                                NumPt = 0
                                for atom in molecule_images[0]:
                                    if atom.symbol in metal_atoms:
                                        NumPt += 1
                                
                                
                                #CList and Olist are a list of Carbon and Oxygen atoms to have their isotopes changed
                                #empty list allows original masses to be kept
                                CList = [[]]
                                OList = [[]]
                                for numCO2change in range(1,NUM_CO + 1):
                                    for C_index in itertools.combinations(Catoms,numCO2change):
                                        CList.append(list(C_index))
                                    for O_index in itertools.combinations(Oatoms,numCO2change):
                                        OList.append(list(O_index))
                                #Get the joint lists of carbon and oxygen atoms to have their isotopes to change
                                COVERAGE = NUM_CO/surface_atoms
                                SURFACE_AREA = unit_cell_diag[0]*unit_cell_diag[1]
                                CO_PER_A2 = NUM_CO/SURFACE_AREA
                                #fingerprint matching is much faster with ints than floats
                                Distances = np.rint(molecule_images[0].get_all_distances(mic=True)).astype('int')
                                Finger_Print_List = []
                                for CO_combo in itertools.product(CList,OList):
                                    #List of Carbon 12 atoms
                                    C12List = [C for C in Catoms if C not in CO_combo[0]]
                                    #List of Oxyge 16 atoms
                                    O16List = [O for O in Oatoms if O not in CO_combo[1]]
                                    #combine C12 and O16 indices
                                    Heads = C12List + O16List
                                    #combine C24 and O32 indices
                                    Tails = CO_combo[0] + CO_combo[1]
                                    #generate fingerprints of C12/O16 atoms distance from each other
                                    Finger_Print1 = list(np.sort(Distances[Heads][:,Heads].flatten()))
                                    #generate fingerprints of C24/O32 atoms distance from each other
                                    Finger_Print2 = list(np.sort(Distances[Tails][:,Tails].flatten()))
                                    #combine C12 and O32 indices
                                    Heads2 = CO_combo[1] + C12List
                                    #combine O16 and C24 indices
                                    Tails2 = O16List + CO_combo[0]
                                    #generate fingerprints of C12/O32 distances from each other
                                    Finger_Print3 = list(np.sort(Distances[Heads2][:,Heads2].flatten()))
                                    #generate fingerprints of C24/O16 distances from each other
                                    Finger_Print4 = list(np.sort(Distances[Tails2][:,Tails2].flatten()))
                                    Finger_Print = [Finger_Print1,Finger_Print2,Finger_Print3,Finger_Print4]
                                    if Finger_Print not in Finger_Print_List:
                                        Finger_Print_List.append(Finger_Print)
                                        #GraphList.append(G)
                                        #create new array of masses
                                        new_masses = molecule_images[0].get_masses()
                                        #set masses of to C12 or O16
                                        new_masses[np.array(molecule_images[0].get_chemical_symbols()) == adsorbate_atoms[0]] = C12
                                        new_masses[np.array(molecule_images[0].get_chemical_symbols()) == adsorbate_atoms[1]] = O16
                                        #set masses in join list to C24 and O32
                                        new_masses[CO_combo[0]] = C24
                                        new_masses[CO_combo[1]] = O32
                                        #set masses of the molecule to the new masses
                                        for molecule in molecule_images:
                                            molecule.set_masses(new_masses)
                                        #Get number of C24 and O32 atoms
                                        NUM_C24 = len(CO_combo[0])
                                        NUM_O32 = len(CO_combo[1])
                                        #calculate the number of C24O32 by identifying if they are nearest neighbors
                                        NUM_C24O32 = 0
                                        for atom_pair in itertools.product(CO_combo[0],CO_combo[1]):
                                            if molecule_images[0].get_distance(atom_pair[0],atom_pair[1],mic=True) < 2:
                                                NUM_C24O32 += 1
                                        NUM_C12O16 = NUM_CO - NUM_C24 - NUM_O32 + NUM_C24O32
                                        NUM_C12O32 = NUM_O32 - NUM_C24O32
                                        NUM_C24O16 = NUM_C24 - NUM_C24O32
                                        
                                        #set values of dictionary
                                        VIB_DICT['NUM_CO'].append(NUM_CO)
                                        VIB_DICT['NUM_C12O16'].append(NUM_C12O16)
                                        VIB_DICT['NUM_C12O32'].append(NUM_C12O32)
                                        VIB_DICT['NUM_C24O16'].append(NUM_C24O16)
                                        VIB_DICT['NUM_C24O32'].append(NUM_C24O32)
                                        #Generate an infrared object
                                        infraredZ = Infrared(molecule_images,delta=delta, nfree=2, directions = 2)
                                        #Get frequencies and remove imaginary parts
                                        FREQUENCIES = infraredZ.get_frequencies().real
                                        #Get intensities of corresponding to frequenices
                                        VIB_DICT['INTENSITIES'].append(infraredZ.intensities)
                                        #Set intensities of imaginary frequencies to 0
                                        VIB_DICT['INTENSITIES'][-1][FREQUENCIES == 0] = 0
                                        #convert arrays to lists to store in json
                                        VIB_DICT['INTENSITIES'][-1] = VIB_DICT['INTENSITIES'][-1].tolist()
                                        VIB_DICT['FREQUENCIES'].append(FREQUENCIES.tolist())
                                        VIB_DICT['GCN'].append(GCN[0])
                                        VIB_DICT['CN_CO'].append(CN_CO[0])
                                        VIB_DICT['CN_PT'].append(TotalNN[0])
                                        VIB_DICT['NUM_PT'].append(NumPt)
                                        VIB_DICT['SURFACE_PT'].append(surface_atoms)
                                        VIB_DICT['SURFACE_AREA'].append(SURFACE_AREA)
                                        VIB_DICT['MIN_SEPARATION'].append(MIN_SEPARATION)
                                        VIB_DICT['CO_CN_CO'].append(CO_CN_CO[0])
                                        VIB_DICT['COVERAGE'].append(COVERAGE)
                                        VIB_DICT['CO_PER_A2'].append(CO_PER_A2)
                                        #Identify each normal mode
                                        MODE_ID = []
                                        REDUCED_MASS = []
                                        REDUCED_MASS_NORMED = []
                                        MIXED_FACTOR = []
                                        SELF_COVERAGE = []
                                        SELF_CO_PER_A2 = []
                                        for modenumber in range(int(num_disp/2)):
                                            #Get directions of displacements of normal mode
                                            mode_cartesion = infraredZ.get_mode(modenumber)
                                            #Get distance moved of each atom
                                            distance_moved = np.sum(mode_cartesion**2,axis=1)**0.5
                                            #get atom thats displaced the most in the normal mode
                                            max_moved = np.argmax(distance_moved)
                                            max_distance = distance_moved[max_moved]
                                            #combine all atoms that move
                                            Allatoms = Catoms+Oatoms
                                            #remove the primary atom from the list
                                            del Allatoms[Allatoms.index(max_moved)]
                                            #Get distances from primary atom to all other Carbon and oxygen atoms
                                            maxm_distances = molecule_images[0].get_distances(max_moved,Allatoms,mic=True)
                                            #partner atom is the nearest neighbor
                                            partner = Allatoms[np.argsort(maxm_distances)[0]]
                                            #identify if atom that moved the most is carbon or oxygen
                                            if molecule_images[0][max_moved].symbol==adsorbate_atoms[0]:
                                                C_idx = max_moved
                                                O_idx = partner
                                            elif molecule_images[0][max_moved].symbol==adsorbate_atoms[1]:
                                                C_idx = partner
                                                O_idx = max_moved
                                            else:
                                                print('max moved is not CO')
                                                C_idx = 0
                                                O_idx = 0 
                                            #List of atom indices moved at least 10% of the max
                                            indices_moved = np.argwhere(distance_moved>0.1*max_distance).flatten().tolist()
                                            #add the partner atom in case it did not move much
                                            indices_moved = list(set(indices_moved + [partner]))
                                            indices_mixed = [i for i in range(len(new_masses)) if np.rint(new_masses[i]).astype('int') \
                                            not in np.rint(new_masses[[max_moved,partner]]).astype('int')]
                                            MIXED_FACTOR.append(np.max(distance_moved[indices_mixed])/max_distance)
                                            #Get displacement vector
                                            C_cart = mode_cartesion[C_idx]
                                            #get the direction the atom moved the most in
                                            C_cart_idx = np.argmax(abs(C_cart))
                                            #get the sign of the displacement
                                            C_sign = np.sign(C_cart[C_cart_idx])
                                            O_cart = mode_cartesion[O_idx]
                                            O_cart_idx = np.argmax(abs(O_cart))
                                            O_sign = np.sign(O_cart[O_cart_idx])
                                            #identify if C12
                                            if np.isclose(new_masses[C_idx], C12):
                                                #identify if O16
                                                if np.isclose(new_masses[O_idx], O16):
                                                    SELF_COVERAGE.append(NUM_C12O16/surface_atoms)
                                                    SELF_CO_PER_A2.append(NUM_C12O16/SURFACE_AREA)
                                                    #identify if it moved the most perpendicular to the surface
                                                    if C_cart_idx == 2 and O_cart_idx ==2:
                                                        #if carbon and oxygen moved in opposite direction it is a C-O stretch
                                                        if C_sign*O_sign == -1:
                                                            MODE_ID.append('CO_C12O16')
                                                        #else it is a perpendicular Pt-CO stretch
                                                        else:
                                                            MODE_ID.append('vPtCO_C12O16')
                                                    #If the oxygen moved more than carbon it is a rotation
                                                    elif (O_cart**2).sum() > (C_cart**2).sum():
                                                        MODE_ID.append('rPtCO_C12O16')
                                                    #if carbon moved more than oxygen it is a parallel stretch
                                                    else:
                                                        MODE_ID.append('pPtCO_C12O16')
                                                elif np.isclose(new_masses[O_idx], O32):
                                                    SELF_COVERAGE.append(NUM_C12O32/surface_atoms)
                                                    SELF_CO_PER_A2.append(NUM_C12O32/SURFACE_AREA)
                                                    if C_cart_idx == 2 and O_cart_idx ==2:
                                                        if C_sign*O_sign == -1:
                                                            MODE_ID.append('CO_C12O32')
                                                        else:
                                                            MODE_ID.append('vPtCO_C12O32')
                                                    elif (O_cart**2).sum() > (C_cart**2).sum():
                                                        MODE_ID.append('rPtCO_C12O32')
                                                    else:
                                                        MODE_ID.append('pPtCO_C12O32')
                                            elif np.isclose(new_masses[C_idx],C24):
                                                if np.isclose(new_masses[O_idx],O16):
                                                    SELF_COVERAGE.append(NUM_C24O16/surface_atoms)
                                                    SELF_CO_PER_A2.append(NUM_C24O16/SURFACE_AREA)
                                                    if C_cart_idx == 2 and O_cart_idx ==2:
                                                        if C_sign*O_sign == -1:
                                                            MODE_ID.append('CO_C24O16')
                                                        else:
                                                            MODE_ID.append('vPtCO_C24O16')
                                                    elif (O_cart**2).sum() > (C_cart**2).sum():
                                                        MODE_ID.append('rPtCO_C24O16')
                                                    else:
                                                        MODE_ID.append('pPtCO_C24O16')
                                                elif np.isclose(new_masses[O_idx],O32):
                                                    SELF_COVERAGE.append(NUM_C24O32/surface_atoms)
                                                    SELF_CO_PER_A2.append(NUM_C24O32/SURFACE_AREA)
                                                    if C_cart_idx == 2 and O_cart_idx ==2:
                                                        if C_sign*O_sign == -1:
                                                            MODE_ID.append('CO_C24O32')
                                                        else:
                                                            MODE_ID.append('vPtCO_C24O32')
                                                    elif (O_cart**2).sum() > (C_cart**2).sum():
                                                        MODE_ID.append('rPtCO_C24O32')
                                                    else:
                                                        MODE_ID.append('pPtCO_C24O32')
                                            else:
                                                MODE_ID.append('OTHER')
                                                SELF_COVERAGE.append(COVERAGE)
                                                SELF_CO_PER_A2.append(CO_PER_A2)
                                            REDUCED_MASS.append(new_masses[C_idx]*new_masses[O_idx]/(new_masses[C_idx]+new_masses[O_idx]))
                                            lk = mode_cartesion**2
                                            REDUCED_MASS_NORMED.append(lk.sum()*np.sum(lk/(new_masses.reshape((-1,1))))**-1)
                                        VIB_DICT['SELF_COVERAGE'].append(SELF_COVERAGE)
                                        VIB_DICT['SELF_CO_PER_A2'].append(SELF_CO_PER_A2)
                                        VIB_DICT['MIXED_FACTOR'].append(MIXED_FACTOR)
                                        VIB_DICT['MODE_ID'].append(MODE_ID)
                                        VIB_DICT['REDUCED_MASS'].append(REDUCED_MASS)
                                        VIB_DICT['REDUCED_MASS_NORMED'].append(REDUCED_MASS_NORMED) 
                                        count_list.append(count)
        #check to see if each CO type is represented equally and sums to the total CO atoms
        for key in ['NUM_C12O16','NUM_C12O32','NUM_C24O16','NUM_C24O32']:
            print(np.unique(VIB_DICT[key],return_counts=True))
        print(np.sum(VIB_DICT['NUM_CO']))
        print(np.sum(VIB_DICT['NUM_C12O16'])*4)
        
        with open(output_file, 'w') as outfile:
            json.dump(VIB_DICT, outfile, sort_keys=True, indent=1)
        self.OUTPUT_DICTIONARY = VIB_DICT
        self.FREQ_FILES = freq_files
        self.CHARGE_FILES = charge_files
        self.INDICES_USED = np.array(count_list)
        
class COVERAGE_SCALING:
    """Class that generates coverage scaling parameters"""
    def __init__(self,primary_data_path):
        """
        Parameters
        ----------
        primary_data_path : str
            Primary isotopic data used in generate coverage scaling realtions
            
        Attributes
        ----------
        PRIMARY_DATA_PATH : str
            Primary isotopic data used in generate coverage scaling realtions
        """
        self.PRIMARY_DATA_PATH = primary_data_path
    
    def get_coverage_parameters(self, coverage_scaling_path):
        """Function that generates coverage scaling relations

        Parameters
        ----------
        coverage_scaling_path : str
        	Location to write coverage scaling parameter dictionary as a json.
                
        """
        COVERAGE_SCALING_DICT = {'CO_INT_EXP':[],'CO_FREQ':[], 'PTCO_INT_EXP':[], 'PTCO_FREQ': []}
        PRIMARY_DATA_PATH = self.PRIMARY_DATA_PATH
        with open(PRIMARY_DATA_PATH, 'r') as infile:
            ES_compressed = pd.io.json.read_json(infile)
        ES_compressed['GCN']= ES_compressed['GCN'].round(decimals=4)
        ES_exploded = explode(ES_compressed,['FREQUENCIES','INTENSITIES','MODE_ID','REDUCED_MASS','REDUCED_MASS_NORMED'\
                                             ,'MIXED_FACTOR','SELF_COVERAGE','SELF_CO_PER_A2'], fill_value='', preserve_index=True)
        def remove_char(string):
            if 'PtCO' in string:
                new_string = string[1:]
            else:
                new_string = string
            return new_string
        
        ES_exploded['MODE_ID'] = ES_exploded['MODE_ID'].apply(remove_char)
        
        ES_first = ES_exploded.drop(columns=['INTENSITIES','FREQUENCIES','REDUCED_MASS_NORMED','MIXED_FACTOR']\
                                    ).groupby([ES_exploded.index,'MODE_ID']\
                                              ).first()
        ES_sum = ES_exploded[['MODE_ID','INTENSITIES','REDUCED_MASS_NORMED']\
                             ].assign(**{'FREQxINT':ES_exploded['INTENSITIES']*ES_exploded['FREQUENCIES']\
                                       ,'NUMBER_OF_MODES':np.ones(len(ES_exploded['REDUCED_MASS_NORMED']),dtype='int')\
                                       ,'MIXEDxINT':ES_exploded['INTENSITIES']*ES_exploded['MIXED_FACTOR']}\
                                      ).groupby([ES_exploded.index,'MODE_ID']\
                                                ).sum().rename(columns={'INTENSITIES':'TOTAL_INTENSITY'})
        ES_sum = ES_sum.assign(**{'WEIGHTED_FREQUENCY':ES_sum['FREQxINT']/ES_sum['TOTAL_INTENSITY']\
                                  ,'AVG_REDUCED_MASS_NORMED':ES_sum['REDUCED_MASS_NORMED']/ES_sum['NUMBER_OF_MODES']\
                                  ,'WEIGHTED_MIXED':ES_sum['MIXEDxINT']/ES_sum['TOTAL_INTENSITY']}\
                               ).drop(columns=['FREQxINT','MIXEDxINT','REDUCED_MASS_NORMED'],inplace=False)
        ES_joined = ES_sum.join(ES_first\
                                ).reset_index(level='MODE_ID')
        ES_Low_Cov = ES_joined[['GCN','CN_CO','CN_PT','MODE_ID','WEIGHTED_FREQUENCY','TOTAL_INTENSITY']][(ES_joined['NUM_CO']==1) & (ES_joined['SURFACE_PT']==36)]
        ES_Low_Cov.rename(columns={'WEIGHTED_FREQUENCY':'LOW_COV_FREQUENCY','TOTAL_INTENSITY':'LOW_COV_INTENSITY'},inplace=True)
        #losing some values due to certain modes not existing on some of the low coverage surfaces for certain isotopes
        ES_joined = ES_joined.reset_index().merge(ES_Low_Cov, how='inner', on=['CN_CO','CN_PT','MODE_ID','GCN']\
                                                  ).set_index('index')  
        INTENSITY_SCALE = np.zeros(ES_joined.shape[0])
        INTENSITY_SHIFT = np.zeros(ES_joined.shape[0])
        SELF_CO = np.zeros(ES_joined.shape[0])
        vectors = zip(ES_joined['MODE_ID'],ES_joined['TOTAL_INTENSITY'],ES_joined['NUM_C12O16'], ES_joined['NUM_C12O32']\
                      ,ES_joined['NUM_C24O16'],ES_joined['NUM_C24O32'],ES_joined['LOW_COV_INTENSITY'])
        for count, parameters in enumerate(vectors):
            if parameters[6] > 0:
                if 'C12O16' in parameters[0]:
                    INTENSITY_SCALE[count] = parameters[1]/(parameters[2]*parameters[6])
                    INTENSITY_SHIFT[count] = parameters[1]/parameters[2] - parameters[6]
                    SELF_CO[count] = parameters[2]
                elif 'C12O32' in parameters[0]:
                    INTENSITY_SCALE[count] = parameters[1]/(parameters[3]*parameters[6])
                    INTENSITY_SHIFT[count] = parameters[1]/parameters[3] - parameters[6]
                    SELF_CO[count] = parameters[3]
                elif 'C24O16' in parameters[0]:
                    INTENSITY_SCALE[count] = parameters[1]/(parameters[4]*parameters[6])
                    INTENSITY_SHIFT[count] = parameters[1]/parameters[4] - parameters[6]
                    SELF_CO[count] = parameters[4]
                elif 'C24O32' in parameters[0]:
                    INTENSITY_SCALE[count] = parameters[1]/(parameters[5]*parameters[6])
                    INTENSITY_SHIFT[count] = parameters[1]/parameters[5] - parameters[6]
                    SELF_CO[count] = parameters[5]
        ES_joined = ES_joined.assign(**{'SELF_CO':SELF_CO\
                                        ,'INTENSITY_SCALE':INTENSITY_SCALE\
                                        ,'FREQUENCY_SCALE':ES_joined['WEIGHTED_FREQUENCY']/ES_joined['LOW_COV_FREQUENCY']\
                                        ,'INTENSITY_SHIFT':INTENSITY_SHIFT\
                                        ,'FREQUENCY_SHIFT':ES_joined['WEIGHTED_FREQUENCY']-ES_joined['LOW_COV_FREQUENCY']\
                                        ,'OTHER_COVERAGE':ES_joined['COVERAGE']-ES_joined['SELF_COVERAGE']\
                                        ,'OTHER_CO_PER_A2':ES_joined['CO_PER_A2']-ES_joined['SELF_CO_PER_A2']})
        
        def INT_func(X,a):
            return np.exp(a*X)
        def reg_m(y, x):
            # with statsmodels
            #X = sm.add_constant(x) # adding a constant
            model = sm.OLS(y-1, x).fit()
            return model
        CO_Filter = (ES_joined['COVERAGE']>0.0278) & (ES_joined['MODE_ID']=='CO_C12O16') & (ES_joined['INTENSITY_SCALE']>0) \
        & (ES_joined['MODE_ID'].str.startswith('CO_C')) & (ES_joined['NUM_C12O32']==0) & (ES_joined['NUM_C24O16']==0)
        popt_n, pcov = curve_fit(INT_func, ES_joined['CO_PER_A2'][CO_Filter], ES_joined['INTENSITY_SCALE'][CO_Filter])
        COVERAGE_SCALING_DICT.update({'CO_INT_EXP':popt_n[0]})
        FreqCO_CNCO = [ES_joined['FREQUENCY_SCALE'][CO_Filter & (ES_joined['CN_CO']==1)]\
                     ,ES_joined['FREQUENCY_SCALE'][CO_Filter & (ES_joined['CN_CO']==2)]
                     ,ES_joined['FREQUENCY_SCALE'][CO_Filter & (ES_joined['CN_CO']==3)]
                     ,ES_joined['FREQUENCY_SCALE'][CO_Filter & (ES_joined['CN_CO']==4)]]
        XCO_CNCO = [ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][CO_Filter & (ES_joined['CN_CO']==1)]\
                    ,ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][CO_Filter & (ES_joined['CN_CO']==2)]\
                    ,ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][CO_Filter & (ES_joined['CN_CO']==3)]\
                    ,ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][CO_Filter & (ES_joined['CN_CO']==4)]]
        for count, (freq_CO, X_CO) in enumerate(zip(FreqCO_CNCO, XCO_CNCO)):
            modelCO = reg_m(freq_CO, X_CO)
            COVERAGE_SCALING_DICT['CO_FREQ'].append(modelCO.params.to_dict())
        
          
        PtCO_Filter = (ES_joined['COVERAGE']>0.0278) & (ES_joined['MODE_ID']=='PtCO_C12O16') & (ES_joined['INTENSITY_SCALE']>0) \
        & (ES_joined['NUM_C12O32']==0) & (ES_joined['NUM_C24O16']==0)
        popt_n, pcov = curve_fit(INT_func, ES_joined['CO_PER_A2'][PtCO_Filter & (ES_joined['CN_CO']==1)], ES_joined['INTENSITY_SCALE'][PtCO_Filter & (ES_joined['CN_CO']==1)])
        COVERAGE_SCALING_DICT.update({'PTCO_INT_EXP':popt_n[0]})
        
        FreqPtCO_CNCO = [ES_joined['FREQUENCY_SCALE'][PtCO_Filter & (ES_joined['CN_CO']==1)]\
                     ,ES_joined['FREQUENCY_SCALE'][PtCO_Filter & (ES_joined['CN_CO']==2)]
                     ,ES_joined['FREQUENCY_SCALE'][PtCO_Filter & (ES_joined['CN_CO']==3)]
                     ,ES_joined['FREQUENCY_SCALE'][PtCO_Filter & (ES_joined['CN_CO']==4)]]
        XPtCO_CNCO = [ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][PtCO_Filter & (ES_joined['CN_CO']==1)]\
                    ,ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][PtCO_Filter & (ES_joined['CN_CO']==2)]\
                    ,ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][PtCO_Filter & (ES_joined['CN_CO']==3)]\
                    ,ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][PtCO_Filter & (ES_joined['CN_CO']==4)]]
                       
        modelCO = reg_m(FreqPtCO_CNCO[0], XPtCO_CNCO[0])
        COVERAGE_SCALING_DICT.update({'PTCO_FREQ':modelCO.params.to_dict()})
        
        with open(coverage_scaling_path, 'w') as outfile:
            json.dump(COVERAGE_SCALING_DICT, outfile, sort_keys=True, indent=1)
            
    def save_coverage_figures(self, figure_directory, adsorbate='CO',metal='Pt'\
                              ,frequency_scale_axis1=[0.98,1.10], frequency_scale_axis2=[0.995,1.138]\
                              ,y_2_ticks=[1,1.02,1.04,1.06,1.08,1.10,1.12],presentation=False):
        """Function generates and saves coverage scaling figures

        Parameters
        ----------
        figure_directory : str
        	Directory where coverage scaling figures are to be saved
        
        adsorbate : str
        	Name of adsorbate
            
        metal : str
        	Name of metal
            
        frequency_scale_axis1 : list of flaot
        	Scale of y axis for first and third subgraphs
            
        frequency_scale_axis2 : list of float
        	Scale of y axis for second and fourth subgraphs
            
        y_2_ticks : list of float
        	Tick points for y-axis of second and fourth subgraphs
            
        presentation : bool
        	Indicates whether presentation style graphs are to be created.
                
        """
        PRIMARY_DATA_PATH = self.PRIMARY_DATA_PATH
        with open(PRIMARY_DATA_PATH, 'r') as infile:
            ES_compressed = pd.io.json.read_json(infile)
        ES_compressed['GCN']= ES_compressed['GCN'].round(decimals=4)
        ES_exploded = explode(ES_compressed,['FREQUENCIES','INTENSITIES','MODE_ID','REDUCED_MASS','REDUCED_MASS_NORMED'\
                                             ,'MIXED_FACTOR','SELF_COVERAGE','SELF_CO_PER_A2'], fill_value='', preserve_index=True)
        def remove_char(string):
            if 'PtCO' in string:
                new_string = string[1:]
            else:
                new_string = string
            return new_string
        
        ES_exploded['MODE_ID'] = ES_exploded['MODE_ID'].apply(remove_char)
        
        ES_first = ES_exploded.drop(columns=['INTENSITIES','FREQUENCIES','REDUCED_MASS_NORMED','MIXED_FACTOR']\
                                    ).groupby([ES_exploded.index,'MODE_ID']\
                                              ).first()
        ES_sum = ES_exploded[['MODE_ID','INTENSITIES','REDUCED_MASS_NORMED']\
                             ].assign(**{'FREQxINT':ES_exploded['INTENSITIES']*ES_exploded['FREQUENCIES']\
                                       ,'NUMBER_OF_MODES':np.ones(len(ES_exploded['REDUCED_MASS_NORMED']),dtype='int')\
                                       ,'MIXEDxINT':ES_exploded['INTENSITIES']*ES_exploded['MIXED_FACTOR']}\
                                      ).groupby([ES_exploded.index,'MODE_ID']\
                                                ).sum().rename(columns={'INTENSITIES':'TOTAL_INTENSITY'})
        ES_sum = ES_sum.assign(**{'WEIGHTED_FREQUENCY':ES_sum['FREQxINT']/ES_sum['TOTAL_INTENSITY']\
                                  ,'AVG_REDUCED_MASS_NORMED':ES_sum['REDUCED_MASS_NORMED']/ES_sum['NUMBER_OF_MODES']\
                                  ,'WEIGHTED_MIXED':ES_sum['MIXEDxINT']/ES_sum['TOTAL_INTENSITY']}\
                               ).drop(columns=['FREQxINT','MIXEDxINT','REDUCED_MASS_NORMED'],inplace=False)
        ES_joined = ES_sum.join(ES_first\
                                ).reset_index(level='MODE_ID')
        ES_Low_Cov = ES_joined[['GCN','CN_CO','CN_PT','MODE_ID','WEIGHTED_FREQUENCY','TOTAL_INTENSITY']][(ES_joined['NUM_CO']==1) & (ES_joined['SURFACE_PT']==36)]
        ES_Low_Cov.rename(columns={'WEIGHTED_FREQUENCY':'LOW_COV_FREQUENCY','TOTAL_INTENSITY':'LOW_COV_INTENSITY'},inplace=True)
        #losing some values due to certain modes not existing on some of the low coverage surfaces for certain isotopes
        ES_joined = ES_joined.reset_index().merge(ES_Low_Cov, how='inner', on=['CN_CO','CN_PT','MODE_ID','GCN']\
                                                  ).set_index('index')  
        INTENSITY_SCALE = np.zeros(ES_joined.shape[0])
        INTENSITY_SHIFT = np.zeros(ES_joined.shape[0])
        SELF_CO = np.zeros(ES_joined.shape[0])
        vectors = zip(ES_joined['MODE_ID'],ES_joined['TOTAL_INTENSITY'],ES_joined['NUM_C12O16'], ES_joined['NUM_C12O32']\
                      ,ES_joined['NUM_C24O16'],ES_joined['NUM_C24O32'],ES_joined['LOW_COV_INTENSITY'])
        for count, parameters in enumerate(vectors):
            if parameters[6] > 0:
                if 'C12O16' in parameters[0]:
                    INTENSITY_SCALE[count] = parameters[1]/(parameters[2]*parameters[6])
                    INTENSITY_SHIFT[count] = parameters[1]/parameters[2] - parameters[6]
                    SELF_CO[count] = parameters[2]
                elif 'C12O32' in parameters[0]:
                    INTENSITY_SCALE[count] = parameters[1]/(parameters[3]*parameters[6])
                    INTENSITY_SHIFT[count] = parameters[1]/parameters[3] - parameters[6]
                    SELF_CO[count] = parameters[3]
                elif 'C24O16' in parameters[0]:
                    INTENSITY_SCALE[count] = parameters[1]/(parameters[4]*parameters[6])
                    INTENSITY_SHIFT[count] = parameters[1]/parameters[4] - parameters[6]
                    SELF_CO[count] = parameters[4]
                elif 'C24O32' in parameters[0]:
                    INTENSITY_SCALE[count] = parameters[1]/(parameters[5]*parameters[6])
                    INTENSITY_SHIFT[count] = parameters[1]/parameters[5] - parameters[6]
                    SELF_CO[count] = parameters[5]
        ES_joined = ES_joined.assign(**{'SELF_CO':SELF_CO\
                                        ,'INTENSITY_SCALE':INTENSITY_SCALE\
                                        ,'FREQUENCY_SCALE':ES_joined['WEIGHTED_FREQUENCY']/ES_joined['LOW_COV_FREQUENCY']\
                                        ,'INTENSITY_SHIFT':INTENSITY_SHIFT\
                                        ,'FREQUENCY_SHIFT':ES_joined['WEIGHTED_FREQUENCY']-ES_joined['LOW_COV_FREQUENCY']\
                                        ,'OTHER_COVERAGE':ES_joined['COVERAGE']-ES_joined['SELF_COVERAGE']\
                                        ,'OTHER_CO_PER_A2':ES_joined['CO_PER_A2']-ES_joined['SELF_CO_PER_A2']})
        
        def INT_func(X,a):
            return np.exp(a*X)
        def reg_m(y, x):
            # with statsmodels
            #X = sm.add_constant(x) # adding a constant
            model = sm.OLS(y-1, x).fit()
            return model
        markers = ['o','^','s','D']
        CO_Filter = (ES_joined['COVERAGE']>0.0278) & (ES_joined['MODE_ID']=='CO_C12O16') & (ES_joined['INTENSITY_SCALE']>0) \
        & (ES_joined['MODE_ID'].str.startswith('CO_C')) & (ES_joined['NUM_C12O32']==0) & (ES_joined['NUM_C24O16']==0)
        popt_n, pcov = curve_fit(INT_func, ES_joined['CO_PER_A2'][CO_Filter], ES_joined['INTENSITY_SCALE'][CO_Filter])
        print('Intensity coverage scaling factors')
        print(popt_n[0])
        
        #Plot C-O intensity scaling vs total spatial coverage
        if presentation == False:
            plt.figure(0,figsize=(3.5,2),dpi=400)
            sizey = 10
        else:
            plt.figure()
            sizey=26
        plt.plot(np.sort(ES_joined['CO_PER_A2'][CO_Filter]),INT_func(np.sort(ES_joined['CO_PER_A2'][CO_Filter]),popt_n[0]),'k-')
        for i in range(4):
            plt.plot(ES_joined['CO_PER_A2'][CO_Filter & (ES_joined['CN_CO']==i+1)],ES_joined['INTENSITY_SCALE'][CO_Filter & (ES_joined['CN_CO']==i+1)],'o',marker=markers[i])
        
        plt.ylabel(r'$\mathrm{\frac{Intensity\ @\ \theta}{Intensity\ @\ 1/36\ ML}}$',size=sizey) 
        plt.xlabel('Total spatial coverage ['+adsorbate+' per $\mathrm{\AA}^{2}$]')
        if presentation == True:
            plt.legend([r'$\mathrm{e^{%.1f\theta}}$' %(popt_n[0]),'Atop','Bridge','3-fold','4-fold'])
        plt.savefig(os.path.join(figure_directory,adsorbate+'_intensity_scale.jpg'), format='jpg')
        plt.close()

        R2 = get_r2(ES_joined['INTENSITY_SCALE'][CO_Filter],INT_func(ES_joined['CO_PER_A2'][CO_Filter],popt_n[0]))
        print(R2)
        
        FreqCO_CNCO = [ES_joined['FREQUENCY_SCALE'][CO_Filter & (ES_joined['CN_CO']==1)]\
                     ,ES_joined['FREQUENCY_SCALE'][CO_Filter & (ES_joined['CN_CO']==2)]
                     ,ES_joined['FREQUENCY_SCALE'][CO_Filter & (ES_joined['CN_CO']==3)]
                     ,ES_joined['FREQUENCY_SCALE'][CO_Filter & (ES_joined['CN_CO']==4)]]
        XCO_CNCO = [ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][CO_Filter & (ES_joined['CN_CO']==1)]\
                    ,ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][CO_Filter & (ES_joined['CN_CO']==2)]\
                    ,ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][CO_Filter & (ES_joined['CN_CO']==3)]\
                    ,ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][CO_Filter & (ES_joined['CN_CO']==4)]]
        
        
        #Plot C-O Frequency scaling vs total spatial coverage 
        params = {'figure.autolayout': False,
                  'axes.labelpad':2}
        rcParams.update(params)
        fig = plt.figure(1,figsize=(7.2,4),dpi=400)
        axes = fig.subplots(nrows=2, ncols=2)
        axes[0,0].set_xticks([])
        axes[0,1].set_xticks([])
        axes[0,1].set_yticks([])
        axes[1,1].set_yticks([])
        axes[0,0].set_ylim(frequency_scale_axis1)
        axes[0,1].set_ylim(frequency_scale_axis1)
        axes[1,0].set_ylim(frequency_scale_axis2)
        axes[1,1].set_ylim(frequency_scale_axis2)
        axes[1,0].set_yticks(y_2_ticks)
        axis_list = [axes[0, 0],axes[0, 1], axes[1, 0], axes[1, 1]]
        abcd = ['(a)','(b)','(c)','(d)']
        for count, (freq_CO, X_CO, marker) in enumerate(zip(FreqCO_CNCO, XCO_CNCO, markers)):
            modelCO = reg_m(freq_CO, X_CO)
            print_model = modelCO.summary()
            print(print_model)
            im=axis_list[count].scatter(X_CO.values[:,0], freq_CO, c=X_CO.values[:,1], marker=marker)
            axis_list[count].text(0.01,0.93,abcd[count],size=8,name ='Calibri',transform=axis_list[count].transAxes)
            axis_list[count].set_xlim([0,0.152])
        plt.gcf().subplots_adjust(bottom=0.09,top=0.98,left=0.09,right=0.97,wspace=0.05,hspace=0.05)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(),fraction=0.07,pad=0.01)
        cbar.set_label('Total spatial coverage ['+adsorbate+' per $\mathrm{\AA}^{2}$]')
        fig.text(0.5, 0.01, 'Spatial coverage of identical sites ['+adsorbate+' per $\mathrm{\AA}^{2}$]', ha='center',size=8)
        fig.text(0.01, 0.5, r'$\mathrm{\frac{Frequency\ @\ \theta}{Frequency\ @\ 1/36\ ML}}$', va='center', rotation='vertical',size=10)
        fig.savefig(os.path.join(figure_directory,adsorbate+'_frequency_scale.jpg'), format='jpg')
        plt.close()
        params = {
                  'figure.autolayout': True}
        rcParams.update(params)
        
          
        PtCO_Filter = (ES_joined['COVERAGE']>0.0278) & (ES_joined['MODE_ID']=='PtCO_C12O16') & (ES_joined['INTENSITY_SCALE']>0) \
        & (ES_joined['NUM_C12O32']==0) & (ES_joined['NUM_C24O16']==0)
        popt_n, pcov = curve_fit(INT_func, ES_joined['CO_PER_A2'][PtCO_Filter & (ES_joined['CN_CO']==1)], ES_joined['INTENSITY_SCALE'][PtCO_Filter & (ES_joined['CN_CO']==1)])
        print('Intensity coverage scaling factors')
        print(popt_n[0])
        R2 = get_r2(ES_joined['INTENSITY_SCALE'][PtCO_Filter & (ES_joined['CN_CO']==1)],INT_func(ES_joined['CO_PER_A2'][PtCO_Filter & (ES_joined['CN_CO']==1)],popt_n[0]))
        print(R2)
        
        FreqPtCO_CNCO = [ES_joined['FREQUENCY_SCALE'][PtCO_Filter & (ES_joined['CN_CO']==1)]\
                     ,ES_joined['FREQUENCY_SCALE'][PtCO_Filter & (ES_joined['CN_CO']==2)]
                     ,ES_joined['FREQUENCY_SCALE'][PtCO_Filter & (ES_joined['CN_CO']==3)]
                     ,ES_joined['FREQUENCY_SCALE'][PtCO_Filter & (ES_joined['CN_CO']==4)]]
        XPtCO_CNCO = [ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][PtCO_Filter & (ES_joined['CN_CO']==1)]\
                    ,ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][PtCO_Filter & (ES_joined['CN_CO']==2)]\
                    ,ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][PtCO_Filter & (ES_joined['CN_CO']==3)]\
                    ,ES_joined[['SELF_CO_PER_A2','CO_PER_A2']][PtCO_Filter & (ES_joined['CN_CO']==4)]]
        
        #Plot atop Pt-CO frequency scaling vs self coverage and intensity scaling vs total coverage
        fig, axes = plt.subplots(2,1, figsize=(3.5,3.8),dpi=400)
        im = axes[0].scatter(XPtCO_CNCO[0].values[:,0], FreqPtCO_CNCO[0], c=XPtCO_CNCO[0].values[:,1])
        axes[0].text(0.01,0.93,'(a)',size=8,name ='Calibri',transform=axes[0].transAxes)
        cbar = fig.colorbar(im, ax=axes[0])
        cbar.set_label('Total spatial coverage ['+adsorbate+' per $\mathrm{\AA}^{2}$]')
        axes[0].set_xlabel('Spatial coverage of identical sites ['+adsorbate+' per $\mathrm{\AA}^{2}$]')
        axes[0].set_ylabel(r'$\mathrm{\frac{Frequency\ @\ \theta}{Frequency\ @\ 1/36\ ML}}$',size=10)
        axes[1].plot(np.sort(ES_joined['CO_PER_A2'][PtCO_Filter & (ES_joined['CN_CO']==1)]),INT_func(np.sort(ES_joined['CO_PER_A2'][PtCO_Filter & (ES_joined['CN_CO']==1)]),popt_n[0]),'k-')
        axes[1].plot(ES_joined['CO_PER_A2'][PtCO_Filter & (ES_joined['CN_CO']==1)],ES_joined['INTENSITY_SCALE'][PtCO_Filter & (ES_joined['CN_CO']==1)],'o')
        axes[1].text(0.01,0.93,'(b)',size=8,name ='Calibri',transform=axes[1].transAxes)
        axes[1].set_xlabel('Total spatial coverage ['+adsorbate+' per $\mathrm{\AA}^{2}$]')
        axes[1].set_ylabel(r'$\mathrm{\frac{Intensity\ @\ \theta}{Intensity\ @\ 1/36\ ML}}$',size=10)
        fig.savefig(os.path.join(figure_directory,metal+adsorbate+'_scale_scale.jpg'), format='jpg')
        plt.close()
            
        modelCO = reg_m(FreqPtCO_CNCO[0], XPtCO_CNCO[0])
        print_model = modelCO.summary()
        print(print_model)
        
        #plot intensity shift of all Pt-CO bindying-types
        plt.figure(3, figsize=(3.5,2),dpi=400)
        for count, (freq_CO, X, marker) in enumerate(zip(FreqPtCO_CNCO, XPtCO_CNCO, markers)):
            X_CO = X.values
            plt.plot(ES_joined['CO_PER_A2'][PtCO_Filter & (ES_joined['CN_CO']==1+count)],ES_joined['INTENSITY_SHIFT'][PtCO_Filter & (ES_joined['CN_CO']==1+count)],marker)
        plt.legend(['Atop','Bridge','3-fold','4-fold'])
        plt.xlabel('Total spatial coverage ['+adsorbate+' per $\mathrm{\AA}^{2}$]' )
        plt.ylabel('Intensity Shift (AU)',size=8)
        plt.savefig(os.path.join(figure_directory,metal+adsorbate+'_all_shift.jpg'), format='jpg')
        plt.close()
        
        plt.figure(3, figsize=(3.5,2),dpi=400)
        for count, (freq_CO, X, marker) in enumerate(zip(FreqPtCO_CNCO, XPtCO_CNCO, markers)):
            X_CO = X.values
            plt.plot(ES_joined['CO_PER_A2'][PtCO_Filter & (ES_joined['CN_CO']==1+count)],ES_joined['TOTAL_INTENSITY'][PtCO_Filter & (ES_joined['CN_CO']==1+count)]/ES_joined['SELF_CO'][PtCO_Filter & (ES_joined['CN_CO']==1+count)],marker)
        plt.legend(['Atop','Bridge','3-fold','4-fold'])
        plt.xlabel('Total spatial coverage ['+adsorbate+' per $\mathrm{\AA}^{2}$]' )
        plt.ylabel('Intensity (AU)',size=8)
        plt.savefig(os.path.join(figure_directory,metal+adsorbate+'_all_shift_INT.jpg'), format='jpg')
        plt.close()