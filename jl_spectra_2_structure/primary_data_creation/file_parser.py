# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 18:46:25 2019

@author: lansf
"""
import os
import re
import numpy as np
import pandas as pd

class VASP_PARSER:
    """Class for parsing vasp and chargemol files"""
    def __init__(self, directory, create_files=False):
        """Summary goes on one line here

        Parameters
        ----------
        directory : str
        	Location of vasp and chargemol files
            
        create_files : bool
        	If True, charg2.extxyzstripped and vasprun2.xmlstripped files
            are generated which compile concatenated DDEC6 charge and
            vasprun.xml files into a format readable by PANDAS or ASE
        	
        Attributes
        ----------
        freq_dirs : list of str
            List of directories where concatenated vasprun.xml and chargemol
            files are located.
            
        create_files : list of str
            list of preprocessed concatenated DECC6 charge files
        		  
        """
        
        freq_dirs = [os.path.join(r,subdirectory) for r,d,f in os.walk(directory) for subdirectory in d\
                      if ('charge2.extxyz' in os.listdir(os.path.join(r,subdirectory))\
                          and 'vasprun2.xml' in os.listdir(os.path.join(r,subdirectory)))]
        self.freq_dirs = freq_dirs
        self.create_files = create_files
    
    def get_freq_files(self, directory=None):
        """Get list of porcessed concatenated vasprun.xml files

        Parameters
        ----------
        directory : str or None
        	directory of concatenated unprocessed concated vasprun.xml files
        		  
        Returns
        -------
        freq_files_clean : list of str
            List of processed concatenated vasprun.xml files
                
        """
        create_files = self.create_files
        if directory is None:
            directory = self.freq_dirs
        num_files = len(directory)
        freq_files = [subdirectory + '\\' + 'vasprun2.xml' for subdirectory in directory]
        freq_files_clean = [None]*num_files
        for count, freq_file in enumerate(freq_files):
            num_lines = sum(1 for line in open(freq_file)) - 10
            freq_files_clean[count] = freq_file + 'stripped.xml'
            if create_files == True:
                with open(freq_files_clean[count],'w') as outfile, open(freq_file,'r') as infile:
                    for num, line in enumerate(infile):
                        if (('xml' not in line and 'modeling' not in line) or num <10 or num > num_lines):
                            outfile.write(line)
        return freq_files_clean
    
    def get_charge_files(self,directory=None):
        """Get list of porcessed concatenated DDEC6 charge files

        Parameters
        ----------
        directory : str or None
        	directory of concatenated unprocessed concated chargemol files
        		  
        Returns
        -------
        charge_files_clean : list of str
            List of processed concatenated chargemol files
                
        """
        create_files = self.create_files
        if directory is None:
            directory = self.freq_dirs
        num_files = len(directory)
        charge_files = [subdirectory + '\\' + 'charge2.extxyz' for subdirectory in directory]
        charge_files_clean = [None]*num_files
        for count, charge_file in enumerate(charge_files):
            charge_files_clean[count] = charge_file + 'stripped.extxyz'
            if create_files == True:
                containsdipole = False
                linenum = 0
                with open(charge_files_clean[count],'w') as outfile, open(charge_file,'r') as infile:
                    for num, line in enumerate(infile):
                        if num==0:
                            atoms = float(line)
                        elif 'atom number' in line and 'net_charge' in line and 'dipole_x' in line:
                            linenum = num
                            if containsdipole is False:
                                outfile.write(line.replace(" ", ""))
                            containsdipole = True
                        elif num <= (linenum + atoms) and containsdipole is True:
                            outfile.write(re.sub(' +',',',line.strip(' ')))
        return charge_files_clean
    
def explode(df, lst_cols, fill_value='', preserve_index=True):
    """Take a pandas data frame with list components and explode
    so there is a separate line for each value in the list

    Parameters
    ----------
    df : pandas.dataframe
    	Dataframe to expand
        
    lst_cols : list
        list of columns names whose cells are lists that are to be expanded
        into their own rows
    fill_value : str
        If a column does not does not have the same number of indices as 
        in a specific row as the other cells with lists replace NA with
        fill_value
        
    preserve_index : bool
        Indicates whether the indices of the dataframe should be preserved
        after expanding
    		  
    Returns
    -------
    res : pandas.dataframe
        Expanded dataframe
            
    """
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values    
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF the values in each list have their own rows
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:        
        res = res.reset_index(drop=True)
    return res