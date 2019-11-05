# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:46:31 2017

@author: lansford
"""

from __future__ import absolute_import, division, print_function
from jl_spectra_2_structure.cross_validation import CROSS_VALIDATION

CV_SPLITS = 3

#write data for CO
CV_class = CROSS_VALIDATION(ADSORBATE='CO',INCLUDED_BINDING_TYPES=[1,2,3,4])
CV_class.generate_test_cv_indices(CV_SPLITS=CV_SPLITS, BINDING_TYPE_FOR_GCN=[1]\
    , test_fraction=0.25, random_state=0, read_file=False, write_file=True)

#Write data for NO
CV_class = CROSS_VALIDATION(ADSORBATE='NO',INCLUDED_BINDING_TYPES=[1,2,3,4])
CV_class.generate_test_cv_indices(CV_SPLITS=CV_SPLITS, BINDING_TYPE_FOR_GCN=[1]\
    , test_fraction=0.25, random_state=0, read_file=False, write_file=True)

#write data for C2H4
CV_class = CROSS_VALIDATION(ADSORBATE='C2H4',INCLUDED_BINDING_TYPES=[1,2])
CV_class.generate_test_cv_indices(CV_SPLITS=CV_SPLITS, BINDING_TYPE_FOR_GCN=[2]\
    , test_fraction=0.25, random_state=0, read_file=False, write_file=True)