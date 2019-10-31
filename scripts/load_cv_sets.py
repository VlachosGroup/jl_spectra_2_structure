# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:46:31 2017

@author: lansford
"""

from __future__ import absolute_import, division, print_function
from jl_spectra_2_structure.cross_validation import LOAD_CROSS_VALIDATION
#wasserstein_loss
#kl_div_loss
#Use 3350 max for C2H4, 2200 for CO, and 2000 for NO. Use 750 points for C2H4 and 500 for CO and 450 for NO.
#coverage is 'low', 'high' or a float <= 1
#assert TARGET in ['binding_type','GCN','combine_hollow_sites']
if __name__ == "__main__":
    CV_class = LOAD_CROSS_VALIDATION()
    CV_class.load_CV_class(0)
    print('Total Explained Variance: ' + str(CV_class.TOTAL_EXPLAINED_VARIANCE))
    #CV_class.run_CV_multiprocess(write_file=True, CV_RESULTS_FILE = None, num_procs=CV_SPLITS+1)
    CV_class.run_CV(write_file=True, CV_RESULTS_FILE = None)