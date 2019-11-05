# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:46:31 2017

@author: lansford
"""

from __future__ import absolute_import, division, print_function
import os
from jl_spectra_2_structure.cross_validation import LOAD_CROSS_VALIDATION
from jl_spectra_2_structure.plotting_tools import set_figure_settings
#wasserstein_loss
#kl_div_loss
#Use 3350 max for C2H4, 2200 for CO, and 2000 for NO. Use 750 points for C2H4 and 500 for CO and 450 for NO.
#coverage is 'low', 'high' or a float <= 1
#assert TARGET in ['binding_type','GCN','combine_hollow_sites']
if __name__ == "__main__":
    Downloads_folder = os.path.join(os.path.expanduser("~"),'Downloads')
    set_figure_settings('paper')
    CV_class = LOAD_CROSS_VALIDATION()
    CV_class.load_CV_class(0)
    print('Total Explained Variance: ' + str(CV_class.TOTAL_EXPLAINED_VARIANCE))
    #CV_class.run_CV_multiprocess(write_file=True, CV_RESULTS_FILE = None, num_procs=CV_class.CV_SPLITS+1)
    BEST_MODELS = CV_class.get_best_models(3, 1)
    keys = CV_class.get_keys(BEST_MODELS)
    print(keys)
    #CV_class.plot_models(BEST_MODELS)
    CV_class.plot_models(BEST_MODELS,figure_directory=Downloads_folder)
    CV_class.plot_models(CV_class.CV_RESULTS,figure_directory=Downloads_folder,model_list=[6,7,9,10])
    CV_class.plot_parity_plots(figure_directory=Downloads_folder,model_list=[6,7,9,10])