# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:43:15 2019

@author: lansf
"""
import numpy as np
Experiment_CO = np.array([[2080,2105,1855,2105,1854,2094,2087,2091,1901]]).T
DFT_CO = np.array([[2055,2071,1835,2075,1844,2068,2064,2068,1877]]).T
CO_SF = np.matmul(np.linalg.inv(np.matmul(DFT_CO.T,DFT_CO)),np.matmul(DFT_CO.T,Experiment_CO))[0][0]
CO_var = np.var(Experiment_CO.reshape(-1)-DFT_CO.reshape(-1)*CO_SF,ddof=1)
VAR_SF_CO = np.linalg.inv(np.matmul(DFT_CO.T,DFT_CO))*CO_var
print('STD in CO SF: '+ str(VAR_SF_CO**0.5))

Experiment_PtC = np.array([[472,464,376,472]]).T
DFT_PtC = np.array([[489,478,386,488]]).T
PtC_SF = np.matmul(np.linalg.inv(np.matmul(DFT_PtC.T,DFT_PtC)),np.matmul(DFT_PtC.T,Experiment_PtC))[0][0]
PtC_var = np.var(Experiment_PtC.reshape(-1)-DFT_PtC.reshape(-1)*PtC_SF,ddof=1)
VAR_SF_PtC = np.linalg.inv(np.matmul(DFT_PtC.T,DFT_PtC))*PtC_var
print('STD in Pt-C SF: '+ str(VAR_SF_PtC**0.5))