#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:47:28 2019

@author: gracepetrosini
"""

import os, sys

#Variables that let you know the path where the script is - keep updated
PROJECT_FAMILY = 'Acceptance Behavioural'
SCRIPT_FOLDER = '2_Python Scripts'

#Find out where you are in interactive or non-interactive mode
try:
    SCRIPTS = os.path.dirname(__file__)
    PROJECT_PATH = SCRIPTS[:-( 1 + len(SCRIPT_FOLDER) )]
except:
    from global_constants import BASE_PROJECT_PATHS    
    #Determina o caminho do projeto e dos scripts
    PROJECT_PATH = os.path.join( BASE_PROJECT_PATHS[PROJECT_FAMILY])
    SCRIPTS = os.path.join(PROJECT_PATH,SCRIPT_FOLDER)
    del(BASE_PROJECT_PATHS)

#Adds the current path to path so modules can be imported
if SCRIPTS not in sys.path:
    sys.path.append( SCRIPTS )

#Join other paths you want
from paths_to_append import PATH_TO_APPEND_LIST
for path in PATH_TO_APPEND_LIST:
    if path not in sys.path:
        sys.path.append( path )

# =============================================================================
#IMPORTS - PYTHON MODULES
        
import pandas as pd

#User defined constants
from constants import TABLES, USE_ONLY_TEST_FOR_METRICS_BOOL, RESULTS
from general_utils import compute_metrics

# =============================================================================
# =============================================================================
writer = pd.ExcelWriter( os.path.join(RESULTS, 'Metrics_p4.xlsx'), engine='xlsxwriter')

#IMPORT DATA
for seg, seg_name in [('P','private'), ('E','enterprise')]: 
    print('-' *50, '\n', seg_name, '\n')
    #for the regression metrics
    with open(os.path.join(TABLES, '%s_entity_model_4.csv'%seg_name)) as file:
        df0 = pd.read_csv(file, sep=';')
        
    for name, (y_pred_cts_aux, y_pred_bin_aux, y_pred_score_aux) in [
            ('current_acceptance', ('predicted_behavioural_risk', 'BC_reg', None) ),
            ('linear_reg', ('linear_reg_pred', 'BC_linear_reg', None) ),
            ('logistic_reg', (None, 'BC_logistic_reg', 'score_logistic_reg') ),
            ('reg_tree', ('reg_tree_pred', 'BC_reg_tree', None) )
            ]:
        if USE_ONLY_TEST_FOR_METRICS_BOOL:
            df = df0.query('train_test_%s == "test" ' % name)

        print(name)
        if y_pred_cts_aux is not None:
            y_pred_cts = df[y_pred_cts_aux]
        else:
            y_pred_cts = None

        if y_pred_bin_aux is not None:
            y_pred_bin = df[y_pred_bin_aux]
        else:
            y_pred_bin = None
            
        if y_pred_score_aux is not None:
            y_pred_score = df[y_pred_score_aux]
        else:
            y_pred_score = None
        
        regression_metrics, classification_metrics = compute_metrics(
                df.behavioural_risk, y_pred_cts, df.BC_bhv, y_pred_bin, y_pred_score
                )
        #Export to excel

        if not regression_metrics.empty:
            regression_metrics.to_excel(writer, sheet_name = seg + '_' + name + "_Reg" )
        if not classification_metrics.empty:            
            classification_metrics.to_excel(writer, sheet_name = seg + '_' + name + "_BC")
        
    # =============================================================================
writer.save()
writer.close()
