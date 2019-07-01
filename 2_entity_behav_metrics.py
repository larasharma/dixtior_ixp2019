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
from constants import TABLES
from general_utils import compute_metrics
# =============================================================================
# =============================================================================
#IMPORT DATA
for seg_name in ['private', 'enterprise']: 
    print('-' *50, '\n', seg_name, '\n')
    #for the regression metrics
    with open(os.path.join(TABLES, '%s_entity_model.csv'%seg_name)) as file:
        df = pd.read_csv(file, sep=';')
    
    regression_metrics, classification_metrics = compute_metrics(
            df.behavioural_risk, df.predicted_behavioural_risk, df.BC_bhv, df.BC_reg
            )
    #Export to excel
    with ExcelWriter("1_Data_Files/Metrics.xlsx") as writer:
        regression_metrics.to_excel(writer, sheet_name = "Linear Regression" )
        classification_metrics.to_excel(writer, sheet_name = "Classification")
        
        
    # =============================================================================
