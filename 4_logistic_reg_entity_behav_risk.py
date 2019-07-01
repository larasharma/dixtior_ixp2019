#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:15:42 2019

@author: larasharma
"""
# =============================================================================
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")

#User defined constants
from constants import TABLES

with open(os.path.join(TABLES, 'enterprise_entity_model.csv')) as file:
    ent_E = pd.read_csv(file, sep=';') 
with open(os.path.join(TABLES, 'private_entity_model.csv')) as file:
    ent_P = pd.read_csv(file, sep=';') 
    
# =============================================================================

#1 represents a high risk account
    
ent_E["BC_bhv"] = ent_E["BC_bhv"].replace(True, 1)
ent_P["BC_bhv"] = ent_P["BC_bhv"].replace(True, 1)

# =============================================================================

logisticRegr = LogisticRegression()

target_P = ent_P["BC_bhv"]
independent_P = ent_P[ ['age_risk', 'nationality_risk', 
    'occupation_risk', 'qualifications_risk', 'country_of_residence_risk'] ]

xp = independent_P
yp = target_P

xp_train, xp_test, yp_train, yp_test = train_test_split(xp, yp, test_size=0.2)

logisticRegr.fit(xp_train, yp_train)
predictions_ent_p = logisticRegr.predict(xp_test)

# =============================================================================

target_E = ent_E["BC_bhv"]
independent_E = ent_E[ ['economic_activity_code_risk',
    'society_type_risk', 'country_of_residence_risk'] ]

xe = independent_E
ye = target_E

xe_train, xe_test, ye_train, ye_test = train_test_split(xe, ye, test_size=0.2)

logisticRegr.fit(xe_train, ye_train)
predictions_ent_e = logisticRegr.predict(xe_test)


