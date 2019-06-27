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
from warnings import warn
from sklearn import metrics
#may need to import more things from sklearn. Not sure why sklearn will not import

#User defined constants
from constants import DATA, TABLES
# =============================================================================
#IMPORT DATA

#for the regression metrics
#with open(os.path.join(DATA, #'regression.csv')) as file:
#    ent = pd.read_csv(file, sep=';')
    
#For the classification metrics
#with open(os.path.join(DATA, #'BC_var.csv')) as file:
#    ent = pd.read_csv(file, sep=';')

#XXX: replace this with csv 
df = entity_behav_risk_final

# =============================================================================
#Regression metrics

y_true = df.behavioural_risk
y_pred = df.reg

regression_dict = {}
regression_dict['explained_variance_score'] = metrics.explained_variance_score(y_true, y_pred)
regression_dict['max_error'] = metrics.max_error(y_true, y_pred)
regression_dict['mean_absolute_error'] = metrics.mean_absolute_error(y_true, y_pred)
regression_dict['mean_squared_error'] = metrics.mean_squared_error(y_true, y_pred)
regression_dict['mean_squared_log_error'] = metrics.mean_squared_log_error(y_true, y_pred)
regression_dict['median_absolute_error'] = metrics.median_absolute_error(y_true, y_pred)
regression_dict['r2'] = metrics.r2_score(y_true, y_pred)

#create DataFrame
regression_metrics = df.from_dict(regression_dict, orient = 'index')

# =============================================================================
#Classification metrics
y_true = df.BC_bhv
y_pred = df.BC_reg

classification_dict = {}
classification_dict['accuracy_score'] = metrics.accuracy_score(y_true, y_pred)
#classification_dict['avg_ps'] = metrics.average_precision_score(y_true, y_score)
classification_dict['confusion_matrix'] = metrics.confusion_matrix(y_true, y_pred)
classification_dict['f1_score'] = metrics.f1_score(y_true, y_pred)
classification_dict['precision_score'] = metrics.precision_score(y_true, y_pred)
classification_dict['recall_score'] = metrics.recall_score(y_true, y_pred)
classification_dict['roc_auc_score'] = metrics.roc_auc_score(y_true, y_pred)
#classification_dict['roc_curve'] = metrics.roc_curve(y_true, y_score)
classification_dict['gini'] = 2*classification_dict['roc_auc_score']-1
classification_dict['sensibility'] = classification_dict['confusion_matrix'][1,1]/sum(classification_dict['confusion_matrix'][1,:])
classification_dict['specificity'] = classification_dict['confusion_matrix'][0,0]/sum(classification_dict['confusion_matrix'][0,:])

#create DataFrame
classification_metrics = df.from_dict(classification_dict, orient = 'index')
# =============================================================================
print(classification_metrics)
print(regression_metrics)

#Export to excel
#pd.regression_metrics.to_excel
# =============================================================================
