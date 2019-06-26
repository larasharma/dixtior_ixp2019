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

#User defined constants
from constants import DATA, TABLES, P, WEIGHTED_HOLDER_BOOL, MODEL_VARIABLES_DICT
# =============================================================================
#IMPORT DATA

#for the regression metrics
with open(os.path.join(DATA, #'regression.csv')) as file:
    ent = pd.read_csv(file, sep=';')
    
#For the classification metrics
with open(os.path.join(DATA, #'BC_var.csv')) as file:
    ent = pd.read_csv(file, sep=';')
# =============================================================================
#Regression metrics

evs = metrics.explained_variance_score(y_true, y_pred)
maxe = metrics.max_error(y_true, y_pred)
mae = metrics.mean_absolute_error(y_true, y_pred)
mse = metrics.mean_squared_error(y_true, y_pred[, …])
msle = metrics.mean_squared_log_error(y_true, y_pred)
medae = metrics.median_absolute_error(y_true, y_pred)
r2 = metrics.r2_score(y_true, y_pred[, …])

#create DataFrame
# =============================================================================
#Classification metrics
acc_score = metrics.accuracy_score(y_true, y_pred[, …])
auc = metrics.auc(x, y[, reorder])
avg_ps = metrics.average_precision_score(y_true, y_score)
c_matrix = metrics.confusion_matrix(y_true, y_pred[, …])
f1 = metrics.f1_score(y_true, y_pred[, labels, …])
prec_score = metrics.precision_score(y_true, y_pred[, …])
recall_score = metrics.recall_score(y_true, y_pred[, …])
roc_auc = metrics.roc_auc_score(y_true, y_score[, …])
roc_curve = metrics.roc_curve(y_true, y_score[, …])
#create DataFrame