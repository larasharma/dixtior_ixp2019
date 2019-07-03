#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:21:05 2019

Generates the regression tree to determine whether the acceptance model 
predicts the behavioural model

@author: brandon
"""

##
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
#import graphviz
#from sklearn.tree import export_graphviz

#User defined constants
from constants import TABLES, MODEL_VARIABLES_DICT, RANDOM_STATE, BEHAVIOURAL_BOUNDARY
#from general_utils import compute_metrics


# =============================================================================
#REGRESSION TREE FOR BOTH ENTITY TYPES

for seg, seg_name in [ ('E', 'enterprise'), ('P', 'private') ]:

    with open(os.path.join(TABLES, '%s_entity_model_3.csv'%seg_name)) as file:
        df = pd.read_csv(file, sep=';')
    
    tree_regression = DecisionTreeRegressor(max_depth = 5, 
                                            random_state = RANDOM_STATE)

    target = df["behavioural_risk"]
    logistic_variable = ['BC_logistic_reg']
    independent_columns = logistic_variable+MODEL_VARIABLES_DICT[seg]
    independent = df[independent_columns]
    
    X_train, X_test, y_train, y_test = train_test_split(
            independent, target, test_size=0.2, random_state = RANDOM_STATE)

    tree_regression.fit(X_train, y_train)
    predictions_ent = pd.Series(tree_regression.predict(independent), 
                                index= df.index )

    df['train_test_reg_tree'] = 'train'
    
    df.loc[X_test.index, 'train_test_reg_tree'] = 'test'

    df['reg_tree_pred'] = predictions_ent
    df['BC_reg_tree'] = predictions_ent > BEHAVIOURAL_BOUNDARY
    
    with open( os.path.join(TABLES, '%s_entity_model_4.csv'%seg_name), 
              'w') as file:
        df.to_csv(file, index = False, sep = ';')



# =============================================================================
"""
are there even metrics to run on this??
regression_tree_metrics = compute_metrics()
"""     
# =============================================================================

