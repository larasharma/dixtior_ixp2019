#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:33:55 2019

@author: larasharma
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
#IMPORTS & PYTHON MODULES
   
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from constants import TABLES, RANDOM_STATE, MODEL_VARIABLES_DICT, BEHAVIOURAL_BOUNDARY
          
# =============================================================================
#LINEAR REGRESSION FOR BOTH ENTITY TYPES

for seg, seg_name in [ ('E', 'enterprise'), ('P', 'private') ]:

    with open(os.path.join(TABLES, '%s_entity_model_1.csv'%seg_name)) as file:
        df = pd.read_csv(file, sep=';')
    
    model = LinearRegression()

    target = df["behavioural_risk"]
    independent = df[ MODEL_VARIABLES_DICT[seg] ]
    
    X_train, X_test, y_train, y_test = train_test_split(
            independent, target, test_size=0.2, random_state = RANDOM_STATE)

    model.fit(X_train, y_train)
    predictions_ent = pd.Series(model.predict(independent), index=df.index )

    df['train_test_linear_reg'] = 'train'
    
    df.loc[X_test.index, 'train_test_linear_reg'] = 'test'
    df['linear_reg_pred'] = predictions_ent
    df['BC_linear_reg'] = predictions_ent > BEHAVIOURAL_BOUNDARY

    
    with open( os.path.join(TABLES, '%s_entity_model_2.csv'%seg_name), 
              'w') as file:
        df.to_csv(file, index = False, sep = ';')
