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
from constants import TABLES
      
with open(os.path.join(TABLES, 'private_entity_model.csv')) as file:
    ent_p2 = pd.read_csv(file, sep=';')
with open(os.path.join(TABLES, 'enterprise_entity_model.csv')) as file:
    ent_e2 = pd.read_csv(file, sep=';')

# =============================================================================

#Script for Linear Regession Enitity Behavioral Risk

independent_variables_e = ent_e2[['economic_activity_code_risk',
               'society_type_risk', 'country_of_residence_risk']].values
target_variable = ent_e2['behavioural_risk'].values

model = LinearRegression()
model.fit(X=independent_variables_e, y=target_variable)
predictions_E = model.predict(independent_variables_e) 
predictions_E = pd.Series(predictions_E )

model.intercept_
coefs_E = model.coef_

## Linear regression for the particulares
independent_variables_p = ent_p2[['age_risk', 'nationality_risk', 'occupation_risk', 
               'qualifications_risk', 'country_of_residence_risk']].values
target_variable_p = ent_p2['behavioural_risk'].values


model.fit(X=independent_variables_p, y=target_variable_p)
predictions_P =  model.predict(independent_variables_p) 
predictions_P = pd.Series(predictions_P )


model.intercept_
coefs_P = model.coef_

# =============================================================================
