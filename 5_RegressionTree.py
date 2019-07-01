#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:21:05 2019

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
import graphviz
from sklearn.tree import export_graphviz

#User defined constants
from constants import TABLES
from general_utils import compute_metrics


with open(os.path.join(TABLES, 'enterprise_entity_model.csv')) as file:
    ent_E = pd.read_csv(file, sep=';') 

with open(os.path.join(TABLES, 'private_entity_model.csv')) as file:
    ent_P = pd.read_csv(file, sep=';') 

# =============================================================================
# Regression Tree
# =============================================================================

#Regression tree for private:
target_regression_tree_P = "behavioural_risk"
independent_regression_tree_P = ['age_risk', 'nationality_risk', 
    'occupation_risk', 'qualifications_risk', 'country_of_residence_risk']

tree_regressor_P = DecisionTreeRegressor(max_depth=5)

tree_regressor_P.fit(ent_P[independent_regression_tree_P], 
                     ent_P[target_regression_tree_P])
regtree_predictions_P = tree_regressor_P.predict(
                      ent_P[independent_regression_tree_P])

##Graphing the tree for private:
dot_data_P = export_graphviz(tree_regressor_P, out_file=None, 
                         feature_names=independent_regression_tree_P,  
                         filled=True, 
                         impurity=False,
                         rounded=True,  
                         special_characters=True)  
    
graph = graphviz.Source(dot_data_P)
graph.format = 'png'
graph.render('tree_P',view=True);


#Regression tree for empresas:
target_regression_tree_E = "behavioural_risk"
independent_regression_tree_E = ['economic_activity_code_risk',
               'society_type_risk', 'country_of_residence_risk']

tree_regressor_E = DecisionTreeRegressor(max_depth=5)

tree_regressor_E.fit(ent_E[independent_regression_tree_E], 
                   ent_E[target_regression_tree_E])
regtree_predictions_E = tree_regressor_E.predict(
                      ent_E[independent_regression_tree_E])



##Graphing the tree for empresas:
dot_data_E = export_graphviz(tree_regressor_E, out_file=None, 
                         feature_names=independent_regression_tree_E,  
                         filled=True, 
                         impurity=False,
                         rounded=True,  
                         special_characters=True)  
    
graph = graphviz.Source(dot_data_E)
graph.format = 'png'
graph.render('tree_E',view=True);
# =============================================================================
"""
are there even metrics to run on this??
regression_tree_metrics = compute_metrics()
"""
# =============================================================================
