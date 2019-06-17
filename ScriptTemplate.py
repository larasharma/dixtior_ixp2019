# -*- coding: utf-8 -*-
"""
Generates BANKA tables
"""

#%% =============================================================================

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

#%% =============================================================================
#IMPORTS - PYTHON MODULES
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from scipy import stats


#User defined constants
from constants import RANDOM_STATE, DATA, P

#%% =============================================================================

#Load the data and fit a decision tree
iris = load_iris()
clf = tree.DecisionTreeClassifier(random_state = RANDOM_STATE)
clf = clf.fit(iris.data, iris.target)

#Show the importance of each feature
feature_names = iris['feature_names']
for feature, importance in zip(feature_names, clf.feature_importances_ ):
    print( '%s has importance %.2f %%' % (feature, importance*100) )

with open(os.path.join(DATA, 'entities.csv')) as file:
    ent = pd.read_csv(file, sep=';', nrows=1000)

with open(os.path.join(DATA, 'accounts.csv')) as file:
    acc = pd.read_csv(file, sep=';', nrows=1000)
    
with open(os.path.join(DATA, 'behavioural_risk.csv')) as file:
    br = pd.read_csv(file, sep=';', nrows=1000)
    
br.convert_objects(convert_numeric=True)

with open(os.path.join(DATA, 'entities.csv')) as file:
    ent = pd.read_csv(file, sep=';', nrows=1000)
    
with open(os.path.join(DATA, 'entity_client.csv')) as file:
    ec = pd.read_csv(file, sep=';', nrows=1000)    

#%% =============================================================================

xls = pd.ExcelFile(os.path.join(DATA, 'Risk Tables.xlsx'))

countrisk = pd.read_excel(xls, 'CountryRisk')

compage = pd.read_excel(xls, 'CompanyAgeRisk')


#%% =============================================================================
   
## DATA CLEANING
    
## Fill NA
br.continuous_risk = br.continuous_risk.fillna(br.continuous_risk.mean())
br.discrete_risk = br.discrete_risk.fillna(br.discrete_risk.mean())

## Duplicates
br.shape
br[br.duplicated()].shape
br.shape

#%%
ent.shape
ent[ent.duplicated()].shape
ent.shape

ec[ec.duplicated()].shape

countrisk[countrisk.duplicated()].shape


## Extreme Values
def outliers_col(df):
    for column in df:
        if df[column].dtype != np.object:
            n_outliers = len(df[(np.abs(stats.zscore(df[column])) > 3)& \
                  (df[column].notnull())
                 ])
            print("{} | {} | {}".format(
                df[column].name,
                n_outliers,
                df[column].dtype
        ))

outliers_col(ent)


'''
## Hölder mean: 
grouped_risk= br_acc.groupby('client_number')

for i in br_acc.continuous_risk:
((1/n)**(1/p)) * (sum(br_acc.continuous_risk[i]**p))**(1/p)
'''


## ENTITY BEHAVIOURAL RISK
# Determine Client risk and then determine the percentage per entity

'''Assumptions:
    
   - Client risk is that of the riskiest account it holds. 
   
   - The client risk is equally distributed among entities 
     that form the client.
     
   - Main entities have the same risk as regular entities.
   
'''
    
## Merge tables
br_acc = pd.merge(acc, br, on='account_number')


## Associate client risk to maximum risk of accounts
max_risk= br_acc.groupby('client_number').max().continuous_risk
risk_client = pd.merge(br_acc, max_risk, on ='client_number')



## Merge the resulting table with the entity_client table
ent_br = pd.merge(ec, risk_client, on='client_number')[['client_number',
        'entity_number', 
        'continuous_risk_y']].rename(columns={
        "continuous_risk_y": 'risk_client'})


## Count the number of entities that form each client
count_entity = ent_br.groupby('client_number').count().entity_number
with_ent = pd.merge(ent_br, count_entity, on='client_number').rename(columns={
        "entity_number_y": 'count_ent'})


    
## Distribute the risk evenly among the entities
with_ent['perc_risk'] = 1/with_ent['count_ent']
with_ent['ent_b_risk'] = with_ent['risk_client'] * with_ent['perc_risk']



## Remove and rename columns to get the final table
final = with_ent[['entity_number_x', 'ent_b_risk']].rename(columns={
        "entity_number_x": 'entity_number'})
    
    
    
    