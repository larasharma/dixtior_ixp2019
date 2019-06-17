#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:47:30 2019

@author: gracepetrosini
"""

# -*- coding: utf-8 -*-
"""
Generates BANKA tables
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
    #Determine the paths to the projects and the scripts
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


#IMPORTS - PYTHON MODULES
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from scipy import stats


#User defined constants
from constants import RANDOM_STATE, DATA, P


#Open files
with open(os.path.join(DATA, 'entities.csv')) as file:
    ent = pd.read_csv(file, sep=';')

with open(os.path.join(DATA, 'accounts.csv')) as file:
    acc = pd.read_csv(file, sep=';')
    
with open(os.path.join(DATA, 'behavioural_risk.csv')) as file:
    br = pd.read_csv(file, sep=';')
    
br.convert_objects(convert_numeric=True)

with open(os.path.join(DATA, 'entities.csv')) as file:
    ent = pd.read_csv(file, sep=';')
    
with open(os.path.join(DATA, 'entity_client.csv')) as file:
    ec = pd.read_csv(file, sep=';')  



xls = pd.ExcelFile(os.path.join(DATA, 'Risk Tables.xlsx'))

countrisk = pd.read_excel(xls, 'CountryRisk')

compage = pd.read_excel(xls, 'CompanyAgeRisk')



## DATA CLEANING

##Verify unique accounts - clearing duplicates
ec.client_number.value_counts().head()
ec.entity_number.value_counts().head()
ec['unique_entities'] = ec.client_number.astype(str).str.cat(
    [ec.entity_number.astype(str)],sep='-')
ec.unique_entities.value_counts().head()
    
acc.client_number.value_counts().head()
acc.account_number.value_counts().head()
acc['unique_accounts'] = acc.client_number.astype(str).str.cat(
    [acc.account_number.astype(str)],sep='-')
acc.unique_accounts.value_counts().head()

##There are no duplicate values
del ec["unique_entities"]
del acc["unique_accounts"]

##Fill NA
br.continuous_risk = br.continuous_risk.fillna(br.continuous_risk.mean())
br.discrete_risk = br.discrete_risk.fillna(br.discrete_risk.mean())

##Duplicates in other tables
br.shape
br[br.duplicated()].shape

ent.shape
ent[ent.duplicated()].shape

ec[ec.duplicated()].shape

countrisk[countrisk.duplicated()].shape

##Data Quality Assurance

ent.date_of_birth.value_counts().head()

#some of the values have a "0" birthdate.

ent.sort_values(by = "date_of_birth").head(10)

ent.loc[(ent['date_of_birth'] == 0)].shape
ent.loc[(ent['date_of_birth'] == 0) & (ent["entity_type"] != "P")].shape
ent.loc[(ent['date_of_birth'] == 0) & (ent["entity_type"] == "P")].shape
ent.loc[(ent['date_of_birth'] < 0)]

ent['date_of_birth'] = abs(ent['date_of_birth'])

len(ent.entity_number)

#Remove the people with birthdate of 0
ent1 = ent[~((ent.entity_type == 'P') & (ent.date_of_birth ==0))]


# Missing Values:

n_records = len(ent)
def missing_values_df(df):
    for column in df:
        print("{} | {} | {}".format(
            column, len(df[df[column].isnull()]) / (1.0*n_records), df[column].dtype
        ))

missing_values_df(ent)


risk_ent = ent.copy()
risk_ent = risk_ent[[
       'age_risk', 'company_age_risk', 'country_of_residence_risk',
        'economic_activity_code_risk', 'nationality_risk',
       'occupation_risk', 'qualifications_risk', 'society_type_risk'] ]

#cutoff_cl
#country_of_res: both
#nationality: P
#qualifications: P

test = ent.isnull().sum(axis=1)

result = []
for seg, ent_aux in ent.groupby('entity_type'):
    null_count = ( ent_aux.isnull().sum()/ent.shape[0] ).to_frame('null_percent')
    null_count['entity_type']  = seg  
    result.append( null_count)
result_df = pd.concat(result).reset_index(drop=False)
## ENTITY BEHAVIOURAL RISK
# Determine Client risk and then determine the percentage per entity

'''Assumptions:
    
   - Client risk is that of the riskiest account it holds. 
   
   - The client risk is equally distributed among entities 
     that form the client.
     
   - Main entities have the same risk as regular entities.
   
'''

'''
## Hölder mean: 
grouped_risk= br_acc.groupby('client_number')

for i in br_acc.continuous_risk:
    ((1/n)**(1/p)) * (sum(br_acc.continuous_risk[i]**p))**(1/p)

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