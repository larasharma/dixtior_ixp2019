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


#%% =============================================================================

xls = pd.ExcelFile(os.path.join(DATA, 'Risk Tables.xlsx'))

countrisk = pd.read_excel(xls, 'CountryRisk')

compage = pd.read_excel(xls, 'CompanyAgeRisk')


#%% =============================================================================
   
## DATA CLEANING

## Verify unique accounts - clearing duplicates
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

## Fill NA
br.continuous_risk = br.continuous_risk.fillna(br.continuous_risk.mean())
br.discrete_risk = br.discrete_risk.fillna(br.discrete_risk.mean())

## Duplicates
br.shape
br[br.duplicated()].shape
br.shape

#%%
##Data Quality Assurance

ent.date_of_birth.value_counts().head()

#some of the values have a "0" birthdate.

ent.sort_values(by = "date_of_birth").head(10)

ent.loc[(ent['date_of_birth'] == 0)].shape
ent.loc[(ent['date_of_birth'] == 0) & (ent["entity_type"] != "P")].shape
ent.loc[(ent['date_of_birth'] == 0) & (ent["entity_type"] == "P")].shape
ent.loc[(ent['date_of_birth'] < 0)]

"""
Here we split the data set into two dataframe by entity_type
This will lower the amount of null values in the data frame because not all of 
the columns are applicable to both "E" and "P" entities. They were then saved
as .csv files.
"""
ent_by_type = ent.groupby("entity_type")

ent_p = ent.loc[ent["entity_type"] == "P"]
ent_p.isnull().sum()
ent_p.drop(["company_age_risk", "economic_activity_code_risk", "society_type_risk"], axis = 1).head()

ent_e = ent.loc[ent["entity_type"] == "E"]
ent_e.isnull().sum()
ent_e.drop(["nationality_risk", "occupation_risk", "qualifications_risk", "age_risk"], axis = 1).head()

ent_p.to_csv("particulares.csv")
ent_e.to_csv("empresas.csv")


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
    
#%%
client_size = ec.groupby('client_number').size().to_frame('client_size')
#Add this to the dataframe (create a new one)
df2 = ec.merge(client_size, left_on = 'client_number', right_index = True)
#Compute inverses
df2['weight_aux'] = 1/df2['client_size']
#Now add all auxiliary weights to normalize
weight_aux_sum = df2.groupby('client_number')['weight_aux'].sum(
        ).to_frame('weight_aux_sum')
#Add the column to the dataframe and compute final weight
df2 = df2.merge(weight_aux_sum, left_on = 'client_number', right_index = True)
df2['weight'] = df2['weight_aux']/df2['weight_aux_sum']

from numpy.linalg import norm
#Comes from constants
P=2

def weighted_holder(series, p=1, weight = None):
    """
    Function to compute the weighted Holder average
    #TODO: complete this docstring

    Arguments:

	    series: df.series
	            Series for which you calculate the weighted holder mean.

	    p: int (default: 1)
	        Determines the weight given to higher numbers.
	        If p=1, lower numbers are as important as larger numbers.
	        If p=np.inf (infinity), the maximum number is the only number
	        that matters.

	    weight: float (default: None)
	        Weight assigned to an element of the series.
	        In the entities-account case, it is the weight assigned to the
	        account risk based on the number of entities that form the
	        client that owns the account.

	    Returns:
	        The weighted hölder mean of the series.
	     
    """

    
    if weight is None:
        weight = 1/series.shape[0]
    
    #Maximum - weights don't matter as long as weights that are not zero
    if p == np.inf:
        result = series.max()
    #Better to assume
    else:
        result = ( weight*series**p ).sum()**(1/p)
    
    return result

#TODO: Generate randomly, then use real behavioural risk
rng = np.random.RandomState(123)
df2['risk'] = rng.uniform(size=df2.shape[0]) 

#TODO: Check if it is correct (manually for 1 or 2 cases - it will also help
#you to get a feel of the impact of the choice of p)
weighted_risk = df2.groupby('client_number')['risk'].agg(weighted_holder)

print(weighted_risk)
