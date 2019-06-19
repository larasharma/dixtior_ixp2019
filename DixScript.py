
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

#Remove the private entities with birthdate of 0
ent1 = ent[~((ent.entity_type == 'P') & (ent.date_of_birth ==0))]


# Missing Values:

## Split into 2 databases to see the missing values in the 2 dataframes:
private = ent1.copy()
private = ent1[ent1.entity_type == 'P']

n_records1 = len(private)
def missing_values_df(df):
    for column in df:
        print("{} | {} | {}".format(
            column, len(df[df[column].isnull()]) / (1.0*n_records1), df[column].dtype
        ))

missing_values_df(private)
## We can see that there is no null value in the risk columns that are 
## associated with private entities.



empresas = ent1.copy()
empresas = ent1[ent1.entity_type == 'E']
empresas

n_records2 = len(empresas)
def missing_values_df(df):
    for column in df:
        print("{} | {} | {}".format(
            column, len(df[df[column].isnull()]) / (1.0*n_records2), df[column].dtype
        ))

missing_values_df(empresas)

## We can see that there is no null value in the risk columns that are 
## associated with enterprises (empresas).


#cutoff_cl
#country_of_res: both
#nationality: P
#qualifications: P

##ENTITY BEHAVIOURAL RISK

#Compute how many entities in each client
client_size = ec.groupby('client_number').size().to_frame('client_size')
#Add this to the dataframe (create a new one)
ec2 = ec.merge(client_size, left_on = 'client_number', right_index = True)
#Compute inverses
ec2['weight_aux'] = 1/ec2['client_size']
#Now add all auxiliary weights to normalize
weight_aux_sum = ec2.groupby('client_number')['weight_aux'].sum(
        ).to_frame('weight_aux_sum')
#Add the column to the dataframe and compute final weight
ec2 = ec2.merge(weight_aux_sum, left_on = 'client_number', right_index = True)
ec2['weight'] = ec2['weight_aux']/ec2['weight_aux_sum']

from numpy.linalg import norm
#Comes from constants
P=2

def weighted_holder(series, p=1, weight = None):
   
    #Function to compute the weighted Holder average
    #TODO: complete this docstring
    
    """
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
    
    #Maximum - weights don't matter as long as weights are not zero
    if p == np.inf:
        result = series.max()
    #Better to assume
    else:
        result = (weight*series**p).sum()**(1/p)
    
    return result

#TODO: Generate randomly, then use real behavioural risk
rng = np.random.RandomState(123)
ec2['risk'] = rng.uniform(size=ec2.shape[0]) 

#TODO: Check if it is correct (manually for 1 or 2 cases - it will also help
#you to get a feel of the impact of the choice of p)
weighted_risk = ec2.groupby('client_number')['risk'].agg(weighted_holder)


#One model for p and E (for each)
#You can focus on one of them

#input: pred variabels, seg name (p or E), 
#output can be the same, 

#Build 2 csvs
#Create br for entities


 
