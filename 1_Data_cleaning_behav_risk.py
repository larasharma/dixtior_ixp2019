#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:11:23 2019

@author: brandon
"""

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
import numpy as np
from warnings import warn

#User defined constants
from constants import DATA, TABLES, P, WEIGHT_HOLDER_BOOL

# =============================================================================
#IMPORTS 

with open(os.path.join(DATA, 'entities.csv')) as file:
    ent = pd.read_csv(file, sep=';')

with open(os.path.join(DATA, 'accounts.csv')) as file:
    acc = pd.read_csv(file, sep=';')
    
with open(os.path.join(DATA, 'behavioural_risk.csv')) as file:
    behavioural_risk = pd.read_csv(file, sep=';')
    
with open(os.path.join(DATA, 'entity_client.csv')) as file:
    ent_client = pd.read_csv(file, sep=';')  

# =============================================================================
## DATA CLEANING
## Verify unique accounts 

ent_client['unique_entity_client'] = ent_client.client_number.astype(str).str.cat(
    [ent_client.entity_number.astype(str)],sep='-')

#In case of doubt, inspect the head of the DataFrame
ent_client_head = ent_client.head(100)


non_unique_entity_client = ent_client[ ent_client['unique_entity_client'].duplicated() ]
if non_unique_entity_client.shape[0] > 0:
    warn('Non-unique combinations of entity and client have been detected')
    
non_unique_acc = acc[ acc['account_number'].duplicated() ]
if non_unique_acc.shape[0] > 0:
    warn('Non-unique account numbers have been detected')

non_unique_behavioural_risk = behavioural_risk[ behavioural_risk['account_number'].duplicated() ]
if non_unique_behavioural_risk.shape[0] > 0:
    warn('Non-unique account numbers have been detected')
    
non_unique_ent = ent[ ent['entity_number'].duplicated() ]
if non_unique_ent.shape[0] > 0:
    warn('Non-unique entity numbers have been detected')


##Missing Values:
n_records = len(ent)
def missing_values_df(df):
    for column in df:
        print("{} | {} | {}".format(
            column, len(df[df[column].isnull()]) / (1.0*n_records), df[column].dtype
        ))

print(missing_values_df(ent))

#Most of the missing values are related to entity type. So, private entities 
#are missing risk that is related to companies while the latter are
#missing risk related to private entities. 

#The dataset was divided into 2 files in the constants file: one for private
# entities and another for enterprises. That will lower the amount of null values
#in the data frame because not all of the columns are applicable to both 
#"E" and "P" entities. They will then be saved as .csv files at the end of this code.
# =============================================================================

##Data Quality Assurance

print( ent.date_of_birth.value_counts() )
#Some of the values have a "0" birthdate.

#How many entities have a birthdate of 0?
print(ent.loc[(ent['date_of_birth'] == 0)].shape)

#How many enterprise entities have a birthdate of 0? 
#These make sense, companies do not have birthdates
print(ent.loc[(ent['date_of_birth'] == 0) & (ent["entity_type"] != "P")].shape)

#How many private entities have a birthdate of 0? 
#These do not make sense -- people have birth dates and this is important 
#information. These age risks should be flagged as 5 or removed from the data set
print(ent.loc[(ent['date_of_birth'] == 0) & (ent["entity_type"] == "P")].shape)

#We see that there are values with date of birth <0 as well
print(ent.loc[(ent['date_of_birth'] < 0)])

#Remove the private entities with 0 as date of birth
ent = ent[~((ent.entity_type == 'P') & ((ent.date_of_birth ==0)))]

#Make sure all the date of births are positive:
ent['date_of_birth'] = abs(ent['date_of_birth'])

# =============================================================================
## ENTITY BEHAVIOURAL RISK
#Determine entity risk

#Determine a function to calculate the weighted holder mean:
def weighted_holder(series, p=P, weight = None):
    """
    Function to compute the weighted Holder average

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

#Assign risk to the entities using the weighted holder mean
ent_client = ent_client.drop(['is_first'], axis=1)
behavioural_risk = behavioural_risk.drop(['discrete_risk'], axis=1)
#br_df = br_df.drop(columns = ['continuous_risk', 'cluster' ], axis=1)
ec_acc_df = pd.merge(ent_client,acc, left_on="client_number", 
                     right_on="client_number")

#Creating one large dataset connecting entity number to behavioral risk
entity_behav_risk_df = pd.merge(behavioural_risk, ec_acc_df, left_on="account_number", 
                                right_on="account_number", how='inner')

#Reordering the columns
entity_behav_risk_df = entity_behav_risk_df[['entity_number','client_number',
                            'account_number','cluster','continuous_risk']]
    
#entity_behav_risk_df.to_csv("entity_behavioral_risk.csv")
from functools import partial

weighted_holder_p = partial(weighted_holder, p=P)

## Weighted_Holder average

if WEIGHT_HOLDER_BOOL:
    #TODO: compute the weight
    acct_numb = ec_acc_df.drop(columns=['unique_entity_client']).groupby('entity_number').size().to_frame('number_of_accts')

    #Add this to the dataframe (create a new one)
    df2 = ec_acc_df.merge(acct_numb, left_on = 'entity_number', right_index = True)

    #Compute inverses
    df2['weight_aux'] = 1/df2['number_of_accts']

    #Now add all auxiliary weights to normalize
    weight_aux_sum = df2.groupby('account_number')['weight_aux'].sum(
        ).to_frame('weight_aux_sum')

    #Add the column to the dataframe and compute final weight
    df2 = df2.merge(weight_aux_sum, left_on = 'account_number', right_index = True)
    df2['weight'] = df2['weight_aux']/df2['weight_aux_sum']
    
else:
    #TODO: the weight is equal to 1/n where n is the number of accounts the entity has
    acct_numb = ec_acc_df.drop(columns=['unique_entity_client']).groupby('entity_number').size().to_frame('number_of_accts')

    #Add this to the dataframe (create a new one)
    df2 = ec_acc_df.merge(acct_numb, left_on = 'entity_number', right_index = True)
    
    #Compute inverses
    df2['weight'] = 1/df2['number_of_accts']

    
entity_behav_risk_df['holder_aux'] = (
    entity_behav_risk_df['weight'] * entity_behav_risk_df['continuous_risk']**P
    )
holder = entity_behav_risk_df.groupby('entity_number')['holder_aux'].sum()**(1/P)



entity_behav_risk_final = pd.merge(entity_behav_risk_df, holder, on='entity_number')
entity_behav_risk_final = entity_behav_risk_final[['entity_number', 'continuous_risk_y', 'client_number', 
               'account_number']].rename(columns={
               "continuous_risk_y":'continuous_risk'})
print(entity_behav_risk_final.head(20))

risk_with_ent_type = pd.merge(entity_behav_risk_final, ent, on='entity_number')
risk_with_ent_type = risk_with_ent_type[['entity_number', 'client_number',
                    'account_number','continuous_risk','entity_type']]


#Accommodate for future immature values by determining if the account 
#is tied to a private or enterprise, then assigning an average value for 
#the risk based on if it is an enterprise or if it is private 
ent_p = risk_with_ent_type[risk_with_ent_type.entity_type == 'P']
ent_e = risk_with_ent_type[risk_with_ent_type.entity_type == 'E']

for seg, behav_aux in ent_p.groupby('continuous_risk'):
        null_count = ( behav_aux.isnull().sum())

n_missing = null_count['continuous_risk']
if n_missing > 0:
    warn("There are missing values in the table for private entities.")    
ent_p.continuous_risk = ent_p.continuous_risk.fillna(ent_p.continuous_risk.mean())


for seg, behav_aux in ent_e.groupby('continuous_risk'):
        null_count1 = ( behav_aux.isnull().sum())

n_missing = null_count['continuous_risk']
if n_missing > 0:
    warn("There are missing values in the table for private entities.")        
ent_e.continuous_risk = ent_e.continuous_risk.fillna(ent_e.continuous_risk.mean())


##Divide the file into 2 csv files: one for the private entities and the other for
#the companies:
with open( os.path.join(TABLES, 'private_entity_model.csv'), 'w') as file:
    ent_p.to_csv(file, sep = ';')

with open( os.path.join(TABLES, 'enterprise_entity_model.csv'), 'w') as file:
    ent_e.to_csv(file, sep = ';')

#Check if it is correct (manually for 1 or 2 cases - it will also help
#you to get a feel of the impact of the choice of p)
#49994 entity
ent_49994 = np.sqrt((0.119944**2)/2) #correct

#50000
ent_50000 = np.sqrt((0.081082**2)/2) #correct

#9:
ent_9 = np.sqrt((0.081082**2)/2) #correct






