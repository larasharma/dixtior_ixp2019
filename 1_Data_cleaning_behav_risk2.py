#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:11:23 2019

Determines the behavioural risk of entities using the weighted holder average.

@author: brandon
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
from constants import DATA, TABLES, P, WEIGHTED_HOLDER_BOOL

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


non_unique_entity_client = ent_client[ 
        ent_client['unique_entity_client'].duplicated() 
        ]
if non_unique_entity_client.shape[0] > 0:
    warn('Non-unique combinations of entity and client have been detected')
    
ent_client = ent_client.drop('unique_entity_client', 1)

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
#TODO: Henrique -> Why are you creating a function and fixing n_records outside?
n_records = len(ent)
def missing_values_df(df):
    for column in df:
        print("{} | {} | {}".format(
            column, len(df[df[column].isnull()]) / (1.0*n_records), df[column].dtype
        ))

missing_values_df(ent)

#Most of the missing values are related to entity type. So, private entities 
#are missing risk that is related to companies while the latter are
#missing risk related to private entities. 

#The dataset was divided into 2 files in the constants file: one for private
# entities and another for enterprises. That will lower the amount of null values
#in the data frame because not all of the columns are applicable to both 
#"E" and "P" entities. They will then be saved as .csv files at the end of this code.
# =============================================================================

##Data Quality Assurance

#TODO: Henrique -> Maybe you should also print the meaning of what you are 
# showing as in the example I made bellow - also no need to use loc
#Also make sure you need all these prints. Simple is better than complicated 
n_ent_zero_birthdate = ent[(ent['date_of_birth'] == 0)].shape[0]
print('There are %d entities with "0" birthdate ' % n_ent_zero_birthdate )

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
    

## Weighted_Holder average

#TODO: not really a TODO, I changed the code to accouts instead of clients but
#I was confused. Now I put the way it should be.

if WEIGHTED_HOLDER_BOOL:
    #Compute how many entities in each client
    client_size = ent_client.groupby('client_number').size().to_frame('client_size')
    #Add this to the entity_behav_risk_df dataframe
    entity_behav_risk_df2 = entity_behav_risk_df.merge(
            client_size, left_on = 'client_number', right_index = True
            )
    #Compute inverses
    entity_behav_risk_df2['weight_aux'] = 1/entity_behav_risk_df2['client_size']
    #Now add all auxiliary weights to normalize
    weight_aux_sum = entity_behav_risk_df2.groupby('client_number')[
            'weight_aux'].sum().to_frame('weight_aux_sum')
    #Add the column to the dataframe and compute final weight
    entity_behav_risk_df2 = entity_behav_risk_df2.merge(
            weight_aux_sum, left_on = 'client_number', right_index = True
            )
    entity_behav_risk_df2['weight'] = (
            entity_behav_risk_df2['weight_aux']/entity_behav_risk_df2['weight_aux_sum']
            )    
else:
    #The weight is equal to 1/n where n is the number of accounts the entity has
    acct_numb = ec_acc_df.drop(['unique_entity_client'], axis=1).groupby('entity_number').size().to_frame('number_of_accts')

    #Add this to the dataframe (create a new one)
    entity_behav_risk_df2 = entity_behav_risk_df.merge(
            acct_numb, left_on = 'entity_number', right_index = True
            )
    
    #Compute inverses
    entity_behav_risk_df2['weight'] = 1/entity_behav_risk_df2['number_of_accts']

#Validate weights add up to one
weight_sum = entity_behav_risk_df2.groupby('entity_number')['weight'].sum()
assert ( (weight_sum-1).abs() > 1e-6 ).sum() == 0, 'The weights dont add up to 1'
    
entity_behav_risk_df2['holder_aux'] = (
    entity_behav_risk_df2['weight'] * entity_behav_risk_df2['continuous_risk']**P
    )
holder = entity_behav_risk_df2.groupby('entity_number')['holder_aux'].sum()**(1/P)


#TODO: Henrique -> You must convert holder to a DataFrame and merge with 
#left_index = 'entity_number', right_index = True since objects created 
#with group by have the grouped column as the index. Use .to_frame above
entity_behav_risk_final = pd.merge(
        entity_behav_risk_df2, holder, left_on='entity_number', right_index=True
        )

#TODO: I'm not sure which one is the right holder_aux (holder_aux_y or horler_aux_x) (Joëlle)
entity_behav_risk_final = entity_behav_risk_final[['entity_number', 'client_number', 
               'account_number', 'holder_aux_y']].rename(columns={
               "holder_aux_y":'holder_aux'})
entity_behav_risk_final = entity_behav_risk_final.drop_duplicates('entity_number')

#TODO: Henrique -> if you just want some columns there is no point in joining 
#the entire ent DataFrame  
#I think you also want the risk predictor columns 
risk_with_ent_type = pd.merge(entity_behav_risk_final, ent[['entity_type', 'entity_number']],
                              on='entity_number')




#Accommodate for future immature values by determining if the account 
#is tied to a private or enterprise, then assigning an average value for 
#the risk based on if it is an enterprise or if it is private 

#TODO: Henrique -> If we export this, we dropped the risk values 
#so how will we run our model? We have no predictor variables :P
ent_p = risk_with_ent_type[risk_with_ent_type.entity_type == 'P']
ent_e = risk_with_ent_type[risk_with_ent_type.entity_type == 'E']


##Divide the file into 2 csv files: one for the private entities and the other for
#the companies:
with open( os.path.join(TABLES, 'private_entity_model.csv'), 'w') as file:
    ent_p.to_csv(file, sep = ';')

with open( os.path.join(TABLES, 'enterprise_entity_model.csv'), 'w') as file:
    ent_e.to_csv(file, sep = ';')

# =============================================================================

#Check if it is correct (manually for 1 or 2 cases - it will also help
#you to get a feel of the impact of the choice of p)
#49994 entity
ent_49994 = np.sqrt((0.119944**2)/2) #correct

#50000
ent_50000 = np.sqrt((0.081082**2)/2) #correct

#9:
ent_9 = np.sqrt((0.081082**2)/2) #correct






