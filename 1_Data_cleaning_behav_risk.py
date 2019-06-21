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
from constants import DATA, P

# =============================================================================
#IMPORTS 

#TODO: Henrique -> After the comment line you can add a title like above and then
#put a comment explaining the code. Names like "ec" and "br" may be quite cryptic
#if you are reading this code a month from now. Names like entity_client and
#behavioural_risk will take more space but be more understandable.

with open(os.path.join(DATA, 'entities.csv')) as file:
    ent = pd.read_csv(file, sep=';')

with open(os.path.join(DATA, 'accounts.csv')) as file:
    acc = pd.read_csv(file, sep=';')
    
with open(os.path.join(DATA, 'behavioural_risk.csv')) as file:
    behavioural_risk = pd.read_csv(file, sep=';')
    
with open(os.path.join(DATA, 'entity_client.csv')) as file:
    ent_client = pd.read_csv(file, sep=';')  

#TODO: Henrique -> avoid leaving so many blank spaces, and try to be consistent
# in the number of blank spaces you use. Also, why are you importing these tables?

xls = pd.ExcelFile(os.path.join(DATA, 'Risk Tables.xlsx'))
countrisk = pd.read_excel(xls, 'CountryRisk')
compage = pd.read_excel(xls, 'CompanyAgeRisk')


# =============================================================================
## DATA CLEANING

## Verify unique accounts - clearing duplicates

non_unique_entity_client = ent_client[ ent_client['unique_entities'].duplicated() ]
if non_unique_entity_client.shape[0] > 0:
    warn('Non-unique combinations of entity and client have beeen detected')

non_unique_acc_client = acc[ acc['unique_accounts'].duplicated() ]
if non_unique_acc_client.shape[0] > 0:
    warn('Non-unique combinations of entity and client have beeen detected')

non_unique_behavioural_risk = behavioural_risk[ behavioural_risk['unique_behavioural_risk'].duplicated() ]
if non_unique_behavioural_risk.shape[0] > 0:
    warn('Non-unique combinations of entity and client have beeen detected')
    
non_unique_ent = ent[ ent['unique_ent'].duplicated() ]
if non_unique_ent.shape[0] > 0:
    warn('Non-unique combinations of entity and client have beeen detected')

non_countrisk = countrisk[ countrisk['unique_countrisk'].duplicated() ]
if non_unique_countrisk.shape[0] > 0:
    warn('Non-unique combinations of entity and client have beeen detected')


#TODO: Henrique -> With pandas, it is better to do the following
ent_client.drop('unique_entities', axis=1, inplace=True)
acc.drop("unique_accounts", axis=1, inplace=True)
behavioural_risk.drop("unique_behavioural_risk", axis=1, inplace=True)
ent.drop("unique_ent", axis=1, inplace=True)
countrisk.drop("unique_countrisk", axis=1, inplace=True)


## Fill NA

behavioural_risk.continuous_risk = behavioural_risk.continuous_risk.fillna(behavioural_risk.continuous_risk.mean())
behavioural_risk.discrete_risk = behavioural_risk.discrete_risk.fillna(behavioural_risk.discrete_risk.mean())

#If there are null values in the future, an improvement to the code needs to 
#be made where we are filling the null values based off the mean values if it is
#empresas or particulares. fill in with the means of those E and P values.

#TODO: Not sure what this is for but kept in ~Lara
ent_client[ent_client.duplicated()].shape
countrisk[countrisk.duplicated()].shape

# =============================================================================

##Data Quality Assurance

#TODO: Henrique -> Again, what is the purpose of these lines without a print?
print( ent.date_of_birth.value_counts().head() )
#some of the values have a "0" birthdate.

ent.sort_values(by = "date_of_birth").head(10)
ent.loc[(ent['date_of_birth'] == 0)].shape
ent.loc[(ent['date_of_birth'] == 0) & (ent["entity_type"] != "P")].shape
ent.loc[(ent['date_of_birth'] == 0) & (ent["entity_type"] == "P")].shape
ent.loc[(ent['date_of_birth'] < 0)]


#Here we split the data set into two dataframe by entity_type
#This will lower the amount of null values in the data frame because not all of 
#the columns are applicable to both "E" and "P" entities. They were then saved
#as .csv files.


#TODO: Henrique -> This object is never used!
#ent_by_type = ent.groupby("entity_type")


#TODO: Henrique -> Again, you are assuming someone is going to run this line
#by line. You should either print, warn or raise an exception when you want
#to ask the attention of the person running the code.

ent_p = ent.loc[ent["entity_type"] == "P"]
print( ent_p.isnull().sum() )
ent_p.drop(["company_age_risk", "economic_activity_code_risk",
            "society_type_risk"], axis = 1).head()

ent_e = ent.loc[ent["entity_type"] == "E"]
ent_e.isnull().sum()
ent_e.drop(["nationality_risk", "occupation_risk", "qualifications_risk",
            "age_risk"], axis = 1).head()

#TODO: Henrique -> This is writing the csv in the wrong place. 
#You should explicitly choose the path. Define a variable TABLES in your 
#constants module that is TABLES = os.path.join(SCRIPTS, 'tables')
#and write your csv there as I am doing in the code bellow
#I also changed the name to something more understandable
#You should also drop the columns you are not using in the model
#you can do this by using the .drop(column_list, axis=1, inplace=True)
with open( os.path.join(TABLES, 'private_entity_model.csv')) as file:
    ent_p.to_csv(file, sep = ';')
    
with open( os.path.join(TABLES, 'enterprise_entity_model.csv')) as file:
    ent_e.to_csv(file, sep = ';')


#TODO: Henrique -> Do this before you export the datataset, so that the 
#private and company model csv have the risk variable ready for the model
#to use

# =============================================================================
## ENTITY BEHAVIOURAL RISK
# Determine entity risk


    
## Merge tables
b_risk_acc = pd.merge(acc, behavioural_risk, on='account_number')


## Associate client risk to maximum risk of accounts
max_risk= b_risk_acc.groupby('client_number').max().continuous_risk
risk_client = pd.merge(b_risk_acc, max_risk, on ='client_number')



## Merge the resulting table with the entity_client table
ent_b_risk = pd.merge(ent_client, risk_client, on='client_number')[['client_number',
        'entity_number', 
        'continuous_risk_y']].rename(columns={
        "continuous_risk_y": 'risk_client'})


## Count the number of entities that form each client
count_entity = ent_b_risk.groupby('client_number').count().entity_number
with_ent = pd.merge(ent_b_risk, count_entity, on='client_number').rename(columns={
        "entity_number_y": 'count_ent'})


    
## Distribute the risk evenly among the entities
with_ent['perc_risk'] = 1/with_ent['count_ent']
with_ent['ent_b_risk'] = with_ent['risk_client'] * with_ent['perc_risk']



## Remove and rename columns to get the final table
final = with_ent[['entity_number_x', 'ent_b_risk']].rename(columns={
        "entity_number_x": 'entity_number'})
    

client_size = ent_client.groupby('client_number').size().to_frame('client_size')
#Add this to the dataframe (create a new one)
df2 = ent_client.merge(client_size, left_on = 'client_number', right_index = True)
#Compute inverses
df2['weight_aux'] = 1/df2['client_size']
#Now add all auxiliary weights to normalize
weight_aux_sum = df2.groupby('client_number')['weight_aux'].sum(
        ).to_frame('weight_aux_sum')
#Add the column to the dataframe and compute final weight
df2 = df2.merge(weight_aux_sum, left_on = 'client_number', right_index = True)
df2['weight'] = df2['weight_aux']/df2['weight_aux_sum']


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

#TODO: Assign risk to the entities using the weighted holder mean
ent_client = ent_client.drop(['is_first'], axis=1)
behavioural_risk = behavioural_risk.drop(['discrete_risk'], axis=1)

#br_df = br_df.drop(columns = ['continuous_risk', 'cluster' ], axis=1)



ec_acc_df = pd.merge(ent_client,acc, left_on="client_number", 
                     right_on="client_number")



#creating one large dataset connecting entity number to behavioral risk
entity_behav_risk_df = pd.merge(behavioural_risk, ec_acc_df, left_on="account_number", 
                                right_on="account_number", how='inner')

#reordering the columns
entity_behav_risk_df = entity_behav_risk_df[['entity_number','client_number',
                            'account_number','cluster','continuous_risk']]

#dormant_acc = entity_behav_risk_df[entity_behav_risk_df.cluster == 660]




#TODO: accomodate for future immature values by determining if the account 
#is tied to a private or enterprise, then assigning an average value for 
#the risk based of if it is an enterprise or if it is a private 

for seg, behav_aux in entity_behav_risk_df.groupby('continuous_risk'):
        null_count = ( behav_aux.isnull().sum())

n_missing = null_count['continuous_risk']
if n_missing > 0:
    warn ("There are missing values")



entity_behav_risk_df.to_csv("entity_behavioral_risk.csv")
    
holder_risk = entity_behav_risk_df.groupby('entity_number', 
                    as_index=False).apply(weighted_holder)
#OR: weighted_risk = holder_risk.groupby('entity_number')['continuous_risk'].agg(weighted_holder)
final = pd.merge(entity_behav_risk_df, holder_risk, on='entity_number')
final= final[['entity_number', 'continuous_risk_y']].rename(columns={
        "continuous_risk_y": 'continuous_risk'})
print(final)

#TODO: Check if it is correct (manually for 1 or 2 cases - it will also help
#you to get a feel of the impact of the choice of p)











