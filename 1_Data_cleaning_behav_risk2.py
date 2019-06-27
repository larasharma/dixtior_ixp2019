#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:11:23 2019

Determines the behavioural risk of entities using the weighted holder average.

@authors: Grace, Lara, Joëlle
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
from sklearn.linear_model import LinearRegression
from warnings import warn

#User defined constants
from constants import DATA, TABLES, P, WEIGHTED_HOLDER_BOOL, MODEL_VARIABLES_DICT, BEHAVIOURAL_BOUNDARY

from weights_and_limits import BEHAVIOURAL_RISK_LIMITS, ACCEPTANCE_RISK_LIMITS

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

non_unique_behavioural_risk = behavioural_risk[ 
        behavioural_risk['account_number'].duplicated() ]
if non_unique_behavioural_risk.shape[0] > 0:
    warn('Non-unique account numbers have been detected')
    
non_unique_ent = ent[ ent['entity_number'].duplicated() ]
if non_unique_ent.shape[0] > 0:
    warn('Non-unique entity numbers have been detected')


##Missing Values:
def missing_values_df(df):
    for column in df:
        print("{} | {} | {}".format(
            column, len(df[df[column].isnull()]),
            df[column].dtype
        ))


print( missing_values_df(ent) )
print ( 'Most of the missing values are related to entity type. So, private entities are missing risk that is related to companies while the latter are missing risk related to private entities. ' )

#The dataset was divided into 2 files in the constants file: one for private
# entities and another for enterprises. That will lower the amount of null values
#in the data frame because not all of the columns are applicable to both 
#"E" and "P" entities. They will then be saved as .csv files at the end of this code.
# =============================================================================

##Data Quality Assurance

n_ent_zero_birthdate = ent[(ent['date_of_birth'] == 0)].shape[0]
print('There are %d entities with "0" birthdate ' % n_ent_zero_birthdate )


#How many enterprise entities have a birthdate of 0? 
#These make sense, companies do not have birthdates
empresas_birthdate = ent[(ent['date_of_birth'] == 0) & 
                         (ent["entity_type"] != "P")].shape[0]
print('There are %d enterprises with "0" birthdate' % empresas_birthdate )

#How many private entities have a birthdate of 0? 
#These do not make sense -- people have birth dates and this is important 
#information. These age risks should be flagged as 5 or removed from the data set
particulares_birthdate = ent[(ent['date_of_birth'] == 0) 
& (ent["entity_type"] == "P")].shape[0]
print('There are %d private customers with "0" birthdate' % particulares_birthdate)

#We see that there are values with date of birth <0 as well
negative_birthdate = ent[(ent['date_of_birth'] < 0)].shape[0]
print('There are %d entities with negative birthdate' % negative_birthdate)

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
entity_behav_risk_df = pd.merge(behavioural_risk, ec_acc_df, 
                                left_on="account_number", 
                                right_on="account_number", how='inner')

#Reordering the columns
entity_behav_risk_df = entity_behav_risk_df[['entity_number','client_number',
                            'account_number','cluster','continuous_risk']]
    
## Weighted_Holder average
if WEIGHTED_HOLDER_BOOL:
    #Compute how many entities in each client
    account_size = ec_acc_df.groupby('account_number').size().to_frame(
            'account_size')
    #Add this to the entity_behav_risk_df dataframe
    entity_behav_risk_df2 = entity_behav_risk_df.merge(
            account_size, left_on = 'account_number', right_index = True
            )
    #Compute inverses
    entity_behav_risk_df2['weight_aux'] = 1/entity_behav_risk_df2['account_size']
    #Now add all auxiliary weights to normalize
    weight_aux_sum = entity_behav_risk_df2.groupby('entity_number')[
            'weight_aux'].sum().to_frame('weight_aux_sum')
    #Add the column to the dataframe and compute final weight
    entity_behav_risk_df2 = entity_behav_risk_df2.merge(
            weight_aux_sum, left_on = 'entity_number', right_index = True
            )
    entity_behav_risk_df2['weight'] = (
    entity_behav_risk_df2['weight_aux']/entity_behav_risk_df2['weight_aux_sum']
            )    
else:
    #The weight is equal to 1/n where n is the number of accounts the entity has
    acct_numb = ec_acc_df.groupby('entity_number').size().to_frame('number_of_accts')

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
holder_df = holder.to_frame('behavioural_risk')
holder_df['BC_bhv'] = holder_df['behavioural_risk']>BEHAVIOURAL_BOUNDARY

entity_behav_risk_final = pd.merge(
        ent, holder_df, left_on='entity_number', 
        right_index=True)

#entity_behav_risk_final['behavioural_risk'].plot(kind='kde')

#make sure no duplcicates
duplicates = entity_behav_risk_final[entity_behav_risk_final.duplicated()].shape

if duplicates[0] != 0:
    warn ('There are duplicate values in the final dataset')

columns_in_both = ['entity_number', 'entity_type', 'behavioural_risk', 'reg', 
                   'BC_reg', 'BC_bhv']

columns_p = columns_in_both + MODEL_VARIABLES_DICT['P']
columns_e = columns_in_both + MODEL_VARIABLES_DICT['E']
ent_p = entity_behav_risk_final[entity_behav_risk_final.entity_type == 'P'][columns_p]
ent_e = entity_behav_risk_final[entity_behav_risk_final.entity_type == 'E'][columns_e]

#Check to see if there are missing values. 
#The meaning is different in each context, requires analysis on what should be filled in

for df in [ent_e, ent_p]:
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            warn('There are missing values in the {} column in {} dataframe'.format(column, df))

#Accommodate for future immature values by determining if the account 
#is tied to a private or enterprise, then assigning an average value for 
#the risk based on if it is an enterprise or if it is private 


##Divide the file into 2 csv files: one for the private entities and the other for
#the companies:


# =============================================================================

df_aux = entity_behav_risk_final[ ['entity_number', 'reg'] ].copy()
df_aux['d_reg'] = np.digitize(df_aux['reg'], [0] + BEHAVIOURAL_RISK_LIMITS + [1] )


def reg_merge( df_aux, risk_limits, string, minimum, maximum ):
    cutoffs = [minimum] + risk_limits + [maximum]
    cutoffs_aux = [ (cutoffs[i], cutoffs[i+1]) for i in range (len(cutoffs) - 1 ) ]
    boundaries = pd.DataFrame( cutoffs_aux, columns = [string + '_lower', string + '_upper'] )
    boundaries['d_reg'] = np.arange( boundaries.shape[0] ) + 1
    merged = pd.merge( df_aux, boundaries, left_on = 'd_reg', right_on = 'd_reg',
                         how = 'left' )
    return merged


dreg_merge = reg_merge( df_aux, BEHAVIOURAL_RISK_LIMITS, 'B', 0, 1 )
print(dreg_merge.columns)

dreg_merge = reg_merge( dreg_merge, ACCEPTANCE_RISK_LIMITS, 'R', 0, 1 )
print(dreg_merge.columns)

R_delta_name = "R_delta"
y_name = 'predicted_behavioural_risk' 


dreg_merge[R_delta_name] = (dreg_merge.reg - dreg_merge.R_lower) / (dreg_merge.R_upper - dreg_merge.R_lower)
dreg_merge[y_name] = (dreg_merge.B_lower) + dreg_merge[R_delta_name]*(dreg_merge.B_upper - dreg_merge.B_lower)

# =============================================================================
# merging dreg_merge with ent-e and ent-p with reg column

#ent_e['reg'].duplicated().sum()
#dreg_merge['reg'].duplicated().sum()


#ent_e_right = ent_e.merge(dreg_merge, on ='reg', how= 'right').shape[0]
#ent_e_left = ent_e.merge(dreg_merge, on='reg', how='left').shape[0]
#ent_e_inner = ent_e.merge(dreg_merge, on='reg', how='inner').shape[0]
#ent_e_outer = ent_e.merge(dreg_merge, on='reg', how='outer').shape[0]




ent_e2 = ent_e.drop(['reg'], axis = 1).merge(dreg_merge.drop(
        ['R_lower', 'R_upper', 'B_lower', 'B_upper', 'R_delta'], axis = 1 
        ), on ='entity_number')


ent_p2 = ent_p.drop(['reg'], axis = 1).merge(dreg_merge.drop(
        ['R_lower', 'R_upper', 'B_lower', 'B_upper', 'R_delta'], axis = 1 
        ), on ='entity_number')

assert ent_e2['entity_number'].duplicated().sum() == 0
assert ent_p2['entity_number'].duplicated().sum() == 0


with open( os.path.join(TABLES, 'private_entity_model.csv'), 'w') as file:
    ent_p2.to_csv(file, index = False, sep = ';')

with open( os.path.join(TABLES, 'enterprise_entity_model.csv'), 'w') as file:
    ent_e2.to_csv(file, index = False, sep = ';')




# =============================================================================
## Check with Henrique and Miguel:



## Linear regression for the empresas
#independent_variables = ent_e[['economic_activity_code_risk',
               #'society_type_risk', 'country_of_residence_risk']]
#target_variable = ent_e['behavioural_risk']

#model = LinearRegression()
#model.fit(X=independent_variables, y=target_variable)
#predictions = model.predict(independent_variables)

#model.intercept_
#model.coef_

## Linear regression for the particulares
#independent_variables_p = ent_p[['age_risk', 'nationality_risk', 'occupation_risk', 
               #'qualifications_risk', 'country_of_residence_risk']]
#target_variable_p = ent_p['behavioural_risk']


#model.fit(X=independent_variables_p, y=target_variable_p)
#predictions = model.predict(independent_variables_p)

#model.intercept_
#model.coef_







# ============================================================================


#df_aux['d_reg'] = 5
#df_aux['d_reg'] = np.where(entity_behav_risk_final['reg']<=0.8, 
                       #4, entity_behav_risk_final['reg'])
#df_aux['d_reg'] = np.where(entity_behav_risk_final['reg']<=0.6, 
                       #3, entity_behav_risk_final['reg'])
#df_aux['d_reg'] = np.where(entity_behav_risk_final['reg']<=0.4,
                      #2, entity_behav_risk_final['reg'])
#df_aux['d_reg'] = np.where(entity_behav_risk_final['reg']<=0.2, 
                       #1, entity_behav_risk_final['reg'])





