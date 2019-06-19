#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:01:31 2019

@author: larasharma
"""
#%%
import pandas as pd
import numpy as np

from warnings import warn

#%%

with open('entity_client.csv') as file:
    ec_df = pd.read_csv(file, sep=';')
    
with open('accounts.csv') as file:
    accounts_df = pd.read_csv(file, sep=';')
    
with open('behavioural_risk.csv') as file:
    br_df = pd.read_csv(file, sep=';')
    
with open('particulares.csv') as file:
    private_df = pd.read_csv(file, sep=';')

with open('empresas.csv') as file:
    enterprise_df = pd.read_csv(file, sep=';')

#%%
ec_df = ec_df.drop(['is_first'], axis=1)
br_df = br_df.drop(['continuous_risk'], axis=1)

#br_df = br_df.drop(columns = ['continuous_risk', 'cluster' ], axis=1)

#%%

ec_acc_df = pd.merge(ec_df,accounts_df, left_on="client_number", right_on="client_number")

#%%

#creating one large dataset connecting entity number to behavioral risk
entity_behav_risk_df = pd.merge(br_df, ec_acc_df, left_on="account_number", right_on="account_number", how='inner')

dormant_acc = entity_behav_risk_df[entity_behav_risk_df.cluster == 660]


#%%
for seg, behav_aux in entity_behav_risk_df.groupby('discrete_risk'):
        null_count = ( behav_aux.isnull().sum())

n_missing = null_count['discrete_risk']
if n_missing > 0:
    warn ("There are missing values")

#%%

entity_behav_risk_df.to_csv("entity_behavioral_risk.csv")
    
#TODO: accomdate for future immature values by determining if the account 
#is tied to a private or enterprise, then assigning an average value for 
#the risk based of if it is an enterprise or if it is a private 


