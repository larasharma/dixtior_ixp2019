#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:39:53 2019

@author: gracepetrosini
"""
import pandas as pd
from sklearn import metrics
# =============================================================================

# =============================================================================
def compute_metrics(y_true_cts, y_pred_cts, y_true_bin, y_pred_bin):
    #Linear Regression metrics
    
    y_true = y_true_cts
    y_pred = y_pred_cts
    
    regression_dict = {}
    regression_dict['explained_variance_score'] = metrics.explained_variance_score(y_true, y_pred)
    regression_dict['max_error'] = metrics.max_error(y_true, y_pred)
    regression_dict['mean_absolute_error'] = metrics.mean_absolute_error(y_true, y_pred)
    regression_dict['mean_squared_error'] = metrics.mean_squared_error(y_true, y_pred)
    #regression_dict['mean_squared_log_error'] = metrics.mean_squared_log_error(y_true, y_pred)
    regression_dict['median_absolute_error'] = metrics.median_absolute_error(y_true, y_pred)
    regression_dict['r2'] = metrics.r2_score(y_true, y_pred)
    
    #create DataFrame
    regression_metrics = pd.DataFrame.from_dict(regression_dict, orient = 'index')
    
    # =============================================================================
    #Classification metrics
    y_true = y_true_bin
    y_pred = y_pred_bin
    
    classification_dict = {}
    classification_dict['accuracy_score'] = metrics.accuracy_score(y_true, y_pred)
    #classification_dict['avg_ps'] = metrics.average_precision_score(y_true, y_score)
    classification_dict['confusion_matrix'] = metrics.confusion_matrix(y_true, y_pred)
    classification_dict['f1_score'] = metrics.f1_score(y_true, y_pred)
    classification_dict['precision_score'] = metrics.precision_score(y_true, y_pred)
    classification_dict['recall_score'] = metrics.recall_score(y_true, y_pred)
    classification_dict['roc_auc_score'] = metrics.roc_auc_score(y_true, y_pred)
    #classification_dict['roc_curve'] = metrics.roc_curve(y_true, y_score)
    classification_dict['gini'] = 2*classification_dict['roc_auc_score']-1
    classification_dict['sensibility'] = classification_dict['confusion_matrix'][1,1]/sum(classification_dict['confusion_matrix'][1,:])
    classification_dict['specificity'] = classification_dict['confusion_matrix'][0,0]/sum(classification_dict['confusion_matrix'][0,:])
    
    #create DataFrame
    classification_metrics = pd.DataFrame.from_dict(classification_dict, orient = 'index')
    # =============================================================================
    print(classification_metrics)
    print(regression_metrics)
    return regression_metrics, classification_metrics