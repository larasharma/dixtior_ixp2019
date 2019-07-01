# -*- coding: utf-8 -*-
"""
Constants.
"""

from __main__ import PROJECT_PATH, SCRIPTS
import os
from numpy import inf

RANDOM_STATE = 123


DATA = os.path.join(PROJECT_PATH, '1_Data Files')

P = inf  #For the Holder average

TABLES = os.path.join(SCRIPTS, 'tables')

WEIGHTED_HOLDER_BOOL = True

MODEL_VARIABLES_DICT = {
        'E' : [ 'economic_activity_code_risk',
               'society_type_risk', 'country_of_residence_risk'],
        'P' : ['age_risk', 'nationality_risk', 'occupation_risk', 
               'qualifications_risk', 'country_of_residence_risk'] }


BEHAVIOURAL_BOUNDARY = 0.6

USE_DORMANT_BOOL = False