# -*- coding: utf-8 -*-



 
#Weights that were estimated by us for the acceptance model
ACCEPTANCE_WEIGHTS = {
        'E' : {
                'company_age_risk' : 0.0287,
                'society_type_risk' : 0.0638,
                'economic_activity_code_risk' : 0.0962,
                'country_of_residence_risk' : 0.0613
                },
        'P': {
                'qualifications_risk' : 0.0148,
                'occupation_risk' : 0.0657,
                'nationality_risk' : 0.0866,
                'country_of_residence_risk' : 0.0829
                }
        }
#The constant term of the regression is always -0.25
#Check if -0.25 + sum_of (weight * variable) = reg column for 1 or 2 examples


BEHAVIOURAL_RISK_LIMITS = [0.2,0.4,0.6,0.8]

ACCEPTANCE_RISK_LIMITS = [0.12, 0.285, 0.54, 0.87]

#You can easily create a DataFrame from these values like this:
import pandas as pd
df_bhv_limits = pd.DataFrame(BEHAVIOURAL_RISK_LIMITS, columns=['BHV_limit'])
df_bhv_limits['merge_col_lower'] = df_bhv_limits.index.values
df_bhv_limits['merge_col_upper'] = df_bhv_limits.index.values + 1



