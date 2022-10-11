
import pandas as pd

def declare():
    ### Global variables related to Moving Averages
    global global_units_index
    global train_subs_df
    global ma_hyperparams_df

    ### Global variables related to Random Forest Regressors
    global trained_regressors_info
    global trained_regressors_dict

    ## Global variable to hold the test dataframe
    global test_grain_df
