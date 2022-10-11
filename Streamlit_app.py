
import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from utilities import global_variable
from streamlit_callers.streamlit_caller import ModelCallers
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

### Declaring root variables
global_variable.declare()

###Initialize
global_variable.global_units_index = dict()
global_variable.test_grain_df = pd.read_csv(ROOT_DIR + "/data_files/test_grain.csv")
global_variable.train_subs_df = pd.read_csv(ROOT_DIR + "/data_files/train_subs_df.csv")
global_variable.ma_hyperparams_df = pd.read_csv(ROOT_DIR + "/data_files/moving_averages_df.csv")
global_variable.trained_regressors_dict = pickle.load(open(ROOT_DIR + "/data_files/trained_regressors_dict",'rb'))
global_variable.trained_regressors_info = pd.read_csv(ROOT_DIR + "/data_files/trained_regressors_info")

# give a title to our app
st.title('Walmart Sales Predictor')

store_numbers_tuple = (str(i) for i in range(1, 46))
item_numbers_tuple = (str(i) for i in range(1, 112))

store_number = st.selectbox('Select your store number : ', store_numbers_tuple)
item_number = st.selectbox('Select your item number : ', item_numbers_tuple)

date = None
test_radio_button = st.radio("Test on", ("validation data", "test data", "single test date"))

if test_radio_button == "validation data":
    st.text("Testing on validation data")
elif test_radio_button == 'test data':
    st.text("Testing on unseen data")
else:
    date = st.date_input('Select the date you wish to predict for : ',datetime(2012, 10, 10).date())

if(st.button('Predict Sales')):

    if test_radio_button == "single test date":
        test_model_callers = ModelCallers(date=date, item_number=item_number,
                                     store_number=store_number, testing_type=test_radio_button)
        ma_predictor_res = test_model_callers.call_ma_predictor()
        rf_predictor_res = test_model_callers.call_rf_predictor()
        st.text(ma_predictor_res)
        st.text(rf_predictor_res)

    elif test_radio_button == 'validation data':
        val_model_callers = ModelCallers(date=date, item_number=item_number,
                                     store_number=store_number, testing_type=test_radio_button)
        res = val_model_callers.call_rf_predictor()
        chart_data = pd.DataFrame(res)
        st.line_chart(chart_data)

        res = val_model_callers.call_ma_predictor()
        chart_data = pd.DataFrame(res)
        st.line_chart(chart_data)

    elif test_radio_button == 'test data':
        test_model_callers = ModelCallers(date=date, item_number=item_number,
                                         store_number=store_number, testing_type=test_radio_button)
        res_ma = test_model_callers.call_ma_predictor()
        res_rf = test_model_callers.call_rf_predictor()

        res_ma_df = pd.DataFrame(res_ma)
        res_rf_df = pd.DataFrame(res_rf)

        res_df = pd.concat([res_ma_df, res_rf_df],axis=1)
        st.line_chart(res_df)


