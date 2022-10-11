import os
import pickle
import pandas as pd

from flask import Flask, request
from datetime import datetime
from utilities import global_variable
from ma_predictor.moving_average_caller import MovingAverageCaller
from regressor_predictor.regressor_predictor import RandomForestPredictor

app = Flask(__name__)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

### Declaring root variables
global_variable.declare()

###Initial
global_variable.global_units_index = dict()
global_variable.train_subs_df = pd.read_csv(ROOT_DIR + "/data_files/train_subs_df.csv")
global_variable.ma_hyperparams_df = pd.read_csv(ROOT_DIR + "/data_files/moving_averages_df.csv")
global_variable.trained_regressors_dict = pickle.load(open(ROOT_DIR + "/data_files/trained_regressors_dict",'rb'))
global_variable.trained_regressors_info = pd.read_csv(ROOT_DIR + "/data_files/trained_regressors_info")

@app.route('/')
def health_check():
    return 'Health check is successful!'

@app.route('/ma/predict_sales', methods=['POST'])
def predict_sales_ma():
    json_request = request.json

    ## fetch individual attributes
    date = datetime.strptime(json_request['date'], "%Y-%m-%d").date()
    item_nbr = json_request['item_nbr']
    store_nbr = json_request['store_nbr']
    grain = str(store_nbr) + '_' + str(item_nbr)

    if (grain not in list(global_variable.train_subs_df['grain'].values)) or (grain not in list(global_variable.ma_hyperparams_df['grain'].values)):
        return {'sma_pred' : 0, 'wma_pred' : 0, 'ewma_pred' : 0, 'message' : "Grain had all zero values in train data!"}
    else:
        try:
            grain_data = global_variable.train_subs_df[global_variable.train_subs_df['grain'] == grain].copy(deep=True)
            grain_data['date'] = pd.to_datetime(grain_data['date']).dt.date
            moving_avg_caller = MovingAverageCaller(date = date, item_nbr=item_nbr, store_nbr=store_nbr, grain_data=grain_data)
            sma_pred, wma_pred, ewma_pred = moving_avg_caller.calc_all_averages()
            return {'sma_pred' : sma_pred, 'wma_pred' : wma_pred, 'ewma_pred' : ewma_pred}
        except Exception as e:
            return {'sma_pred' : 0, 'wma_pred' : 0, 'ewma_pred' : 0, 'message' : str(e)}

@app.route('/regressor/predict_sales', methods=['POST'])
def predict_sales_rf():
    json_request = request.json

    ## fetch individual attributes
    date = pd.to_datetime(pd.Series(json_request['date']))
    item_nbr = json_request['item_nbr']
    store_nbr = json_request['store_nbr']
    grain = str(store_nbr) + '_' + str(item_nbr)

    try:
        rf_predictor = RandomForestPredictor(date, grain)
        pred, adj_pred = rf_predictor.predict_units()
        return {'rf_pred' : pred, 'rf_adj_pred' : adj_pred}
    except Exception as e:
        return {'rf_pred' : 0, 'rf_adj_pred' : 0, 'message' : str(e)}


if __name__ == '__main__':
    app.run()