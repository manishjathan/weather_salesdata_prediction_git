import datetime

import pandas as pd
import numpy as np
from utilities import rf_utility
from ma_predictor.moving_average_caller import MovingAverageCaller
from utilities import global_variable
from regressor_predictor.regressor_predictor import RandomForestPredictor


class ModelCallers:
    def __init__(self, date, item_number, store_number, testing_type):
        self.ma_date = date
        self.rf_date = date
        if testing_type == 'single test date':
            self.rf_date = pd.to_datetime(pd.Series(date))
        self.item_nbr = str(item_number)
        self.store_nbr = str(store_number)
        self.grain = self.store_nbr + "_" + self.item_nbr
        self.testing_type = testing_type


    def call_ma_predictor(self):
        if self.testing_type == "single test date":
            try:
                grain_data = global_variable.train_subs_df[global_variable.train_subs_df['grain'] == self.grain].copy(deep=True)
                grain_data['date'] = pd.to_datetime(grain_data['date']).dt.date
                moving_avg_caller = MovingAverageCaller(date = self.ma_date, item_nbr=self.item_nbr, store_nbr=self.store_nbr, grain_data=grain_data)
                sma_pred, wma_pred, ewma_pred = moving_avg_caller.calc_all_averages()
                return {'sma_pred' : sma_pred, 'wma_pred' : wma_pred, 'ewma_pred' : ewma_pred}
            except Exception as e:
                return {'sma_pred' : 0, 'wma_pred' : 0, 'ewma_pred' : 0, 'message' : str(e)}

        elif self.testing_type == "validation data":
            try:
                grain_data = global_variable.train_subs_df[global_variable.train_subs_df['grain'] == self.grain].copy(deep=True)
                grain_data['date'] = pd.to_datetime(grain_data['date']).dt.date
                train_x, train_y, val_x, val_y = rf_utility.split_dataset(grain_data)

                sma_predictions = []
                wma_predictions = []
                ewma_predictions = []
                observations = []
                val_x.reset_index(inplace=True)
                val_y = val_y.values
                first_date = val_x.loc[0, 'date']
                date_range = [first_date + datetime.timedelta(days=x) for x in range(120)]

                val_index = 0
                for date in date_range:
                    moving_avg_caller = MovingAverageCaller(date = date, item_nbr=self.item_nbr, store_nbr=self.store_nbr, grain_data=grain_data)
                    sma_pred, wma_pred, ewma_pred = moving_avg_caller.calc_all_averages()
                    if date in list(val_x.date.values):
                        sma_predictions.append(sma_pred)
                        wma_predictions.append(wma_pred)
                        ewma_predictions.append(ewma_pred)
                        observations.append(val_y[val_index])
                        val_index += 1

                sma_predictions = pd.Series(np.array(sma_predictions))
                wma_predictions = pd.Series(np.array(wma_predictions))
                ewma_predictions = pd.Series(np.array(ewma_predictions))
                observations = pd.Series(np.array(observations))
                return {'observations' : observations, 'sma_predictions' : sma_predictions,
                            'wma_predictions' : wma_predictions, 'ewma_predictions' : ewma_predictions}
            except Exception as e:
                print("MA Exception : ", e)
                return {'observations' : pd.Series(np.zeros(120)), 'sma_predictions' : pd.Series(np.zeros(120)),
                        'wma_predictions' : pd.Series(np.zeros(120)), 'ewma_predictions' : pd.Series(np.zeros(120))}

        elif self.testing_type == "test data":
            try:
                grain_data = global_variable.train_subs_df[global_variable.train_subs_df['grain'] == self.grain].copy(deep=True)
                grain_data['date'] = pd.to_datetime(grain_data['date']).dt.date
                test_grain = global_variable.test_grain_df[global_variable.test_grain_df['grain'] == self.grain]
                test_grain['date'] = pd.to_datetime(test_grain['date']).dt.date

                sma_predictions = []
                wma_predictions = []
                ewma_predictions = []

                for date in test_grain.date.values:
                    moving_avg_caller = MovingAverageCaller(date = date, item_nbr=self.item_nbr, store_nbr=self.store_nbr, grain_data=grain_data)
                    sma_pred, wma_pred, ewma_pred = moving_avg_caller.calc_all_averages()
                    sma_predictions.append(sma_pred)
                    wma_predictions.append(wma_pred)
                    ewma_predictions.append(ewma_pred)

                sma_predictions = pd.Series(np.array(sma_predictions))
                wma_predictions = pd.Series(np.array(wma_predictions))
                ewma_predictions = pd.Series(np.array(ewma_predictions))
                return {'sma_predictions' : sma_predictions,'wma_predictions' : wma_predictions, 'ewma_predictions' : ewma_predictions}
            except Exception as e:
                print("MA Exception : ", e)
                return {'sma_predictions' : pd.Series(np.zeros(30)),'wma_predictions' : pd.Series(np.zeros(30)), 'ewma_predictions' : pd.Series(np.zeros(30))}

    def call_rf_predictor(self):
        rf_predictor = RandomForestPredictor(self.rf_date, self.grain, self.testing_type)
        print("Testing type : ", self.testing_type)

        if self.testing_type == "single test date":
            try:
               pred, adj_pred = rf_predictor.predict_units()
               return {'rf_pred' : pred, 'rf_adj_pred' : adj_pred}
            except Exception as e:
               return {'rf_pred' : 0, 'rf_adj_pred' : 0, 'message' : str(e)}

        elif self.testing_type == "validation data":
            try:
                observations, predictions, new_predictions = rf_predictor.predict_units()
                return {'observations' : observations, 'predictions' : predictions, 'new_predictions' : new_predictions}
            except Exception as e:
                print("RF Exception : ", e)
                return {'observations' : pd.Series(np.zeros(120)), 'predictions' : pd.Series(np.zeros(120)), 'new_predictions' : pd.Series(np.zeros(120))}

        elif self.testing_type == "test data":
            try:
                predictions, new_predictions = rf_predictor.predict_units()
                return {'predictions' : predictions, 'new_predictions' : new_predictions}
            except Exception as e:
                print("RF Exception : ", e)
                return {'predictions' : pd.Series(np.zeros(30)), 'new_predictions' : pd.Series(np.zeros(30))}
