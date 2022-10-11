

from utilities import global_variable
from utilities import rf_utility
import numpy as np
import pandas as pd

class RandomForestPredictor():

    def __init__(self, date, grain, testing_type):
        self.date = date
        self.grain = grain
        self.testing_type = testing_type
        if self.testing_type == 'single test date':
            self.day = self.date.dt.day.values[0]
            self.wday = self.date.dt.weekday.values[0]
            self.year = self.date.dt.year.values[0]
            self.month = self.date.dt.month.values[0]
            self.week = self.date.dt.isocalendar().week.values[0]


    def predict_units(self):

        if self.grain in global_variable.trained_regressors_dict.keys():
            print("Initializing trained rf regressor")
            rf_regressor = global_variable.trained_regressors_dict[self.grain]

            if self.testing_type == 'single test date':
                pred = round(rf_regressor.predict(np.array([self.day, self.wday, self.year, self.month, self.week]).reshape(1, -1))[0], 2)
                adj_pred = round(pred - global_variable.trained_regressors_info[global_variable.trained_regressors_info['grain'] == self.grain]['adj_diff_param'].values[0], 2)
                return pred, adj_pred

            elif self.testing_type == 'validation data':
                try:
                    grain_data = global_variable.train_subs_df[global_variable.train_subs_df['grain'] == self.grain].reset_index()
                    grain_data['date'] = pd.to_datetime(grain_data['date'])
                    grain_data = rf_utility.create_individual_date_cols(grain_data)
                    grain_data = grain_data[['day', 'wday', 'year', 'month', 'week', 'units']]
                    train_x, train_y, val_x, val_y = rf_utility.split_dataset(grain_data)
                    predictions = rf_regressor.predict(val_x)

                    mean_prediction = np.mean(predictions)
                    adj_diff_param = global_variable.trained_regressors_info[global_variable.trained_regressors_info['grain'] == self.grain]['adj_diff_param']

                    new_predictions = np.zeros(len(predictions))
                    index = 0
                    for prediction in predictions:
                        if prediction < mean_prediction:
                            new_predictions[index] = prediction - adj_diff_param
                        else:
                            new_predictions[index] = prediction
                        index += 1
                    return val_y, predictions, new_predictions
                except:
                    return pd.Series(np.zeros(10).reshape(-1,1)), pd.Series(np.zeros(10).reshape(-1,1)), pd.Series(np.zeros(10).reshape(-1,1))

            elif self.testing_type == 'test data':
                try:
                    test_data = global_variable.test_grain_df[global_variable.test_grain_df['grain'] == self.grain].reset_index()
                    test_data['date'] = pd.to_datetime(test_data['date'])
                    test_data = rf_utility.create_individual_date_cols(test_data)
                    test_data = test_data[['day', 'wday', 'year', 'month', 'week']]
                    predictions = rf_regressor.predict(test_data)
                    mean_prediction = np.mean(predictions)
                    adj_diff_param = global_variable.trained_regressors_info[global_variable.trained_regressors_info['grain'] == self.grain]['adj_diff_param']

                    new_predictions = np.zeros(len(predictions))
                    index = 0
                    for prediction in predictions:
                        if prediction < mean_prediction:
                            new_predictions[index] = prediction - adj_diff_param
                        else:
                            new_predictions[index] = prediction
                        index += 1

                    return predictions, new_predictions
                except Exception as e:
                    return pd.Series(np.zeros(10)), pd.Series(np.zeros(10))
        else:
            if self.testing_type == 'validation data':
                return pd.Series(np.zeros(10)), pd.Series(np.zeros(10)), pd.Series(np.zeros(10))
            elif self.testing_type == 'single test date':
                raise Exception("Grain not found in Train Data, units prediction value must be zero!")



