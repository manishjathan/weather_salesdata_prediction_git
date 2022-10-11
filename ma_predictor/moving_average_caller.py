
from ma_predictor.moving_average_predictor import MovingAveragePredictor
from utilities import ma_utility, global_variable

class MovingAverageCaller:
    def __init__(self, date, store_nbr, item_nbr, grain_data):
        self.grain = store_nbr + "_" + item_nbr
        self.date = date
        self.grain_data = grain_data



    def calc_all_averages(self):

        sma_window = int(global_variable.ma_hyperparams_df[global_variable.ma_hyperparams_df['grain'] == self.grain]['sma_best_param'].values[0])
        wma_window = int(global_variable.ma_hyperparams_df[global_variable.ma_hyperparams_df['grain'] == self.grain]['wma_best_param'].values[0])
        ewma_window = int(global_variable.ma_hyperparams_df[global_variable.ma_hyperparams_df['grain'] == self.grain]['ewma_best_param'].values[0])


        sma_units = ma_utility.get_input_units(self.grain, self.grain_data, self.date, sma_window, 'sma')
        wma_units = ma_utility.get_input_units(self.grain, self.grain_data, self.date, wma_window, 'wma')
        ewma_units = ma_utility.get_input_units(self.grain, self.grain_data, self.date, ewma_window, 'ewma')

        ma_predictor = MovingAveragePredictor(sma_units = sma_units, wma_units = wma_units, ewma_units = ewma_units)
        ## Simple Moving Average prediction
        sma_pred = ma_predictor.get_sma()
        global_variable.global_units_index[(self.grain, self.date, 'sma')] = sma_pred
        ## Weighted Moving Average prediction
        wma_pred = ma_predictor.get_wma()
        global_variable.global_units_index[(self.grain, self.date, 'wma')] = wma_pred
        ## Exponential Weighted Moving Average prediction
        ewma_pred = ma_predictor.get_ewma()
        global_variable.global_units_index[(self.grain, self.date, 'ewma')] = ewma_pred

        return sma_pred, wma_pred, ewma_pred



