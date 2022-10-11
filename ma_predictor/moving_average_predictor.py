import numpy as np

class MovingAveragePredictor:

    def __init__(self, sma_units, wma_units, ewma_units):
        self.sma_units = sma_units
        self.wma_units = wma_units
        self.ewma_units = ewma_units

    def get_sma(self):
        return np.mean(self.sma_units)

    def get_wma(self):
        window_size = len(self.wma_units)
        win_index = 0
        window_avg = 0
        for unit in self.wma_units:
            window_avg += (window_size-win_index)*unit
            win_index += 1
        window_avg /= (window_size*(window_size+1)/2)
        return window_avg

    def get_ewma(self):
        window_size = len(self.ewma_units)
        alpha = 2/(window_size+1)
        win_index = 0
        ewa_num = 0
        ewa_denom = 0
        for unit in self.ewma_units:
            ewa_num += (((1 - alpha)**win_index) * unit)
            ewa_denom += ((1 - alpha)**win_index)
            win_index += 1
        ewa_avg = ewa_num/ewa_denom
        return ewa_avg


