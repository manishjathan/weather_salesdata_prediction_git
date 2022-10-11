
from datetime import timedelta
from utilities import global_variable
import pandas as pd

def get_window_dates(date, window_size):
    return [date - timedelta(days = window) for window in range(1,window_size+1)]

def get_input_units(grain, grain_data, date, window_size, algo):
    window_dates = get_window_dates(date, window_size)
    required_units = []
    history_count = 0
    predicted_count = 0

    for window_date in window_dates:
        if window_date in grain_data.date.values:
            unit = grain_data[grain_data['date'] == window_date]['units'].values[0]
            if window_date not in global_variable.global_units_index.keys():
                global_variable.global_units_index[(grain, window_date, algo)] = unit
            history_count += 1
            required_units.append(unit)
        else:
            unit = global_variable.global_units_index[(grain, window_date, algo)]
            predicted_count +=1
            required_units.append(unit)
    #print("Number of points from history : ",history_count)
    #print("Number of points from prediction : ",predicted_count)
    return required_units