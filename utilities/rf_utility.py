
import math

def split_dataset(inp_df):
    grain_index = inp_df.index
    train_len = math.floor(len(grain_index)  * 0.9)
    train_index = grain_index[:train_len]
    val_index = grain_index[train_len:]
    train_data = inp_df.loc[train_index,:]
    val_data = inp_df.loc[val_index,:]

    train_x = train_data.drop(columns = ['units'])
    train_y = train_data['units']
    val_x = val_data.drop(columns = ['units'])
    val_y = val_data['units']
    return train_x, train_y, val_x, val_y

def create_individual_date_cols(ts):
    # converting to datetime
    ts['day'] = ts.date.dt.day
    ts['wday'] = ts.date.dt.weekday
    ts['year'] = ts.date.dt.year
    ts['month'] = ts.date.dt.month
    ts['week'] = ts.date.dt.isocalendar().week
    return ts