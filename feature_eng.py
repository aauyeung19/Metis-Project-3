import sys
print(sys.version)

import pandas as pd
import numpy as np
import pickle
import cleaning

query = "SELECT date, temp_avg, dp_avg, humid_avg, ws_avg, press_avg, precip FROM daily ORDER BY date"
wdf = cleaning.get_df_from_sql(query)


wdf = pd.read_pickle('src/EWRweather.pickle')

humdf = wdf[['date', 'humid_avg']]
humdf = humdf.set_index('date')
humdf.shift(-1)
humdf.reset_index(inplace=True)
humdf.rename(columns={'humid_avg': 'humid_avg_lag1'}, inplace=True)

wdf.merge(humdf, on='date')
def prep_wdf():
    ###################### Change the pickle to the clenaed pickle name
    wdf = pd.read_pickle('src/EWRweather.pickle')
    wdf.set_index('date')
    lags = wdf.drop(columns='raining')
    lags = lags.shift(1)
    wdf.raining.merge(lags, left_index=True, right_index=True)

    
def set_precip_level(thresh):
    """
    Sets the threshold on precipitation level to be considered rain
    args:
        df : (DataFrame) df with precipitation level
        thresh : (float) Sets the decimal threshold for how many inches of precipitation to consider it raining

    returns:
        df : (DataFrame) with new column with 1 or 0 for rain or no
    """

    wdf['raining'] = wdf.precip.map(lambda x: int(x > thresh))

def difference(dataset, lag=1):
    d = [(dataset[i] - dataset[i-lag]) for i in range(lag, len(dataset))]
    return d

def rolling_difference_mean(dataset, window):
    """
    Creates rolling difference between the current value and the mean
    """
    return dataset-dataset.rolling(window=window).mean()

def ma_shifts(window, l, wdf, col_name):
    """
    Calculates the rolling average based on windowsize.
    Generates the lagged differences between the target and lags

    args: 
        window (int): Size of window to take moving average
        l (int): Days of lag
        wdf (DataFrame): weather dataframe
        col_name (str): name of column to create lag differences
    returns: 
        ma_df (DataFrame): Dataframe with new lagged features
    """
    ma_df = wdf[[col_name]]
    # create rolling average
    roll_col = col_name + '_roll_' + str(window)
    ma_df.loc[:,roll_col] = ma_df.loc[:, col_name].rolling(window=window).mean()
    col_err = col_name + '_error'
    ma_df[col_err] = ma_df[col_name] - ma_df[roll_col]
    # get diff
    # lag columns 
    lags = range(1,l)
    # columns_to_lag = [col_err]
    ma_df = ma_df.assign(**{col_err+'_lag_'+str(lag_n): ma_df[col_err].shift(lag_n) for lag_n in lags}) 
    # ma_df.drop(columns = [col_err], inplace=True)
    return ma_df

# newtemp = Yesterday's rolling(10) average + C1 * (Lag1 error) + C2*(lag2 error)
# n day diff
# n day rolling average minus prev day val
# Convert temp into Kelvin
# Use Autoregression on humidity time series to predict the next humidity
# features would be Humidity rolling diff, predicted humidity from trend

