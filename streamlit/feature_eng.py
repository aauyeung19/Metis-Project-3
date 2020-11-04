"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

Author: @Andrew Auyeung andrew.k.auyeung@gmail.com
Location: Metis-Project-3/lib/
Dependencies: cleaning.py
Functions in this module are used to prepare cleaned data for classification.
Features are engineered to determine how they change in the leading rows.
"""

import sys
print(sys.version)

import pandas as pd
import numpy as np
import pickle

    
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

def ma_shifts(window, lags, wdf, col_name):
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
    ma_df = ma_df.assign(**{col_err+'_lag_'+str(lag_n): ma_df[col_err].shift(lag_n) for lag_n in lags}) 
    return ma_df

def leading_trends(window, lags, wdf, col_name):
    """
    Generates the leading trend based on the number of days lagged

    Similar to the ma_shifts method with minor change.  
    Calculates the rolling average based on windowsize.
    Generates the lagged differences between the target and lags
    Sums the differences to get the trend of the change in the last 
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
    ma_df = ma_df.assign(**{col_err+'_lag_'+str(lag_n): ma_df[col_err].shift(lag_n) for lag_n in lags}) 
    return ma_df


