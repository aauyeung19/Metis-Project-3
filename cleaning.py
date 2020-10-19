"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

Author: Andrew Auyeung

Methods in this module are used to clean Scraped data from Wunderground and NOAA Request
"""

import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import psycopg2 as pg


def make_lag(df, col_name, lags):
    """
    Lags ONE column
    Takes the column name and adds columns on the end
    of the dataframe depending on the list of lags wanted
    args:
        df (DataFrame) : 
        col_name (str) : Name of column to be lagged
        lags (list) : List of lags desired
    returns:
        df (DataFrame) :  new df with lagged columns
    """
    for l in lags:
        df_to_lag = df[['date', col_name]]
        df_to_lag = df_to_lag.set_index('date')
        df_to_lag = df_to_lag.shift(l, axis=0)
        df_to_lag.reset_index(inplace=True)
        new_name = col_name+'_lag'+str(l)
        df_to_lag.rename(columns={col_name: new_name}, inplace=True)

        df = df.merge(df_to_lag, on='date')
    return df

def set_precip_level(df, thresh):
    """
    Sets the threshold on precipitation level to be considered rain
    args:
        df : (DataFrame) df with precipitation level
        thresh : (float) Sets the decimal threshold for how many inches of precipitation to consider it raining
    returns:
        df : (DataFrame) with new column with 1 or 0 for rain or no
    """
    df['raining'] = df.precip.map(lambda x: int(x > thresh))
    return df

def get_df_from_sql(query):
    """
    Sends query to SQL for information. 
    DEFAULT: Returns ENTIRE Table.
    args: 
        query (str): SQL query 
    returns: 
        dataframe
    """
    connection_args = {
        'host': 'localhost', 
        'dbname': 'weather'
    }
    connection = pg.connect(**connection_args)
    return pd.read_sql(query, connection)

def convert_to_kelvin(temp_col):
    temp_col = temp_col.map(lambda x: (x - 32) * 5/9 + 273.15)
    return temp_col

def parse_month_year(df):
    """
    Parse date to Month and Years
    """
    df['year'] = df.date.map(lambda x: x.year)
    df['month'] = df.date.map(lambda x: x.month)
    return df

def clean_noaa():
    """
    Cleans noaa dataframe for joining with wunderground df
    """
    noaa = pd.read_csv('src/NOAA_EWR.csv')
    noaa.DATE = pd.to_datetime(noaa.DATE)
    noaa.rename(columns={'DATE':'date'}, inplace=True)
    noaa = parse_month_year(noaa)
    noaa = noaa[noaa.year>=1990][['date', 'PRCP', 'SNOW']].copy(deep=True)

    return noaa

def get_cleaned_df():
    """
    Returns a cleaned DataFrame merging NOAA SNOW and PRCP columns with 
    Wunderground data
    """
    query = "SELECT date, temp_avg, ws_avg, press_avg, humid_avg, dp_avg, dp_max, temp_min FROM daily;"
    wdf = get_df_from_sql(query)
    wdf['under_dp'] = (wdf['temp_min'] <= wdf['dp_max']).astype(int)
    wdf['temp_kelvin'] = convert_to_kelvin(wdf.temp_avg)
    wdf = parse_month_year(wdf)
    wdf.date = pd.to_datetime(wdf.date)
    wdf = wdf.merge(clean_noaa(), left_on='date', right_on='date')
    wdf['precip'] = wdf.PRCP + wdf.SNOW
    wdf = set_precip_level(wdf, 0)
    wdf.drop(columns=['dp_max', 'temp_min'], inplace=True)

    return wdf

if __name__ == "__main__":

    print('This is the cleaned DataFrame')
    wdf = get_cleaned_df()
    print(wdf.head())

    noaa = pd.read_csv('src/NOAA_EWR.csv')
    noaa.DATE = pd.to_datetime(noaa.DATE)
    noaa.rename(columns={'DATE':'date'}, inplace=True)
    noaa = parse_month_year(noaa)
    noaa = noaa[noaa.year>=1990]