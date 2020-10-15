"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

if __name__ == "__main__":

    query = "SELECT * FROM daily;"
    wdf = get_df_from_sql(query)



