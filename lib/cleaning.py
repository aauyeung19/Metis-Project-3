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
import csv
import math


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
    noaa = pd.read_csv('../src/NOAA_EWR.csv')
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

def convert_lat_long(l):
    """
    Converts latitude and longitude to decimal
    """
    l = str(l)
    l = l.strip()
    if 'S' in l or 'W' in l:
        mod = -1
    else:
        mod = 1
    return mod*float(l[:-1])


def haversine(lat,lon):
    """
    Returns the distance of the lat/long coord from EWR
    using the haversine formula
    args: 
        lat (float): latitude of storm
        long (float): longitude of storm
    returns:
        distance (float): distance from EWR in kilometers
    """
    # Coordinates of EWR
    ewr_coord = (40.6895, -74.1745)
    # approximate radius of earth in km
    R = 6373.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.radians(ewr_coord[0])
    lon2 = math.radians(ewr_coord[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def clean_hurr():
    """
    Reads data from the Northeast and North Central Pacific hurricane database 
    Note: raw data from the database contains multiple tables in one CSV
    args:
        None
    returns:
        hurr (DataFrame): cleaned dataframe
    """

    # Read hurricane data
    hurr = convert_hurr_to_df('../src/hurdat2-1851-2019-052520.txt')

    hurr.lat = hurr.lat.apply(convert_lat_long)
    hurr.lon = hurr.lon.apply(convert_lat_long)
    # hurr = hurr.groupby(['basin', 'name', 'date','status'], as_index=False).mean()
    hurr.date = hurr.date.map(lambda d: d[:4] + '-' + d[4:6] + '-' + d[6:])
    hurr.date = pd.to_datetime(hurr.date)
    hurr['distance'] = [haversine(x,y) for x,y in zip(hurr.lat, hurr.lon)]

def convert_hurr_to_df(path):
    """
    Function to create a dataframe from HURDAT2
    args:
        path(str): relative path to .txt file
    returns:
        df(dataframe): Dataframe file
    """
    with open(path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    
    df_list = []
    for row in data:
        if len(row)==4:
            basin = row[0]
            name = row[1].strip()
        else:
            curr_row = dict(zip(['date', 'time', 'record_id', 'status', 'lat', 'lon', 'max_ws', 'min_press'], row[:8]))
            curr_row['basin'] = basin
            curr_row['name'] = name
            df_list.append(curr_row)
    
    df = pd.DataFrame(df_list)
    return df


if __name__ == "__main__":

    print('This is the cleaned DataFrame')
    wdf = get_cleaned_df()
    print(wdf.head())

    noaa = pd.read_csv('../src/NOAA_EWR.csv')
    noaa.DATE = pd.to_datetime(noaa.DATE)
    noaa.rename(columns={'DATE':'date'}, inplace=True)
    noaa = parse_month_year(noaa)
    noaa = noaa[noaa.year>=1990]

