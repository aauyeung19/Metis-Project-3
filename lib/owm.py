"""
Author: @Andrew Auyeung

Queries openweathermap.org API to collect 
historical 5 day data and 7 day forecast

Documentation: 
https://openweathermap.org/api/one-call-api
"""

import requests
import pandas as pd
import numpy as np
import feature_eng as fe
import datetime as dt
import pickle

api_key = pickle.load(open('api_key.pickle', 'rb'))
# Get current date
# Loop through date - date-5

def parse_month_year(df):
    """
    Parse date to Month and Years
    """
    df['year'] = df.date.map(lambda x: x.year)
    df['month'] = df.date.map(lambda x: x.month)
    return df

def convert_to_kelvin(temp_col):
    temp_col = temp_col.map(lambda x: (x - 32) * 5/9 + 273.15)
    return temp_col


def get_owm(lat,lon):
    """
    Gets 5 day history and 7 day forecast from OpenWeatherMap.org
    args:
        lat (float): Latitude
        lon (float): Longitude
    returns:
        owm_df (DataFrame): DataFrame with OpenWeatherMap data
    """
    # get current date
    now = dt.datetime.now()

    #start empty dataframe
    history_list = []

    #### Get History
    # Loop through 5 previous days
    for days in range(1,6):
        curr_day = int((now - dt.timedelta(days)).timestamp())
        url = f'https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={curr_day}&units=imperial&appid={api_key}'
        
        page = requests.get(url)

        day_dict = page.json()
        today = day_dict['current']
        history_list.append(today)
    
    # owm_df = pd.DataFrame(history_list)

    #### Get Forecast

    url = f'https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,alerts&units=imperial&appid={api_key}'
    page = requests.get(url)
    forecast = page.json()
    
    # Add today's data
    history_list.append(forecast['current'])
    owm_df = pd.DataFrame(history_list)
    owm_df['rain'] = np.zeros(len(owm_df))
    # Add Forecast
    
    forecast = forecast['daily']
    forecast = pd.DataFrame(forecast)
    forecast.temp = [t['day'] for t in forecast.temp]
    forecast.feels_like = [f['day'] for f in forecast.feels_like]
    owm_df = pd.concat([owm_df, forecast], ignore_index=True)

    owm_df.drop(columns='weather', inplace=True)
    try:
        owm_df.drop_duplicates(inplace=True)
    except:
        pass
    
    # Clean owm_df
    owm_df.dt = owm_df.dt.map(dt.datetime.fromtimestamp)
    owm_df.pressure = owm_df.pressure.map(lambda x: x/33.86)

    # rename columns to match columns from model
    owm_df.rename(
        columns={
            'dt':'date',
            'temp': 'temp_avg',
            'pressure': 'press_avg',
            'humidity': 'humid_avg',
            'dew_point': 'dp_avg',
            'wind_speed': 'ws_avg',
            'rain': 'PRCP'
        },
        inplace=True)
    owm_df['day'] = owm_df.date.map(lambda x: x.day)
    owm_df.drop_duplicates('day', inplace=True)
    # owm_df.head()
    # Clean data
    owm_df = parse_month_year(owm_df)
    owm_df['temp_kelvin'] = convert_to_kelvin(owm_df.temp_avg)
    owm_df.sort_values(by='date', ascending=True, inplace=True)

    owm_df['press_delta'] = owm_df.press_avg.diff()
    owm_df['humid_delta'] = owm_df.humid_avg.diff()
    owm_df['ws_delta'] = owm_df.ws_avg.diff()

    owm_df['temp_trend'] = fe.ma_shifts(3, range(1,5), owm_df, 'temp_kelvin').iloc[:,2:].sum(axis=1)
    owm_df['humid_trend'] = fe.ma_shifts(3, range(1,5), owm_df, 'humid_avg').iloc[:,2:].sum(axis=1)
    owm_df['press_trend'] = fe.ma_shifts(3, range(1,5), owm_df, 'press_avg').iloc[:,2:].sum(axis=1)
    owm_df['PRCP'].fillna(0, inplace=True)
    owm_df['pop'].fillna(0, inplace=True)
    owm_df['SNOW'] = np.zeros(len(owm_df))
    owm_df['under_dp'] = (owm_df['temp_avg'] <= owm_df['dp_avg']).astype(int)
    owm_df.drop(columns=['day', 'visibility', 'year'], inplace=True)
    
    return owm_df

if __name__=='__main__':
    # Test functions
    ewr = [40.6895, -74.1745]
    rdu = [35.8801, -78.7880]
    lat = float(rdu[0])
    lon = float(rdu[1])
    df = get_owm(lat, lon)
    print('This is the data from RDU')
    df.head()