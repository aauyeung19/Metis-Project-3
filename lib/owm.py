"""
Author: @Andrew Auyeung

Queries openweathermap.org API to collect 
historical 5 day data and 7 day forcast

API Key:
98ec6864b86b42efec56dc8a1b9abcef

Documentation: 
https://openweathermap.org/api/one-call-api
"""

# Historical API: 
# https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={time}&appid={API key}
# Get Forcast:
# https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude={part}&appid={API key}
import requests
import pandas as pd
import numpy as np
import cleaning as cl
import feature_eng as fe
import datetime as dt

# Get current date
# Loop through date - date-5
ewr = (40.6895, -74.1745)
lat = ewr[0]
lon = ewr[1]

# get current date
now = dt.datetime.now()

#start empty dataframe
owm_df = pd.DataFrame()

#### Get History
# Loop through 5 previous days
for days in range(1,6):
    curr_day = int((now - dt.timedelta(days)).timestamp())
    url = f'https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={curr_day}&units=imperial&appid=98ec6864b86b42efec56dc8a1b9abcef'

    
    page = requests.get(url)

    day_dict = page.json()

    day = pd.DataFrame(day_dict['current'])
    owm_df = pd.concat([owm_df, day], ignore_index=True)

owm_df.drop(columns='wind_gust', inplace=True)
#### Get Forcast

url = f'https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,alerts&units=imperial&appid=98ec6864b86b42efec56dc8a1b9abcef'
page = requests.get(url)
forcast = page.json()

# Add Today's data
owm_df = pd.concat([owm_df, pd.DataFrame(forcast['current'])], ignore_index=True)

forcast = pd.DataFrame(forcast['daily'])
forcast.drop(columns='feels_like')
forcast.temp = [t['day'] for t in forcast.temp]
forcast.feels_like = [f['day'] for f in forcast.feels_like]
owm_df = pd.concat([owm_df, forcast], ignore_index=True)

owm_df.drop(columns='weather', inplace=True)
owm_df.drop_duplicates(inplace=True)

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
owm_df.head()
# Clean data
owm_df = cl.parse_month_year(owm_df)
owm_df['temp_kelvin'] = cl.convert_to_kelvin(owm_df.temp_avg)
owm_df.sort_values(by='date', ascending=True, inplace=True)

owm_df['press_delta'] = owm_df.press_avg.diff()
owm_df['humid_delta'] = owm_df.humid_avg.diff()
owm_df['ws_delta'] = owm_df.ws_avg.diff()

owm_df['temp_trend'] = fe.ma_shifts(3, range(1,5), owm_df, 'temp_kelvin').iloc[:,2:].sum(axis=1)
owm_df['humid_trend'] = fe.ma_shifts(3, range(1,5), owm_df, 'humid_avg').iloc[:,2:].sum(axis=1)
owm_df['PRCP'].fillna(0, inplace=True)
owm_df['pop'].fillna(0, inplace=True)
owm_df['SNOW'] = np.zeros(len(owm_df))
owm_df['under_dp'] = (owm_df['temp_avg'] <= owm_df['dp_avg']).astype(int)
owm_df.drop(columns=['day', 'visibility'], inplace=True)

# insert under_dp
if __name__=='__main__':
    pass