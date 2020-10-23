import streamlit as st
import pickle
import pandas as pd
import feature_eng as fe
import owm
from collections import defaultdict
import matplotlib.pyplot as plt
#open model
vote = pickle.load(open('../models/Version 5/vote.pickle', 'rb'))

v = 5
knn = pickle.load(open(f'../models/Version {v}/knn.pickle', 'rb'))
logreg = pickle.load(open(f'../models/Version {v}/logreg.pickle', 'rb'))
rf = pickle.load(open(f'../models/Version {v}/rf.pickle', 'rb'))
xgb = pickle.load(open(f'../models/Version {v}/xgb.pickle', 'rb'))

train_cols = ['temp_avg', 'ws_avg', 'press_avg', 'humid_avg', 'dp_avg', 'under_dp',
       'temp_kelvin', 'month', 'PRCP', 'SNOW', 'lat', 'lon', 'distance',
       'distance_delta', 'press_delta', 'humid_delta', 'ws_delta',
       'temp_trend', 'humid_trend', 'month_1', 'month_2', 'month_3', 'month_4',
       'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10',
       'month_11', 'month_12']

st.title('Will it Rain Tomorrow?')
st.write(

# This is a header. 

)
location = st.selectbox('Pick a location', ('EWR', 'ORD', 'RDU'))

loc_dict = {'EWR': (40.6895, -74.1745), 'ORD': (41.9742, -87.9073), 'RDU': (35.8801, -78.7880)}

curr_loc = loc_dict[location]
lat, lon = curr_loc[0], curr_loc[1]
df = owm.get_owm(lat, lon)
# add columns to match train/test
df[['month_1', 'month_2', 'month_3',
    'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
    'month_10', 'month_11', 'month_12']]=0
for each in df.month:
    df['month_'+str(each)] = 1

storm = False
if not storm:
    df['lat'] = -9999
    df['lon'] = -9999
    df['distance'] = -9999
    df['distance_delta'] = -9999

show_plots = st.checkbox('Show Plots', value=True)

if show_plots:
    model=vote
    preds = model.predict(df[train_cols].dropna())[-7:]
    proba = model.predict_proba(df[train_cols].dropna())[-7:,1]
    fig = plt.figure()
    fig = plt.plot(df['date'].iloc[6:], preds, label='Prediction')
    # plt.plot(df['date'].iloc[6:], proba, label='Prediction Chance')
    fig = plt.plot(df['date'].iloc[6:], df['pop'].tail(7), label='Forcast')
    fig = plt.ylim(0,1.1)
    fig = plt.xlabel('Date')
    fig = plt.ylabel('Chance of Rain')
    fig = plt.xticks(rotation=90)
    fig = plt.legend()

    st.write(fig)


st.checkbox('Add Storm?', value=True)

# Select Location from List
# FSM: Ping API- Assume no Storm
# Compare Predictions to Current 7 day forcast
# After add Lat Long sliders for location of storm