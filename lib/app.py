import streamlit as st
import pickle
import pandas as pd
import feature_eng as fe
import owm
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates
import datetime
#open model


def load_models():
    v = 5
    vote = pickle.load(open('../models/Version 5/vote.pickle', 'rb'))
    knn = pickle.load(open(f'../models/Version {v}/knn.pickle', 'rb'))
    logreg = pickle.load(open(f'../models/Version {v}/logreg.pickle', 'rb'))
    rf = pickle.load(open(f'../models/Version {v}/rf.pickle', 'rb'))
    xgb = pickle.load(open(f'../models/Version {v}/xgb.pickle', 'rb'))
    model_dict = {'Voting': vote, 'KNearestNeighbors': knn.best_estimator_, 'LogisticRegression': logreg.best_estimator_, 'Random Forest': rf.best_estimator_, 'XGBoost': xgb.best_estimator_}
    return model_dict


train_cols = ['temp_avg', 'ws_avg', 'press_avg', 'humid_avg', 'dp_avg', 'under_dp',
    'temp_kelvin', 'month', 'PRCP', 'SNOW', 'lat', 'lon', 'distance',
    'distance_delta', 'press_delta', 'humid_delta', 'ws_delta',
    'temp_trend', 'humid_trend', 'month_1', 'month_2', 'month_3', 'month_4',
    'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10',
    'month_11', 'month_12']
if __name__=="__main__":
    models = load_models()



    st.title('Will it Rain Tomorrow?')

    # Select Location
    loc_dict = {
        'Newark': [40.6895, -74.1745], 
        'Chicago': [41.9742, -87.9073], 
        'Raleigh': [35.8801, -78.7880],
        'Seattle': [47.6062, -122.3321],
        'Los Angeles': [34.0522, -118.2437],
        'Atlanta': [33.7490, -84.3880],
        'Washington DC': [38.9072, -77.0369],
        'Dallas': [32.7767, -96.7970],
        'San Francisco': [37.7749, 122.4194]}
    location = st.sidebar.selectbox('Pick a location', list(loc_dict.keys()))
    curr_loc = loc_dict[location]
    lat, lon = curr_loc[0], curr_loc[1]
    df = owm.get_owm(lat, lon)

    st.write(
        f"Location: {location}"
    )
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

    # Selection of Models 
    selection = st.sidebar.selectbox('Select an Estimator', list(models.keys()))
    model=models[selection]
    preds = model.predict(df[train_cols].dropna())[-7:]
    proba = model.predict_proba(df[train_cols].dropna())[-7:,1]
    
    pred_type = st.sidebar.selectbox('How would you like the model to show its prediction?', ['Prediction', 'Probability'])
    fig = plt.figure()
    if pred_type == 'Prediction':
        plt.plot(df['date'].iloc[6:], preds, label='Prediction')
    else:
        plt.plot(df['date'].iloc[6:], proba, label='Prediction Chance')
    plt.plot(df['date'].iloc[6:], df['pop'].tail(7), label='Forcast')
    plt.ylim(0,1.1)
    plt.xlabel('Date')
    plt.ylabel('Chance of Rain')
    plt.xticks(rotation=90)
    plt.legend()

    st.pyplot(fig)

    # Select Location from List
    # FSM: Ping API- Assume no Storm
    # Compare Predictions to Current 7 day forcast
    # After add Lat Long sliders for location of storm