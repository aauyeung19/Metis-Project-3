import streamlit as st
import pickle
import pandas as pd
import feature_eng as fe
import owm
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates
import datetime
import seaborn as sns


#open model

######### To Do: ############
# Write load all models and predict for all possible locations given
# Heirarchy of Dictionaries:
# Location
# ....Model
# ........Predict
# ........Proba
# ....True
#
# example: cached_dict['Newark']['knn']['Predict'] should return predictions from the knn model
#####
# Wishlist:
# .. add sunrise and sunset times per day 
# .. widget for current conditions
# .. Add storm to update storm data

@st.cache(allow_output_mutation=True)
def load_models():
    v = 5
    vote = pickle.load(open('../models/Version 5/vote.pickle', 'rb'))
    knn = pickle.load(open(f'../models/Version {v}/knn.pickle', 'rb'))
    logreg = pickle.load(open(f'../models/Version {v}/logreg.pickle', 'rb'))
    rf = pickle.load(open(f'../models/Version {v}/rf.pickle', 'rb'))
    xgb = pickle.load(open(f'../models/Version {v}/xgb.pickle', 'rb'))
    model_dict = {'Voting': vote, 'KNearestNeighbors': knn.best_estimator_, 'LogisticRegression': logreg.best_estimator_, 'Random Forest': rf.best_estimator_, 'XGBoost': xgb.best_estimator_}
    
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
    
    train_cols = ['temp_avg', 'ws_avg', 'press_avg', 'humid_avg', 'dp_avg', 'under_dp',
        'temp_kelvin', 'month', 'PRCP', 'SNOW', 'lat', 'lon', 'distance',
        'distance_delta', 'press_delta', 'humid_delta', 'ws_delta',
        'temp_trend', 'humid_trend', 'month_1', 'month_2', 'month_3', 'month_4',
        'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10',
        'month_11', 'month_12']

    data = {}
    for curr_loc in loc_dict:
        lat, lon = loc_dict[curr_loc][0], loc_dict[curr_loc][1]
        # Ping API
        df = owm.get_owm(lat, lon)
        # Add dummies
        df[['month_1', 'month_2', 'month_3', 'month_4', 'month_5', 
            'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 
            'month_11', 'month_12']] = 0
        for each in df.month:
            df['month_'+str(each)] = 1
        # Assume no current storm
        df[['lat', 'lon', 'distance', 'distance_delta']] = -9999
        # Add all predictions and probabilities to the dictionairy
        # Keys are the names of the models
        preds_dict = {}
        for model in model_dict:
            preds = model_dict[model].predict(df[train_cols].dropna())[-7:]
            proba = model_dict[model].predict_proba(df[train_cols].dropna())[-7:,1]
            preds_dict[model] = {'preds': preds, 'proba': proba}
        # Add predictions and probabilities to current location key in Dictionairy 
        data[curr_loc] = preds_dict
        data[curr_loc]['actual'] = df
    
    return data




if __name__=="__main__":
    
    cached_data = load_models()

    st.title('Seven Day Weather Forecast')

    # Select Location
    locations = ['Newark', 'Chicago', 'Raleigh', 'Seattle', 'Los Angeles', 'Atlanta', 'Washington DC', 'Dallas', 'San Francisco']
    loc = st.selectbox('Pick a location', locations)
    df = cached_data[loc]['actual'].copy(deep=True)
    df.sort_values(by='date', inplace=True)
    df.date = df.date.map(lambda x: x.day_name())
    # Selection of Models 
    models = ['Voting', 'KNearestNeighbors', 'LogisticRegression', 'Random Forest', 'XGBoost']

    show_all = st.checkbox('Show All Estimators', value=True)
    if show_all:
        all_df = pd.DataFrame()
        for m in models:
            curr_df = pd.DataFrame(zip(df['date'].iloc[6:], cached_data[loc][m]['proba']), columns=['Date', 'Probability'])
            curr_df['Model'] = m
            all_df = all_df.append(curr_df, ignore_index=True)

        # all_df.sort_values(by='Date', inplace=True)
        # all_df.Date=all_df.Date.map(lambda x: x.day_name())
        fig = sns.catplot(x='Date', y='Probability', data=all_df, kind='strip', hue='Model', legend=False, palette='Set2', height=5, aspect=1.5)
        plt.plot(df['date'].iloc[6:], df['pop'].tail(7), label='Forecast', color='lightcoral', linewidth=4)
        sns.scatterplot(df['date'].iloc[6:], cached_data[loc]['Voting']['preds'], label='Prediction', marker='o', s=150)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title('Comparisons of Models against Prediction from OpenWeatherMap', fontsize=16)
        plt.legend(bbox_to_anchor=(1,0.7))
        plt.ylim((0,1.1))

        st.pyplot(fig)

    else:
        model = st.selectbox('Select an Estimator', models)
        preds = cached_data[loc][model]['preds']
        proba = cached_data[loc][model]['proba']

        pred_type = st.selectbox("How would you like to show the model's prediction?", ['Prediction', 'Probability'])
        fig = plt.figure()
        if pred_type == 'Prediction':
            plt.plot(df['date'].iloc[6:], preds, label='Prediction', linewidth=4)
        else:
            plt.plot(df['date'].iloc[6:], proba, label='Prediction Chance', linewidth=4)
        plt.plot(df['date'].iloc[6:], df['pop'].tail(7), label='Forecast')
        plt.ylim(0,1.1)
        plt.xlabel('Date')
        plt.ylabel('Chance of Rain')
        plt.xticks(rotation=90)
        plt.title(f'Prediction of Rain in {loc} Using {model} Estimator')
        plt.legend(bbox_to_anchor=(1,0.7))

        st.pyplot(fig)

