# Project 3: Classification
## Objective:
Classify days as if it were to rain or not on the next day. 

## Methodology
The goal of the classification was to predict if it were to rain using given information that could have been collected from today and previous dates. 
To accomplish this, I played around with lagged features from the dataset to inform my model.  This included one day deltas and 3 day moving averages.  
As I tuned my classifiers, different models would perform better at different metrics.  As an example: Logistic Regression had great recall but poor precision whereas XGBoost yielded opposite results.  After tuning the separate models, I landed on modelling the content using a soft voting ensemble to group together the probabilities for classification. To check my classifier's attempt, my Streamlit app allows for a comparison of my seven day forecast against the seven day forecast of OpenWeatherMap.org.  

## Data:
Daily Summaries from Weather Underground from Newark Liberty International Airport from 1990 to 2020
NOAA Climate report to cross reference rainy days

## Technologies:
BeautifulSoup and Selenium webscraping
Imblearn - SMOTE
Sklearn - KNearestNeighbors, Logistic Regression, Random Forest, XGBoost, Ensemble
Streamlit

## Summary
Overall, my voting classifier predicts results that are similar to existing forecasting models.  Although I trained my model only on data from Newark, NJ, the preditions were still close to that of OpenWeatherMap with matching predictions for about 5/7 days.  
