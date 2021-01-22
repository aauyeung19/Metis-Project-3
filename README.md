# RainOne - A classification task to predict weather

## Abstract:
Have you ever looked at the weather forecast and had the question: what does 50% chance rain actually mean?
Does it mean that it will cover 50% of the location specified?  Would a frequentist say that if we repeated the next day 100 times, 50 of those days would rain and 50 would not? 
A quick google search brings up some conflicting information.  (ScienceNotes.org)[https://sciencenotes.org/percent-chance-rain-mean/#:~:text=Weather%20forecasters%20use%20the%20term,point%20in%20a%20selected%20area.&text=This%20is%20the%20percent%20of,if%20there%20is%20any%20precipitation.] describes the probability of precipitation as a product of how sure the forecaster is of rain and the forecast area.  On the other hand, (Weather.gov)[https://www.weather.gov/ffc/pop] states that probability of precipitation "simply describes the probability that the forecast grid/point in question will receive at least 0.01" of rain." 
This project uses classification to predict rainy events based on historical weather data focusing on climate data scrapped from Wunderground.  You can see it in action (here)[INSERT LINK TO VIDEO HERE]

## Objective:
This project uses classification to predict rainy events based on historical weather data focusing
I set out to attempt to predict whether or not the next day would rain.  
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
