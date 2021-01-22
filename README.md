# RainOne - A classification task to predict weather

## Abstract:
Have you ever looked at the weather forecast and had the question: what does 50% chance rain actually mean?
Does it mean that it will cover 50% of the location specified?  Would a frequentist say that if we repeated the next day 100 times, 50 of those days would rain and 50 would not? 
A quick google search brings up some conflicting information.  [ScienceNotes.org](https://sciencenotes.org/percent-chance-rain-mean) describes the probability of precipitation as a product of how sure the forecaster is of rain and the forecast area.  On the other hand, [Weather.gov](https://www.weather.gov/ffc/pop) states that probability of precipitation *"simply describes the probability that the forecast grid/point in question will receive at least 0.01" of rain."* 
This project uses classification to predict rainy events based on historical weather data focusing on climate data scrapped from Wunderground.  You can see it in action [here](https://drive.google.com/file/d/1CMlts_MTHpOTi7iND1qdTGP0xNIgBAr7/view?usp=sharing). 

![[Streamlit Video Demo of rainOne](https://github.com/aauyeung19/rainOne/blob/main/Visualizations/Streamlit_screenshot.png)](https://drive.google.com/file/d/1CMlts_MTHpOTi7iND1qdTGP0xNIgBAr7/view?usp=sharing)
*The solid blue line represents the predictions from my model where the solid orange line represents the weather prediction of OpenWeatherMap.org*

## Motivation:
Traditionally you would anticipate a forecast to be using a time series analysis or ARIMA model so why do I use classification here?  I wanted one model that would work accurately for the entire year regardless of when the model would be used.  Weather data is also inheritely seasonal which makes it something I need to adjust for. In addition, it seems like the simplest solution considering the amount of weather services that have their entire forecasting system completed.  I decided I could leverage their predictions and 5 day histories to give my own "probability of precipitation."  

## Technologies:
* Selenium Webscraping
* PostgreSQL
* SciKit Learn -- Ensemble, KNN, LogReg, Random Forest, XGBoost
* imblearn -- SMOTE, Pipelines
* Streamlit

## Methodology
### Data
To train my models, I used daily weather summaries scraped from Weather Underground focusing on information from one weather station at Newark International Airport (EWR) from 1990 to 2020. That data included indicators like daily humidity, average windspeed, and average pressure.  
Each event also recorded the amount of precipitation at that day.  As my target variable, I converted that column into a binary target to help with my classification.  Rain on a parade is still rain on a parade no matter how much comes down.  As a baseline, I lagged every feature one day to use as the predictors for precipitation.  In other words, my baseline predicted if it would rain today using conditions that I see from only today.  
### Feature Engineering
To improve the predictions, I calculated the deltas of the leading features to see how much yesterday's features changed from the day before.  Did the humidity increase? decrease?  Does this affect the probability of rain? 
One roadblock I stumbled into was dealing with imbalanced classes.  There were vastly larger amounts of dry days compared to rainy days.  To overcome this, I used SMOTE to oversample the data.  It was important to note that I used imblearn's pipeline instead of scikit learn's implementation.  Imblearn's pipeline will apply the oversampling to the train data but not the test data when cross validating.  
### Model
Tuning the model was another headache I had to overcome.  I wanted the model to be better at recall because I'd rather plan my day around the possibility of rain.  If I planned my day around a forecast, I'd rather the day be a false positive over a false negative.  Ultimately, the F-beta over 1 would shift the predictions too much to recall so I stayed with the base F1 score.  
My alternative to dealing with the events was to ensemble different algorithms together that were successfull in different things.  Logistic Regression had great recall but poor precision whereas XGBoost yielded opposite results. 
![Confusion Matrixies for XGBoost and LogReg](https://github.com/aauyeung19/rainOne/blob/main/Visualizations/WeatherDeck3b.png?raw=true)
At the end of this project, I had trained four models (Logistic Regression, Random Forest, KNearestNeighbors, and XGBoost) and ensembled them together using soft voting.  I placed a slightly higher weight on LogReg to still try to push the model towards recall.  
<p align="center">
  <img src="https://github.com/aauyeung19/rainOne/blob/main/Visualizations/WeatherDeck3a.png?raw=true">
</p>

### Implementation
I pushed the model to a local webapp using Streamlit to compare its prediction against existing forecasts.  I chose to compare my model against [OpenWeatherMap](https://www.openweathermap.org).  My app would ping the API for five day historical weather features and their 7 day forecast.  The app would use the historical data as predictors for the subsequent days.  What was nice was that my model could output both hard predictions but also its own probability of precipitation.  In this case, it would only mean the probability of rain using past conditions.  What was surprising was that although I trained the model on data from EWR, it predicted similar to 5/7 days on locations with different geographic features.  

## Next Steps
I would like to revisit this with a more robust feature selection. I only used a one day lag to check the change in conditions from the prior two days.  If I had more time I would like to investigate which lags would have the strongest importance.  If yesterday's pressure is above the rolling average of 5 days, would it be a strong indicator? 
