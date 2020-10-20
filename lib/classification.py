"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

Author: @Andrew Auyeung


10/19/2020 - baseline still performing better than MA.   Need to fine tune Lags and Window
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import cleaning as cl
import feature_eng as fe
from imblearn.over_sampling import RandomOverSampler

def baseline_X_y():
    # query = "SELECT * FROM daily WHERE date>'2014-01-01' ORDER BY date;"
    # wdf = cl.get_df_from_sql(query)
    wdf = cl.get_cleaned_df()
    # wdf = cl.set_precip_level(wdf, 0)

    # set aside 2019 and 2020 data as holdout sets 
    wdf = wdf[wdf.year<2019].copy(deep=True)

    wdf.set_index('date', inplace=True)
    X = wdf.drop(columns=['raining']).shift(1)
    X.dropna(inplace=True)
    X = X[['temp_avg', 'ws_avg', 'press_avg', 'humid_avg', 'year', 'month']]
    y = wdf.raining[1:]
    assert len(X) == len(y), "X and y are different lengths"
    return X, y


def elbow_knn(X, y, n_max, rs=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rs)
    error_rate = []
    for n in range(1, n_max):
        knn = Pipeline([
            ('scaler', StandardScaler()), 
            ('knn', KNeighborsClassifier(n_neighbors=n, n_jobs=-1))
            ])
        knn.fit(X_train, y_train)
        curr_pred = knn.predict(X_test)
        error_rate.append(np.mean(curr_pred != y_test))
    plt.figure(figsize=(20,10))
    plt.plot(range(1, n_max), error_rate, marker='o', markersize=8)

def baseline_classifiers(X_train, y_train, X_test, y_test, threshold = 0.5):
    ####################################
    # What is best practice for storing this first run?
    ####################################
    """
    First Classification of data with wdf = fe.feat_eng_v1()

    args: 
        X_train, y_train (array): Train dataset
        X_test, y_test (array): Test dataset
    """
    # All Predictions is Raining 
    rain_preds = np.ones(len(X_test))

    # K Nearest Neighbors Baseline
    knn = Pipeline(steps=[('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=37, n_jobs=-1))])
    
    knn.fit(X_train, y_train)
    # knn_preds = knn.predict(X_test)
    knn_proba = knn.predict_proba(X_test)[:,1]
    knn_preds = (knn_proba > threshold).astype(int)

    # Log Regression Baseline
    logreg = LogisticRegression(penalty='none')
    logreg.fit(X_train, y_train)
    logreg_proba = logreg.predict_proba(X_test)[:,1]
    # logreg_preds = logreg.predict(X_test)
    logreg_preds = (logreg_proba > threshold).astype(int)

    # Random Forest Baseline
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_train, y_train)
    # rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:,1]
    rf_preds = (rf_proba > threshold).astype(int)

    # XGBoost
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    # xgb_preds = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:,1]
    xgb_preds = (xgb_proba > threshold).astype(int)

    print('Baseline confusion matrix: \n', confusion_matrix(y_test, rain_preds),'\n',classification_report(y_test, rain_preds))
    print('knn confusion matrix: \n',confusion_matrix(y_test, knn_preds),'\n',classification_report(y_test, knn_preds))
    print('logreg confusion matrix: \n',confusion_matrix(y_test, logreg_preds),'\n', classification_report(y_test, logreg_preds))
    print('random forest matrix: \n',confusion_matrix(y_test, rf_preds),'\n', classification_report(y_test, rf_preds))
    print('XGBoost matrix: \n',confusion_matrix(y_test, xgb_preds),'\n', classification_report(y_test, xgb_preds))

    
    fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_proba, pos_label=1)
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test, logreg_proba, pos_label=1)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba, pos_label=1)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba, pos_label=1)
    plt.plot(fpr_knn, tpr_knn, label='Knn')
    plt.plot(fpr_logreg, tpr_logreg, label='LogReg')
    plt.plot(fpr_rf, tpr_rf, label='Random Forest')
    plt.plot(fpr_xgb, tpr_xgb, label='XGBoost')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

def knn_cv(X_train, y_train, cv=5):
    """
    Fits and trains the KNeighborsClassifier with a GridSearchCV
    args:
        X_train (array): Train dataset with Features
        y_train (array): Train Target
        cv (CrossVal): Default 5 Fold CrossValidation
    returns:
        model (estimator): best estimator with highest recall
    """
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
    params = [{'knn__n_neighbors': range(2,50), 'knn__p': [1, 2]}]

    model = GridSearchCV(pipe, params, cv=cv, n_jobs=-1, scoring='recall')
    model.fit(X_train, y_train)
    return model

def logreg_cv(X_train, y_train, cv=5):
    """
    Fits and trains a Logistic Regression model with GridSearchCV
    args:
        X_train (array): Train dataset with Features
        y_train (array): Train Target
    returns:
        model (estimator): best estimator with highest recall
    """
    logreg = LogisticRegression()
    params = [{'penalty': ['l1', 'l2', 'elasticnet', 'none'],
           'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
    model = GridSearchCV(logreg, params, cv=cv, n_jobs=-1, scoring='recall')
    model.fit(X_train, y_train)
    return model

def rf_cv(X_train, y_train, cv=5):
    """
    Fits and trains a Random Forest model with GridSearchCV
    args:
        X_train (array): Train dataset with Features
        y_train (array): Train Target
    returns:
        model (estimator): best estimator with highest recall
    """
    rf = RandomForestClassifier()
    params = [{'n_estimators': range(50, 550, 50), \
            'max_depth': [5, 10, None], \
            'min_samples_split': [2, 10, 20], \
            'max_features': ['sqrt', 8, 10, 12], \
            'criterion': ['gini']}]
    model = GridSearchCV(rf, params, cv=cv, n_jobs=-1, scoring='recall', verbose=True)
    model.fit(X_train, y_train)
    return model


if __name__ == '__main__':

    # Run simple Linear Regression on each ma result. 
    # Use Linear Model to predict true temp.
    # Merge Linear Model Predictions back to date df. 
    
    wdf = fe.feat_eng_v2()
    # set aside 2019 and 2020 data as holdout sets 
    holdout = wdf[wdf.year>=2019].copy(deep=True)
    X_holdout = holdout.drop(columns=['raining'])[:-1]
    y_holdout = holdout.raining[1:]

    wdf = wdf[wdf.year<2019].copy(deep=True)
    X = wdf.drop(columns=['raining'])[:-1]
    y = wdf.raining[1:]

    baseline = False # to test baseline
    if baseline:
        X, y = baseline_X_y()

    assert len(X)==len(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=None)

    ros = RandomOverSampler(random_state=None)
    X_train, y_train = ros.fit_sample(X_train,y_train)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    est = rf_cv(X_train, y_train, cv)

    print(confusion_matrix(y_test, (est.predict_proba(X_test)[:,1]>0.5).astype(int)))
    print(classification_report(y_test, (est.predict_proba(X_test)[:,1]>0.5).astype(int)))
    