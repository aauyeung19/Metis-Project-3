"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

Author: @Andrew Auyeung
"""
import cleaning as cl
import feature_eng as fe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as imbPipeline

f2scorer = make_scorer(fbeta_score, beta=1.5)

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
    """
    First Classification of data with wdf = fe.feat_eng_v1()

    args: 
        X_train, y_train (array): Train dataset
        X_test, y_test (array): Test dataset
    """

    from sklearn.metrics import roc_curve, confusion_matrix, classification_report

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

    model_dict = {
        'knn': knn, 
        'logreg': logreg, 
        'random forest': rf,
        'xgboost': xgb_model
    }
    return model_dict

def knn_cv(X_train, y_train, cv=5, verbose=False):
    """
    Fits and trains the KNeighborsClassifier with a GridSearchCV
    args:
        X_train (array): Train dataset with Features
        y_train (array): Train Target
        cv (object): Default 5-Fold CrossValidation
        Verbose (bool): True to see verbose GridSearchCV 
    returns:
        model (estimator): best estimator with highest recall
    """
    pipe = imbPipeline(steps=[
        ('sample', SMOTE()), 
        ('scaler', MinMaxScaler()), 
        ('knn', KNeighborsClassifier())
        ])
    params = [{'knn__n_neighbors': range(2,25), 'knn__p': [1, 2]}]

    model = GridSearchCV(pipe, params, cv=cv, n_jobs=-1, scoring=f2scorer, verbose=verbose)
    model.fit(X_train, y_train)
    return model

def logreg_cv(X_train, y_train, cv=5, verbose=False):
    """
    Fits and trains a Logistic Regression model with GridSearchCV
    args:
        X_train (array): Train dataset with Features
        y_train (array): Train Target
        cv (object): Default 5-Fold CrossValidation
        Verbose (bool): True to see verbose GridSearchCV 
    returns:
        model (estimator): best estimator with highest recall
    """
    logreg = LogisticRegression()
    # weights = np.linspace(0.05, 0.95, 10)
    params = [{
        'logreg__penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
        # 'logreg__class_weight': [{0:x, 1:1-x} for x in weights]

        }]
    pipe = imbPipeline(steps=[
        ('sample', SMOTE()), 
        ('logreg', logreg)
        ])
    model = GridSearchCV(pipe, params, cv=cv, n_jobs=-1, scoring=f2scorer, verbose=verbose)
    model.fit(X_train, y_train)
    return model

def rf_cv(X_train, y_train, cv=5, verbose=False):
    """
    Fits and trains a Random Forest model with GridSearchCV
    args:
        X_train (array): Train dataset with Features
        y_train (array): Train Target
        cv (object): Default 5-Fold CrossValidation
        Verbose (bool): True to see verbose GridSearchCV 
    returns:
        model (estimator): best estimator with highest recall
    """
    rf = RandomForestClassifier()
    params = [{
        'rf__n_estimators': range(50, 450, 50),
        'rf__max_depth': [5, 10], 
        'rf__min_samples_split': [2, 10, 20],
        'rf__max_features': ['sqrt', 8, 10],
        'rf__criterion': ['gini'], 
        }]
    pipe = imbPipeline(steps=[('sample', SMOTE()), ('rf', rf)])
    model = GridSearchCV(pipe, params, cv=cv, n_jobs=-1, scoring=f2scorer, verbose=verbose)
    model.fit(X_train, y_train)
    return model

def xgb_cv(X_train, y_train, X_test, y_test, cv=5, verbose=False):
    """
    Fits and trains an XGBoost model with GridsearchCV
        args:
        X_train (array): Train dataset with Features
        y_train (array): Train Target
        cv (object): Default 5-Fold CrossValidation
        Verbose (bool): True to see verbose GridSearchCV 
    returns:
        model (estimator): best estimator with highest recall
    """
    xgb = XGBClassifier()
    params = [{
        'n_estimators': [1000], 
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.25, 0.5, 1], 
        'min_child_weight': [1],
        'colsample_bytree': [.8],
        'scale_pos_weight': [1, 10, 25, 50, 75, 99]
    }]
    fit_params={
        "early_stopping_rounds":30, 
        "eval_metric" : "auc", 
        "eval_set" : [[X_test, y_test]],
        "verbose": verbose
    }
    model = GridSearchCV(
        estimator=xgb, 
        param_grid=params, 
        cv=cv,
        n_jobs=-1,
        scoring='roc_auc')
    model.fit(X_train, y_train, **fit_params)
    return model

def vote(X_train, y_train, estimators, **kwargs):
    """
    Stacks models based on predictions given from past iterations.
    args:
        X_train (array): Train dataset with Features
        y_train (array): Train Target
        cv (object): Default 5-Fold CrossValidation
        Verbose (bool): True to see verbose GridSearchCV 
        estimators (list): List of trained estimators
    returns:
        model (estimator): best estimator with highest recall
    """
    model = VotingClassifier(
        estimators=estimators,  
        n_jobs=-1,
        **kwargs)
    model.fit(X_train, y_train)
    return model
# REDO MODELS FOR CLASS WEIGHT vs SMOTE
# Random Forest: class_weight={0:1, 1:2}
# Doubles the weight of rain train data