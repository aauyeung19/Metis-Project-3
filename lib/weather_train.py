"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

Author: @Andrew Auyeung
Location: Metis-Project-3/lib/
Dependencies: cleaning.py, feature_eng.py, classification.py
Use this file to Train Models and pickle them into "/models"
"""
# import my libraries
import cleaning as cl
import feature_eng as fe
import classification as wcl

# import standard libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import RandomOverSampler

wdf = fe.feat_eng_v2()
# set aside 2019 and 2020 data as holdout sets 
wdf.drop(columns=['basin', 'name', 'status'], inplace=True)
holdout = wdf[wdf.year>=2018].copy(deep=True)
X_holdout = holdout.drop(columns=['raining'])[:-1]
y_holdout = holdout.raining[1:]

wdf = wdf[wdf.year<2018].copy(deep=True)
X = wdf.drop(columns=['raining'])[:-1]
y = wdf.raining[1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=None)
# Store test and holdout in a dictionary to pickle later. 
test_holdout = {
    'X_test': X_test,
    'y_test': y_test,
    'X_holdout': X_holdout,
    'y_holdout': y_holdout
}

# Train baseline classifiers
baseline = wcl.baseline_classifiers(X_train, y_train, X_test, y_test)

# Train Models 
cv = StratifiedKFold(n_splits=5, shuffle=True)
knn = wcl.knn_cv(X_train, y_train, cv=cv, verbose=True)
logreg = wcl.logreg_cv(X_train, y_train, cv=cv, verbose=True)
xgb = wcl.xgb_cv(X_train, y_train, X_test, y_test, cv=cv, verbose=True)
rf = wcl.rf_cv(X_train, y_train, cv=cv, verbose=True)

models = ['baseline', 'test_holdout', 'knn', 'logreg', 'rf', 'xgb']

if __name__ == '__main__':
    
    print('Are you sure you want to train all models?\n')
    check = input('0 for no, 1 for yes\n')
    if check:
        for model in models:
            curr_model = eval(model)
            with open(f"../models/{model}.pickle", "wb") as pfile:
                pickle.dump(curr_model, pfile)
