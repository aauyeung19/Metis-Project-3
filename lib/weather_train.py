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
from sklearn.preprocessing import OneHotEncoder

wdf = fe.feat_eng_v2()
# set aside 2019 and 2020 data as holdout sets 
wdf.drop(columns=['basin', 'name', 'status', 'precip'], inplace=True)
holdout = wdf[wdf.year>=2018].copy(deep=True)
X_holdout = holdout.drop(columns=['raining'])[:-1]
y_holdout = holdout.raining[1:]

wdf = wdf[wdf.year<2018].copy(deep=True)
X = wdf.drop(columns=['raining', 'year'])[:-1]
y = wdf.raining[1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=None)
X_train.drop(columns=['date'], inplace=True)

# Encode Months
ohe = OneHotEncoder()
ohe.fit(X_train[['month']])
X_train_encoded = pd.DataFrame(
    ohe.transform(X_train[['month']]).toarray(), 
    index=X_train.index,
    columns=ohe.get_feature_names(['month']))
X_test_encoded = pd.DataFrame(
    ohe.transform(X_test[['month']]).toarray(), 
    index=X_test.index,
    columns=ohe.get_feature_names(['month']))
X_holdout_encoded = pd.DataFrame(
    ohe.transform(X_holdout[['month']]).toarray(), 
    index=X_holdout.index,
    columns=ohe.get_feature_names(['month']))
X_train = X_train.join(X_train_encoded)
X_test = X_test.join(X_test_encoded)
X_holdout = X_holdout.join(X_holdout_encoded)

# Store test and holdout in a dictionary to pickle later. 
data = {
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test,
    'X_holdout': X_holdout,
    'y_holdout': y_holdout
}
X_holdout.drop(columns=['year'], inplace=True)
X_test.drop(columns=['date'], inplace=True)
# Train baseline classifiers
# baseline = wcl.baseline_classifiers(X_train, y_train, X_test, y_test)

# Train Models 
cv = StratifiedKFold(n_splits=5, shuffle=True)
knn = wcl.knn_cv(X_train, y_train, cv=cv, verbose=True)
logreg = wcl.logreg_cv(X_train, y_train, cv=cv, verbose=True)
xgb = wcl.xgb_cv(X_train, y_train, X_test, y_test, cv=cv, verbose=True)
rf = wcl.rf_cv(X_train, y_train, cv=cv, verbose=True)

estimators = [
    ('knn', knn.best_estimator_),
    ('logreg', logreg.best_estimator_),
    ('rf', rf.best_estimator_),
    ('xgb', xgb.best_estimator_)
]
weights = [1, 1.4, 1, 0.7]
kwargs = {'voting':'soft', 'weights':weights}
vote = wcl.vote(X_train, y_train, estimators=estimators, verbose=True, **kwargs)


models = ['data', 'knn', 'logreg', 'rf', 'xgb', 'vote']

with open('../models/vote.pickle', 'wb') as pfile:
    pickle.dump(vote, pfile)

if __name__ == '__main__':
    
    print('Are you sure you want to train all models?\n')
    check = input('0 for no, 1 for yes\n')
    if check:
        for model in models:
            curr_model = eval(model)
            with open(f"../models/{model}.pickle", "wb") as pfile:
                pickle.dump(curr_model, pfile)
