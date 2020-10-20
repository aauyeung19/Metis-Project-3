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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import RandomOverSampler

wdf = fe.feat_eng_v2()
# set aside 2019 and 2020 data as holdout sets 
holdout = wdf[wdf.year>=2019].copy(deep=True)
X_holdout = holdout.drop(columns=['raining'])[:-1]
y_holdout = holdout.raining[1:]

wdf = wdf[wdf.year<2019].copy(deep=True)
X = wdf.drop(columns=['raining'])[:-1]
y = wdf.raining[1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=None)

ros = RandomOverSampler(random_state=None)
X_train, y_train = ros.fit_sample(X_train,y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True)
est = rf_cv(X_train, y_train, cv)

print(confusion_matrix(y_test, (est.predict_proba(X_test)[:,1]>0.5).astype(int)))
print(classification_report(y_test, (est.predict_proba(X_test)[:,1]>0.5).astype(int)))

if __name__ == '__main__':

    # Run simple Linear Regression on each ma result. 
    # Use Linear Model to predict true temp.
    # Merge Linear Model Predictions back to date df. 
    
