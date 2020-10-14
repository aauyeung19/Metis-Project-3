"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

wdf = pd.read_pickle('src/EWRweather_cleaned.pickle')
wdf.set_index('date', inplace=True)
X = wdf.drop(columns=['raining']).shift(1)
y = wdf.raining[1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# KNN BASE LINE
# Log BASE LINE
