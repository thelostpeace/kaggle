#!/usr/bin/env python
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

datafile = 'train.csv'

def preprocessing(filename, info=None, mode='train'):
    def embark_map(val):
        if val == 'C':
            return 0
        elif val == 'Q':
            return 1
        elif val == 'S':
            return 2
        return np.nan
    def sex_map(val):
        if val == 'male':
            return 1
        elif val == 'female':
            return 0
    def fare_map(val):
        if val == 0.:
            return np.nan
        return val

    if info == None:
        info = dict()

    data = pd.read_csv(filename)
    data.Embarked = data.Embarked.map(embark_map)
    data.Sex = data.Sex.map(sex_map)
    data.Fare = data.Fare.map(fare_map)
    Y = None
    if mode == 'train':
        Y = data['Survived']
    X = data[['Pclass', 'Sex', 'Fare', 'Embarked']]
    columns = X.columns

    if mode == 'train':
        info['imputer'] = SimpleImputer(strategy='median')
        info['imputer'].fit(X)
    imputer = info['imputer']
    X = imputer.transform(X)
    X = pd.DataFrame(X, columns=columns)

    if mode == 'train':
        info['fare'] = dict()
        info['fare']['mean'] = data.Fare.mean()
        info['fare']['std'] = data.Fare.std()
    fare_mean = info['fare']['mean']
    fare_std = info['fare']['std']
    X.Fare = (data.Fare - fare_mean) / fare_std

    return X, Y, info

train_x, train_y, info = preprocessing(datafile)

params = {
    'max_depth': [2,3,4,5],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.6, 0.8, 1.0],
    'gamma': [0.5, 1, 1.5, 2, 2.5],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic', n_jobs=12)

folds = 3
iteration = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True)
random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=iteration, scoring='roc_auc', n_jobs=12, cv=skf.split(train_x, train_y), verbose=3)

random_search.fit(train_x, train_y)
print(random_search.cv_results_)
print(random_search.best_estimator_)
print(random_search.best_score_)
print(random_search.best_params_)

testfile = 'test.csv'
test_x, _, info = preprocessing(testfile, info, mode='test')
preds = random_search.predict(test_x)
print(preds)

data = pd.read_csv(testfile)
data['Survived'] = preds
data.to_csv('result.csv', columns=['PassengerId', 'Survived'], index=False)
