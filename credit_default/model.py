import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin
import glob
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import type_of_target
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

data = pd.read_excel('data.xls', skiprows=0, header=1)
data = data.drop(columns=['ID'])

# use pandas sample rather than sklearn train_test_split
# so I can keep dataframe and do preprocessing later
train = data.sample(frac=0.8, random_state=1992)
test = data.drop(train.index)

train_x = train.drop(columns=['default payment next month'])
train_y = train['default payment next month'].astype('int64')
#print(train_y)

test_x = test.drop(columns=['default payment next month'])
test_y = test['default payment next month'].astype('int64')

params = {
    'feature__threshold': [0.01, 0.05, 0.1, 0.5, 1, 'mean', 'median', '1.5*mean', '1.5*median'],
    'feature__estimator__penalty': ['l1', 'l2'],
    'feature__estimator__loss': ['hinge', 'squared_hinge'],
    'feature__estimator__dual': [True, False],
    'xgboost__n_estimators': range(100, 1000, 100),
    'xgboost__max_depth': range(2, 10, 1),
    'xgboost__gamma': np.arange(0, 5, 0.5),
    'xgboost__min_child_weight': range(1, 6, 1),
    'xgboost__subsample': np.arange(0.6, 1, 0.1),
    'xgboost__colsample_bytree': np.arange(0.1, 1, 0.1)
}

pipeline = Pipeline([
    ('feature', SelectFromModel(LinearSVC())),
    ('xgboost', XGBClassifier(learning_rate=0.02, objective='binary:logistic', n_jobs=8))
])

model = RandomizedSearchCV(pipeline, param_distributions=params, n_iter=20, n_jobs=8, cv=StratifiedKFold(shuffle=True), verbose=3, random_state=1992, scoring='accuracy')
model.fit(train_x, train_y)

print(model.best_estimator_)
print(model.best_score_)
print(model.best_params_)
print("accuracy: ", accuracy_score(test_y, model.predict(test_x)))

print("selected features: ", ' | '.join(train.columns[:-1][model.best_estimator_.named_steps['feature'].get_support()].tolist()))
