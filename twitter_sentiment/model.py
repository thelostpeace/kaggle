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

data = pd.read_csv('data/data.tsv', sep='\t')

# use pandas sample rather than sklearn train_test_split
# so I can keep dataframe and do preprocessing later
train = data.sample(frac=0.8, random_state=1992)
test = data.drop(train.index)

train_x = train['SentimentText']
train_y = train['Sentiment'].astype('int64')
#print(train_y)

test_x = test['SentimentText']
test_y = test['Sentiment'].astype('int64')

class Preprocessing(TransformerMixin):
    def __init__(self):
        self.transformer = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,2), norm='l2')

    def transform(self, data):
        return self.transformer.transform(data)

    def fit(self, data):
        self.transformer.fit(data)
        return self

preprocess = Preprocessing()
train_x = preprocess.fit_transform(train_x)
print("train shape: ", train_x.shape)
test_x = preprocess.transform(test_x)

params = {
    'n_estimators': range(100, 1000, 100),
    'max_depth': range(2, 10, 1),
    'gamma': np.arange(0, 5, 0.5),
    'min_child_weight': range(1, 6, 1),
    'subsample': np.arange(0.6, 1, 0.1),
    'colsample_bytree': np.arange(0.1, 1, 0.1)
}

xgb = XGBClassifier(learning_rate=0.02, objective='binary:logistic', n_jobs=8)
model = RandomizedSearchCV(xgb, param_distributions=params, n_iter=10, n_jobs=8, cv=StratifiedKFold(shuffle=True), verbose=3, random_state=1992, scoring='accuracy')
model.fit(train_x, train_y)

print(model.best_estimator_)
print(model.best_score_)
print(model.best_params_)
print("accuracy: ", accuracy_score(test_y, model.predict(test_x)))
