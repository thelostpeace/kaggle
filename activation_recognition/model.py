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

files = glob.glob('data/activity/*.csv')

columns = ['index', 'x', 'y', 'z', 'label']
data = pd.DataFrame(columns=columns)
for fname in files:
    data = data.append(pd.read_csv(fname, names=columns), sort=False)

data.drop(labels=['index'], axis=1, inplace=True)

# use pandas sample rather than sklearn train_test_split
# so I can keep dataframe and do preprocessing later
train = data.sample(frac=0.8, random_state=1992)
test = data.drop(train.index)

train_x = train.drop(labels=['label'], axis=1)
train_y = train['label'].astype('int64')

test_x = test.drop(labels=['label'], axis=1)
test_y = test['label'].astype('int64')

class Preprocessing(TransformerMixin):
    def __init__(self):
        self.normalizer = StandardScaler()

    def transform(self, data):
        x = data.copy()
        return self.normalizer.transform(data)

    def fit(self, data):
        self.normalizer.fit(data)
        return self

preprocess = Preprocessing()
train_x = preprocess.fit_transform(train_x)
test_x = preprocess.transform(test_x)

params = {
    'n_estimators': range(100, 1000, 100),
    'max_depth': [2,3,4,5,6],
    'gamma': np.arange(0, 5, 0.5),
    'min_child_weight': range(1, 6, 1),
    'subsample': np.arange(0.6, 1, 0.1),
    'colsample_bytree': np.arange(0.1, 1, 0.1)
}

xgb = XGBClassifier(learning_rate=0.02, objective='multi:softmax', n_jobs=6, num_class=8)
model = RandomizedSearchCV(xgb, param_distributions=params, n_iter=10, n_jobs=6, cv=StratifiedKFold(shuffle=True), verbose=3, random_state=1992)
model.fit(train_x, train_y)

print(model.best_estimator_)
print(model.best_score_)
print(model.best_params_)
print("accuracy: ", accuracy_score(test_y, model.predict(test_x)))
