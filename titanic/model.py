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
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data/train.csv')

label_sex = LabelEncoder()
data.Sex = label_sex.fit_transform(data.Sex)
label_embarked = LabelEncoder()
embarked = data[data.Embarked.isnull() == False]
data.loc[embarked.index, 'Embarked'] = label_embarked.fit_transform(embarked.Embarked)

def map_cabin(val):
    def remove_char(s, d):
        if isinstance(val, str):
            for c in d:
                s = s.replace(c, '')
            return ''.join(set(s))
        else:
            return val

    return remove_char(val, " 0123456789")

data.Cabin = data.Cabin.map(map_cabin)
label_cabin = LabelEncoder()
cabin = data[data.Cabin.isnull() == False]
data.loc[cabin.index, 'Cabin'] = label_cabin.fit_transform(cabin.Cabin)

#fill missing embarked
embarked_train = data[data.Embarked.isnull() == False][['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
embarked_x = embarked_train.drop(columns=['Embarked'])
embarked_y = embarked_train['Embarked'].astype('int64')
params = {
    'n_estimators': range(100, 1000, 100),
    'max_depth': [2,3,4,5,6],
    'gamma': np.arange(0, 5, 0.5),
    'min_child_weight': range(1, 6, 1),
    'subsample': np.arange(0.6, 1, 0.1),
    'colsample_bytree': np.arange(0.1, 1, 0.1)
}

xgb = XGBClassifier(learning_rate=0.02, objective='multi:softmax', n_jobs=6, num_class=3)
model_embarked = RandomizedSearchCV(xgb, param_distributions=params, n_iter=10, n_jobs=6, cv=StratifiedKFold(shuffle=True), verbose=3, random_state=1992)
model_embarked.fit(embarked_x, embarked_y)

embarked_fill = data[data.Embarked.isnull() == True][['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']]
data.loc[embarked_fill.index, 'Embarked'] = model_embarked.predict(embarked_fill)

# fill missing Cabin
cabin_train = data[data.Cabin.isnull() == False][['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin']]
cabin_x = cabin_train.drop(columns=['Cabin'])
cabin_y = cabin_train['Cabin'].astype('int64')

xgb = XGBClassifier(learning_rate=0.02, objective='multi:softmax', n_jobs=6, num_class=9)
model_cabin = RandomizedSearchCV(xgb, param_distributions=params, n_iter=10, n_jobs=6, cv=StratifiedKFold(shuffle=True), verbose=3, random_state=1992)
model_cabin.fit(cabin_x, cabin_y)

cabin_fill = data[data.Cabin.isnull() == True][['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
data.loc[cabin_fill.index, 'Cabin'] = model_cabin.predict(cabin_fill)

# fill missing fare
fare_train = data[data.Fare != 0.][['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
fare_x = fare_train.drop(columns=['Fare'])
fare_y = fare_train['Fare']

xgb = XGBRegressor(learning_rate=0.02, objective='reg:squarederror', n_jobs=6)
model_fare = RandomizedSearchCV(xgb, param_distributions=params, n_iter=10, n_jobs=6, cv=KFold(shuffle=True), verbose=3, random_state=1992)
model_fare.fit(fare_x, fare_y)

fare_fill = data[data.Fare == 0.][['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']]
data.loc[fare_fill.index, 'Fare'] = model_fare.predict(fare_fill)

# fill missing age
age_train = data[data.Age.isnull() == False][['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Age']]
age_x = age_train.drop(columns=['Age'])
age_y = age_train['Age']

xgb = XGBRegressor(learning_rate=0.02, objective='reg:squarederror', n_jobs=6)
model_age = RandomizedSearchCV(xgb, param_distributions=params, n_iter=10, n_jobs=6, cv=KFold(shuffle=True), verbose=3, random_state=1992)
model_age.fit(age_x, age_y)

age_fill = data[data.Age.isnull() == True][['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
data.loc[age_fill.index, 'Age'] = model_age.predict(age_fill)

# now we predict Survived
train_x = data.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket'])
train_y = data['Survived']

xgb = XGBClassifier(learning_rate=0.02, objective='binary:logistic', n_jobs=6)
model = RandomizedSearchCV(xgb, param_distributions=params, n_iter=10, n_jobs=6, cv=StratifiedKFold(shuffle=True), verbose=3, random_state=1992)
model.fit(train_x, train_y)

data = pd.read_csv('data/test.csv')
test_x = data.drop(columns=['PassengerId', 'Name', 'Ticket'])
test_x.Sex = label_sex.transform(test_x.Sex)
test_x.Embarked = label_embarked.transform(test_x.Embarked)

fare_fill = test_x[test_x.Fare == 0.][['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']]
test_x.loc[fare_fill.index, 'Fare'] = model_fare.predict(fare_fill)

test_x.Cabin = test_x.Cabin.map(map_cabin)
cabin = test_x[test_x.Cabin.isnull() == False]
test_x.loc[cabin.index, 'Cabin'] = label_cabin.transform(cabin.Cabin)
cabin_fill = test_x[test_x.Cabin.isnull() == True][['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
test_x.loc[cabin_fill.index, 'Cabin'] = model_cabin.predict(cabin_fill)

age_fill = test_x[test_x.Age.isnull() == True][['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
test_x.loc[age_fill.index, 'Age'] = model_age.predict(age_fill)
data['Survived'] = model.predict(test_x)
data.to_csv('result.csv', columns=['PassengerId', 'Survived'], index=False)
