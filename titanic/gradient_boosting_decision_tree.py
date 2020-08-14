#!/usr/bin/env python
from scipy.stats import chi2, chi2_contingency
import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt

train = "data/train.csv"

data = pd.read_csv(train)

def add_age_cate(data):
    # age category
    age_cate = []
    for age in data["Age"]:
        if math.isnan(age):
            age_cate.append("none")
        elif age >= 0. and age < 10.:
            age_cate.append("0-10")
        elif age >= 10. and age < 20.:
            age_cate.append("10-20")
        elif age >= 20. and age < 30.:
            age_cate.append("20-30")
        elif age >= 30. and age < 40.:
            age_cate.append("30-40")
        elif age >= 40. and age < 50.:
            age_cate.append("40-50")
        elif age >= 50. and age < 60.:
            age_cate.append("50-60")
        elif age >= 60. and age < 70.:
            age_cate.append("60-70")
        elif age >= 70. and age < 80.:
            age_cate.append("70-80")
        elif age >= 80.:
            age_cate.append("80+")
        else:
            print("skip age: %s" % age)
    data["AgeCate"] = age_cate
add_age_cate(data)

def add_cabin_cate(data):
    # cabin category
    cabin_cate = []
    for cabin in data["Cabin"]:
        if isinstance(cabin, float) and math.isnan(cabin):
            cabin_cate.append("none")
        else:
            cabin_cate.append(cabin[0])
    data["CabinCate"] = cabin_cate
add_cabin_cate(data)

def add_fare_cate(data):
    # fare category
    fare_cate = []
    for fare in data["Fare"]:
        if math.isnan(fare):
            fare_cate.append("none")
        elif fare >= 0. and fare < 50.:
            fare_cate.append("0-50")
        elif fare >= 50. and fare < 100.:
            fare_cate.append("50-100")
        elif fare >= 100. and fare < 150.:
            fare_cate.append("100-150")
        elif fare >= 150. and fare < 200.:
            fare_cate.append("150-200")
        elif fare >= 200. and fare < 250.:
            fare_cate.append("200-250")
        elif fare >= 250. and fare < 300.:
            fare_cate.append("250-300")
        elif fare >= 300.:
            fare_cate.append("300+")
        else:
            print("skip fare: %s" % fare)
    data["FareCate"] = fare_cate
add_fare_cate(data)

def add_embark_cate(data):
    # cabin category
    embark_cate = []
    for embark in data["Embarked"]:
        if isinstance(embark, float) and math.isnan(embark):
            embark_cate.append("none")
        else:
            embark_cate.append(embark)
    data["EmbarkCate"] = embark_cate
add_embark_cate(data)

train_X = pd.DataFrame()

labelenc_sex = LabelEncoder()
label_sex = labelenc_sex.fit_transform(data["Sex"])
train_X["sex"] = label_sex

labelenc_pclass = LabelEncoder()
label_pclass = labelenc_pclass.fit_transform(data["Pclass"])
train_X["pclass"] = label_pclass

labelenc_cabin = LabelEncoder()
label_cabin = labelenc_cabin.fit_transform(data["CabinCate"])
train_X["cabin"] = label_cabin
train_X["fare"] = data["Fare"]
train_X["sibsp"] = data["SibSp"]

labelenc_embark = LabelEncoder()
label_embark = labelenc_embark.fit_transform(data["EmbarkCate"])
train_X["embark"] = label_embark

labelenc_parch = LabelEncoder()
parch_classes = [x for x in range(10)]
label_parchclass = labelenc_parch.fit_transform(parch_classes)
label_parch = labelenc_parch.transform(data["Parch"])
train_X["parch"] = label_parch

train_X["age"] = data["Age"]

train_mean = train_X.mean()
train_X = train_X.fillna(train_mean)

labelenc_survive = LabelEncoder()
y = labelenc_survive.fit_transform(data["Survived"])

test = "data/test.csv"
test_data = pd.read_csv(test)
add_age_cate(test_data)
add_cabin_cate(test_data)
add_fare_cate(test_data)
add_embark_cate(test_data)
test_X = pd.DataFrame()
test_label_sex = labelenc_sex.transform(test_data["Sex"])
test_X["sex"] = test_label_sex
test_label_pclass = labelenc_pclass.transform(test_data["Pclass"])
test_X["pclass"] = test_label_pclass
test_label_cabin = labelenc_cabin.transform(test_data["CabinCate"])
test_X["cabin"] = test_label_cabin
test_X["fare"] = test_data["Fare"]
test_X["sibsp"] = test_data["SibSp"]
test_label_embark = labelenc_embark.transform(test_data["EmbarkCate"])
test_X["embark"] = test_label_embark
test_label_parch = labelenc_parch.transform(test_data["Parch"])
test_X["parch"] = test_label_parch
test_X["age"] = test_data["Age"]

test_X = test_X.fillna(train_mean)
print(test_X)

params = {
    "loss": ["deviance", "exponential"],
    "learning_rate": [0.1],
    "n_estimators": range(20, 100),
    "criterion": ["friedman_mse", "mse", "mae"],
    "max_depth": range(2, 12)
}
model = GridSearchCV(GradientBoostingClassifier(), param_grid=params, verbose=16, n_jobs=24)
model.fit(train_X, y)
print(model.best_params_)

acc = accuracy_score(model.predict(train_X), y)
print("train acc: %s" % acc)

pred = model.predict(test_X)
print("preds:")
print(pred)

test_data["Survived"] = pred
test_data.to_csv("result.csv", columns=["PassengerId", "Survived"], index=False)
