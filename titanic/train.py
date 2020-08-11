#!/usr/bin/env python
from scipy.stats import chi2, chi2_contingency
import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

train = "data/train.csv"

data = pd.read_csv(train)

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

# cabin category
cabin_cate = []
for cabin in data["Cabin"]:
    if isinstance(cabin, float) and math.isnan(cabin):
        cabin_cate.append("none")
    else:
        cabin_cate.append(cabin[0])
data["CabinCate"] = cabin_cate

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

labelenc_sex = LabelEncoder()
label_sex = labelenc_sex.fit_transform(data["Sex"])
onehotenc_sex = OneHotEncoder(categories="auto")
onehot_sex = onehotenc_sex.fit_transform(label_sex.reshape(len(label_sex), 1)).toarray()

X = onehot_sex

pca = PCA(n_components=2)
X = pca.fit_transform(X)

labelenc_survive = LabelEncoder()
y = labelenc_survive.fit_transform(data["Survived"])

model = LinearRegression().fit(X, y)

acc = accuracy_score(model.predict(X) > 0.5, y)
print("train acc: %s" % acc)

test = "data/test.csv"
test_data = pd.read_csv(test)
test_label_sex = labelenc_sex.transform(test_data["Sex"])
test_onehot_sex = onehotenc_sex.transform(test_label_sex.reshape(len(test_label_sex), 1)).toarray()

test_X = test_onehot_sex
test_X = pca.transform(test_X)
pred = model.predict(test_X)
print("preds:")
print(pred)

f = lambda x : 1 if x > 0.5 else 0
test_data["Survived"] = [f(x) for x in pred]
test_data.to_csv("result.csv", columns=["PassengerId", "Survived"], index=False)
