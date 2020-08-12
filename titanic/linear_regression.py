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

labelenc_sex = LabelEncoder()
label_sex = labelenc_sex.fit_transform(data["Sex"])
onehotenc_sex = OneHotEncoder(categories="auto")
onehot_sex = onehotenc_sex.fit_transform(label_sex.reshape(len(label_sex), 1)).toarray()

labelenc_pclass = LabelEncoder()
label_pclass = labelenc_pclass.fit_transform(data["Pclass"])
onehotenc_pclass = OneHotEncoder(categories="auto")
onehot_pclass = onehotenc_pclass.fit_transform(label_pclass.reshape(len(label_pclass), 1)).toarray()

labelenc_cabin = LabelEncoder()
label_cabin = labelenc_cabin.fit_transform(data["CabinCate"])
onehotenc_cabin = OneHotEncoder(categories="auto")
onehot_cabin = onehotenc_cabin.fit_transform(label_cabin.reshape(len(label_cabin), 1)).toarray()

labelenc_fare = LabelEncoder()
fare_classes = ["none", "0-50", "50-100", "100-150", "150-200", "200-250", "250-300", "300+"]
label_fareclass = labelenc_fare.fit_transform(fare_classes)
label_fare = labelenc_fare.transform(data["FareCate"])
onehotenc_fare = OneHotEncoder(categories="auto")
onehotenc_fare.fit(label_fareclass.reshape(len(label_fareclass), 1))
onehot_fare = onehotenc_fare.transform(label_fare.reshape(len(label_fare), 1)).toarray()

labelenc_sibsp = LabelEncoder()
label_sibsp = labelenc_sibsp.fit_transform(data["SibSp"])
onehotenc_sibsp = OneHotEncoder(categories="auto")
onehot_sibsp = onehotenc_sibsp.fit_transform(label_sibsp.reshape(len(label_sibsp), 1)).toarray()

labelenc_embark = LabelEncoder()
label_embark = labelenc_embark.fit_transform(data["EmbarkCate"])
onehotenc_embark = OneHotEncoder(categories="auto")
onehot_embark = onehotenc_embark.fit_transform(label_embark.reshape(len(label_embark), 1)).toarray()

labelenc_parch = LabelEncoder()
parch_classes = [x for x in range(10)]
label_parchclass = labelenc_parch.fit_transform(parch_classes)
label_parch = labelenc_parch.transform(data["Parch"])
onehotenc_parch = OneHotEncoder(categories="auto")
onehotenc_parch.fit(label_parchclass.reshape(len(label_parchclass), 1))
onehot_parch = onehotenc_parch.transform(label_parch.reshape(len(label_parch), 1)).toarray()

labelenc_age = LabelEncoder()
label_age = labelenc_age.fit_transform(data["AgeCate"])
onehotenc_age = OneHotEncoder(categories="auto")
onehot_age = onehotenc_age.fit_transform(label_age.reshape(len(label_age), 1)).toarray()

X = np.concatenate((onehot_sex, onehot_pclass, onehot_cabin, onehot_fare, onehot_sibsp, onehot_embark, onehot_parch, onehot_age), axis=1)

pca = PCA(n_components=2)
#pca = PCA()
X = pca.fit_transform(X)

labelenc_survive = LabelEncoder()
y = labelenc_survive.fit_transform(data["Survived"])
print(y)

plt.scatter(X[:, 0], X[:, 1], alpha=0.2, c=y)
plt.savefig("data.png")

model = LinearRegression().fit(X, y)

acc = accuracy_score(model.predict(X) > 0.5, y)
print("train acc: %s" % acc)

test = "data/test.csv"
test_data = pd.read_csv(test)
add_age_cate(test_data)
add_cabin_cate(test_data)
add_fare_cate(test_data)
add_embark_cate(test_data)
test_label_sex = labelenc_sex.transform(test_data["Sex"])
test_onehot_sex = onehotenc_sex.transform(test_label_sex.reshape(len(test_label_sex), 1)).toarray()
test_label_pclass = labelenc_pclass.transform(test_data["Pclass"])
test_onehot_pclass = onehotenc_pclass.transform(test_label_pclass.reshape(len(test_label_pclass), 1)).toarray()
test_label_cabin = labelenc_cabin.transform(test_data["CabinCate"])
test_onehot_cabin = onehotenc_cabin.transform(test_label_cabin.reshape(len(test_label_cabin), 1)).toarray()
test_label_fare = labelenc_fare.transform(test_data["FareCate"])
test_onehot_fare = onehotenc_fare.transform(test_label_fare.reshape(len(test_label_fare), 1)).toarray()
test_label_sibsp = labelenc_sibsp.transform(test_data["SibSp"])
test_onehot_sibsp = onehotenc_sibsp.transform(test_label_sibsp.reshape(len(test_label_sibsp), 1)).toarray()
test_label_embark = labelenc_embark.transform(test_data["EmbarkCate"])
test_onehot_embark = onehotenc_embark.transform(test_label_embark.reshape(len(test_label_embark), 1)).toarray()
test_label_parch = labelenc_parch.transform(test_data["Parch"])
test_onehot_parch = onehotenc_parch.transform(test_label_parch.reshape(len(test_label_parch), 1)).toarray()
test_label_age = labelenc_age.transform(test_data["AgeCate"])
test_onehot_age = onehotenc_age.transform(test_label_age.reshape(len(test_label_age), 1)).toarray()

test_X = np.concatenate((test_onehot_sex, test_onehot_pclass, test_onehot_cabin, test_onehot_fare, test_onehot_sibsp, test_onehot_embark, test_onehot_parch, test_onehot_age), axis=1)
test_X = pca.transform(test_X)
print(test_X)
pred = model.predict(test_X)
print("preds:")
print(pred)

f = lambda x : 1 if x > 0.5 else 0
test_data["Survived"] = [f(x) for x in pred]
test_data.to_csv("result.csv", columns=["PassengerId", "Survived"], index=False)
