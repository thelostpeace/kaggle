#!/usr/bin/env python
import pandas as pd
from scipy.stats import chi2
from scipy.stats import chi2_contingency
import numpy as np
import math

train = "data/train.csv"

data = pd.read_csv(train)
significance = 0.05

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

print(len(age_cate))
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

def chi2_analysis(data, target, features):
    feature_selected = []
    for feature in features:
        tb = pd.crosstab(data[target], data[feature])
        print(tb)
        chi, p, dof, expected = chi2_contingency(tb)
        if p < significance:
            print("%s is a good feature" % feature)
            feature_selected.append((feature, p))
        else:
            print("%s is a bad feature" % feature)
    return sorted(feature_selected, key=lambda x : x[1])


features = chi2_analysis(data, "Survived", ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "AgeCate", "CabinCate", "FareCate"])

print("feature\t\tp-value")
for feature in features:
    print("%s\t\t%s" % (feature[0], feature[1]))

