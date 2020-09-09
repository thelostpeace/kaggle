#!/usr/bin/env python
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

datafile = 'train.csv'
data = pd.read_csv(datafile)

