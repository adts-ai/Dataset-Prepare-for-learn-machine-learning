# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from   sklearn.impute import SimpleImputer
from   sklearn.compose import ColumnTransformer
from   sklearn.preprocessing import OneHotEncoder
from   sklearn.preprocessing import LabelEncoder
from   sklearn.model_selection import train_test_split
from   sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler

dt = pd.read_csv('Data.csv')
Information = dt.iloc[:, :-1].values
Target = dt.iloc[:, -1].values

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(Information[:, 1:3])
Information[:, 1:3] = imp.transform(Information[:, 1:3])



ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
Information = np.array(ct.fit_transform(Information))

le = LabelEncoder()
Target = le.fit_transform(Target)



train1, test1, train2, test2 = train_test_split(Information, Target, test_size = 0.4, random_state = 4)

sc = StandardScaler()
train1[:, 3:] = sc.fit_transform(train1)
test1[:, 3:] = sc.transform(test1)
