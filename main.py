import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(X[:, 1:3])
X[:, 1:3] = impute.transform(X[:, 1:3])

# encoding the independent variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# encoding the dependent variable
le = LabelEncoder()
y = le.fit_transform(y)

# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# feature scaling
# standardisation = (x-mean(x))/standard deviation(x)=[-3:3]
# normalisation = (x-min(x))/(max(x)-min(x))=[0:1]
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
