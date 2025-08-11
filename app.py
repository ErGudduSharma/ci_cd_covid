import numpy as np
import pandas as pd

df = pd.read_csv('insurance.csv')
print(df.head())

print(df.info())
print(" ")

print(df.describe())
print(" ")
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype =='object':
        df[col] = le.fit_transform(df[col])

print(df.head())

from sklearn.model_selection import train_test_split
x = df.drop(columns='charges', axis=1)
y = df['charges']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(np.round(x_train.describe(), 2))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train_sc = sc.fit_transform(x_train)
x_train_new = pd.DataFrame(x_train_sc, columns = x_train.columns)

print(np.round(x_train_new.describe(), 2))


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

from sklearn.metrics import r2_score
print('pridicted accuracy:', r2_score(y_test, y_pred))

from sklearn.tree import DecisionTreeRegressor
dc = DecisionTreeRegressor()
dc.fit(x_train, y_train)
y_pred = dc.predict(x_test)

from sklearn.metrics import r2_score
print('pridicted accuracy:', r2_score(y_test, y_pred))


from sklearn.ensemble import RandomForestRegressor 
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

from sklearn.metrics import r2_score
print('pridicted accuracy:', r2_score(y_test, y_pred))


