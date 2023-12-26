"""
A DVD rental company needs your help! They want to figure out how many days a customer will rent a DVD for based on some features 
and has approached you for help. They want you to try out some regression models which will help predict the number of days 
a customer will rent a DVD for. The company wants a model which yeilds a MSE of 3 or less on a test set. 
The model you make will help the company become more efficient inventory planning.

The data they provided is in the csv file rental_info.csv. It has the following features:

"rental_date": The date (and time) the customer rents the DVD.
"return_date": The date (and time) the customer returns the DVD.
"amount": The amount paid by the customer for renting the DVD.
"amount_2": The square of "amount".
"rental_rate": The rate at which the DVD is rented for.
"rental_rate_2": The square of "rental_rate".
"release_year": The year the movie being rented was released.
"length": Lenght of the movie being rented, in minuites.
"length_2": The square of "length".
"replacement_cost": The amount it will cost the company to replace the DVD.
"special_features": Any special features, for example trailers/deleted scenes that the DVD also has.
"NC-17", "PG", "PG-13", "R": These columns are dummy variables of the rating of the movie. It takes the value 1 if the move is rated as the column name and 0 otherwise. For your convinience, the reference dummy has already been dropped.

"""
#import
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge

#Prepocessing
df = pd.read_csv('rental_info.csv')
print(df.info())
print(df.describe())
print(df.head())
print(df[['amount','length', 'rental_rate']])

df['return_date'] = pd.to_datetime(df['return_date'])
df['rental_date'] = pd.to_datetime(df['rental_date'])

df['rental_length_days'] = (df['return_date'] - df['rental_date']).dt.days
print(df[['rental_length_days', 'return_date', 'rental_date']])
df = df.drop(['return_date', 'rental_date'], axis=1)

#one_hot encoding
dfDum = pd.get_dummies(df['special_features'])
dfDum['behind_the_scenes'] = 0
dfDum['deleted_scenes'] = 0
for i, row in dfDum.iterrows():
    if row['{"Behind the Scenes"}'] == 1 \
    or row['{"Deleted Scenes","Behind the Scenes"}'] == 1 \
    or row['{Commentaries,"Behind the Scenes"}'] == 1 \
    or row['{Commentaries,"Deleted Scenes","Behind the Scenes"}'] == 1 \
    or row['{Trailers,"Behind the Scenes"}'] == 1 \
    or row['{Trailers,"Deleted Scenes","Behind the Scenes"}'] == 1 \
    or row['{Trailers,Commentaries,"Behind the Scenes"}'] == 1 \
    or row['{Trailers,Commentaries,"Deleted Scenes","Behind the Scenes"}'] == 1 \
    or row['{Trailers,Commentaries,"Behind the Scenes"}'] == 1:
        dfDum.loc[i, ['behind_the_scenes']] = 1
    else: dfDum.loc[i, ['behind_the_scenes']] = 0
    if row['{"Deleted Scenes"}'] == 1 \
    or row['{"Deleted Scenes","Behind the Scenes"}'] == 1 \
    or row['{Commentaries,"Deleted Scenes","Behind the Scenes"}'] == 1 \
    or row['{Commentaries,"Deleted Scenes"}'] == 1 \
    or row['{Trailers,"Deleted Scenes","Behind the Scenes"}'] == 1 \
    or row['{Trailers,"Deleted Scenes"}'] == 1 \
    or row['{Trailers,Commentaries,"Deleted Scenes","Behind the Scenes"}'] == 1 \
    or row['{Trailers,Commentaries,"Deleted Scenes"}'] == 1:
        dfDum.loc[i, ['deleted_scenes']] = 1
    else: dfDum.loc[i, ['deleted_scenes']] = 0
df['deleted_scenes'] = dfDum['deleted_scenes']
df['behind_the_scenes'] = dfDum['behind_the_scenes']
#print(df['behind_the_scenes'].sum())
#print(df['deleted_scenes'].sum())

#Separation data  
X = df.drop(['rental_length_days', 'special_features'], axis=1)
y = df['rental_length_days']
print(X.info())
print(X[['deleted_scenes','behind_the_scenes']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)


#Hyperparams tuning
params = {
    'Ridge': {'alpha' : [0.01, 0.1, 0.5, 1]}
}

best_params = {}
model = Ridge()
grid = GridSearchCV(model, params['Ridge'], cv=3, scoring='r2')
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
best_params['Ridge'] = grid.best_params_
mse_scores = mean_squared_error(y_test, y_pred)
print(best_params)
print(mse_scores)

#predict model
best_model = Ridge(alpha=1)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_train)
mse_scores = mean_squared_error(y_train, y_pred)
print(mse_scores)
y_pred = best_model.predict(X_test)
best_mse = mean_squared_error(y_test, y_pred)
print(best_mse)