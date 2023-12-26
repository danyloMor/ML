import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  f1_score, accuracy_score

telecom_demographics = pd.read_csv('telecom_demographics.csv')
telecom_usage = pd.read_csv('telecom_usage.csv')

print(telecom_demographics.info())
print(telecom_usage.info())
print(telecom_demographics.describe())
print(telecom_usage.describe())
print(telecom_demographics.head())
print(telecom_usage.head())

churn_df = telecom_demographics.merge(telecom_usage, on=['customer_id'], how='inner', suffixes=('_dem', '_usage'))
print(churn_df.info())

#convert categorical features into numeric
churn_df['gender'] = churn_df['gender'].apply(lambda x: 1 if x == 'M' else 0)

churn_df['telecom_partner'] = churn_df['telecom_partner'].replace({'Airtel': 1, 'Reliance Jio': 2, 'Vodafone': 3, 'BSNL':4}) 

#print(churn_df['pincode'].nunique())
churn_df = churn_df.drop(['pincode'], axis=1) #cuz count of unique values near count of all values, it is useless for model

#print(churn_df[['state','city']])
churn_df = churn_df.drop(['state'], axis=1) #cuz one city can't be located in different states

print(churn_df['city'].unique())
churn_df[['Delhi','Hyderabad','Chennai','Bangalore','Kolkata','Mumbai']] = pd.get_dummies(churn_df['city'])
churn_df = churn_df.drop(['city'], axis=1)
#print(churn_df[['Delhi','Hyderabad','Chennai','Bangalore','Kolkata','Mumbai']])

churn_df['registration_event'] = pd.to_datetime(churn_df['registration_event'])
churn_df['registration_year'] = churn_df['registration_event'].dt.year
churn_df['registration_month'] = churn_df['registration_event'].dt.month
churn_df = churn_df.drop(['registration_event'], axis=1)
#print(churn_df['registration_year'])
#print(churn_df['registration_month'])

#due to we have negative values, we can delete them from data or replace them by avg of customers(same city for excample)
churn_df = churn_df[(churn_df['calls_made'] >= 0) & (churn_df['sms_sent'] >= 0) & (churn_df['data_used'] >= 0)]
print(churn_df[['calls_made', 'sms_sent', 'data_used']].describe())

print(churn_df.info())

X = churn_df.drop(['churn', 'customer_id'], axis=1)
y = churn_df['churn'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

#preparing for hyperparam tuning
scaler = StandardScaler() #normalize data due to big variance diff
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    'RidgeClassifier': RidgeClassifier(),
    'RandomForest': RandomForestClassifier()
}
params = {
    'RidgeClassifier': {'alpha': [0.01, 0.1, 0.5, 1, 10], 'random_state': [42]},
    'RandomForest': {'n_estimators': [50, 75, 100], 'max_depth': [2,4,6,8], 'random_state': [42]},
}
best_hyperparameters = {}
accuracy = {}
f1 = {}
for model_name, model in models.items():
    clf = GridSearchCV(model, params[model_name], cv=3, scoring='accuracy')
    clf.fit(X_train, y_train)
    best_hyperparameters[model_name] = clf.best_params_
    y_pred = clf.predict(X_test)
    accuracy[model_name] = accuracy_score(y_test, y_pred)
    f1[model_name] = f1_score(y_test, y_pred, average='weighted')
print(best_hyperparameters)
print(accuracy)
print(f1)

#final model
model = RandomForestClassifier(max_depth = 8, n_estimators = 100, random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print(accuracy)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
importance = pd.Series(model.feature_importances_, index=X_train.columns)
print(importance)