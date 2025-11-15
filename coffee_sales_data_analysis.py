#  Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  Load Data
data = pd.read_csv('coffe_sales.csv')

# Initial Inspection
print(data.head())
print(data.info())
print(data.columns)

#  Clean Data
# Drop duplicates if any
data.drop_duplicates(inplace=True)

# Fill missing values for numerical columns (e.g., money)
num_cols = ['money', 'hour_of_day', 'Weekdaysort', 'Monthsort']
for col in num_cols:
    if col in data.columns:
        data[col].fillna(data[col].mean(), inplace=True)

# Fill missing values for categorical columns
cat_cols = ['cash_type', 'coffee_name', 'Time_of_Day', 'Weekday', 'Month_name']
for col in cat_cols:
    if col in data.columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

# Feature Engineering
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month

# Visualization
plt.figure(figsize=(12,6))
sns.boxplot(data=data, x='coffee_name', y='money')
plt.title('Sales distribution by Coffee Type')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(data=data.groupby('Weekday')['money'].sum().reset_index(), x='Weekday', y='money')
plt.title('Total Sales by Day of Week')
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(data=data.groupby('Month_name')['money'].sum().reset_index(), x='Month_name', y='money')
plt.title('Total Sales by Month')
plt.show()

#  Prepare Features and Target
features = ['hour_of_day', 'Weekdaysort', 'Monthsort', 'DayOfWeek', 'Month']
coffee_dummies = pd.get_dummies(data['coffee_name'], prefix='coffee', drop_first=True)
cash_type_dummies = pd.get_dummies(data['cash_type'], prefix='cash', drop_first=True)

X = pd.concat([data[features], coffee_dummies, cash_type_dummies], axis=1)
y = data['money']

# x Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=101)
model.fit(X_train, y_train)

# x Predict & Evaluate
y_pred = model.predict(X_test)
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('R^2 Score:', r2_score(y_test, y_pred))

#  Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print('Feature Importances:')
print(importances)

plt.figure(figsize=(10,6))
importances.plot(kind='bar')
plt.title('Feature Importance')
plt.show()