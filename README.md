Coffee Sales Data Analysis and Prediction
This project analyzes historical coffee sales transaction data from a coffee shop and performs data analysis, feature engineering, and prediction of sales prices using a Random Forest regression model.

Features
Loads and processes 4,000+ coffee sales transactions spanning 9 months (March-December 2024).

Combines temporal, product, and payment data into a comprehensive dataset.

Cleans and preprocesses the dataset (handling missing values, removing duplicates, converting data types).

Engineers temporal features including hour of day, day of week, month, and time period categorization.

Performs exploratory data analysis including:

Sales distribution by coffee type (box plots)

Total sales by day of week (bar charts)

Monthly sales performance tracking

Correlation analysis between features

Feature engineering for price prediction including one-hot encoding of coffee types and payment methods.

Uses Random Forest Regressor (100 estimators) to predict coffee sales prices.

Evaluates model performance using R² Score, MAE, and MSE metrics.

Analyzes feature importance to identify key sales drivers.

Visualizes sales patterns, trends, and model performance.

Installation
Create python virtual environment:
python -m venv venv
.\venv\Scripts\activate

Install the required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn

Usage
Ensure Coffe_sales.csv is in the project directory.

Run coffee_sales_data_analysis.py to perform complete analysis, train the model, evaluate performance, and generate visualizations.

The script will output:

Data cleaning summary

Exploratory visualizations (sales by type, day, month)

Model performance metrics (MAE, MSE, R² Score)

Feature importance rankings

Prediction accuracy plots

Files
Coffe_sales.csv: Raw coffee sales transaction data with 4,000+ records.

coffee_sales_data_anal
