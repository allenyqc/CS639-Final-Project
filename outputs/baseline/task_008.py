from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

def train_and_evaluate_linear_regression(data, target_column):
    # Split the data into features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the performance metrics
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return model

# Example usage:
# Assuming 'data' is a pandas DataFrame with house features and 'price' as the target column
# data = pd.read_csv('house_prices.csv')
# model = train_and_evaluate_linear_regression(data, 'price')