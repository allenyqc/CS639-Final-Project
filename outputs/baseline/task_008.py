import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate_linear_regression(data, target_column):
    """
    Trains a linear regression model to predict house prices and evaluates its performance.

    Parameters:
        data (pd.DataFrame): The dataset containing features and target.
        target_column (str): The name of the target column (house prices).

    Returns:
        dict: A dictionary containing the model, RMSE, and R^2 score.
    """
    # Split the data into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Return the model and performance metrics
    return {
        "model": model,
        "RMSE": rmse,
        "R2": r2
    }

# Example usage:
# Assuming `df` is a pandas DataFrame with house features and a target column named 'price'
# result = train_and_evaluate_linear_regression(df, 'price')
# print("RMSE:", result["RMSE"])
# print("R^2:", result["R2"])