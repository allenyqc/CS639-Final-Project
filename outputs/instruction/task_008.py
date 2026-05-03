import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

def train_and_evaluate_linear_regression(data, target_column):
    """
    Trains a linear regression model to predict house prices and evaluates its performance.

    Parameters:
    - data (pd.DataFrame): The dataset containing features and target.
    - target_column (str): The name of the target column in the dataset.

    Returns:
    - dict: A dictionary containing model performance metrics.
    """
    # Step 1: Split the data into train and test sets
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('model', LinearRegression())  # Linear regression model
    ])

    # Step 3: Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Step 4: Make predictions on the test data
    y_pred = pipeline.predict(X_test)

    # Step 5: Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Return performance metrics
    return {
        'Mean Squared Error': mse,
        'R^2 Score': r2
    }

# Example usage:
# data = pd.read_csv('house_prices.csv')  # Load your dataset
# metrics = train_and_evaluate_linear_regression(data, target_column='price')
# print(metrics)