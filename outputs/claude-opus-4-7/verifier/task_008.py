import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_house_price_model(X=None, y=None, test_size=0.2, random_state=42, cv_folds=5):
    """
    Train a linear regression model to predict house prices and report performance.

    Parameters
    ----------
    X : array-like or pandas.DataFrame, optional
        Feature matrix. If None, the California housing dataset is used.
    y : array-like, optional
        Target vector (house prices). If None, the California housing target is used.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.
    cv_folds : int, default=5
        Number of folds for cross-validation.

    Returns
    -------
    dict
        Dictionary containing the trained model, evaluation metrics,
        feature coefficients, and the train/test splits.
    """
    # Load default dataset if no data is provided
    if X is None or y is None:
        data = fetch_california_housing(as_frame=True)
        X = data.data
        y = data.target
        feature_names = list(X.columns)
    else:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            X = np.asarray(X)
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        y = np.asarray(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Pipeline: scale features then fit linear regression
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
    }

    # Cross-validation R^2 on training data
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="r2")
    metrics["cv_r2_mean"] = cv_scores.mean()
    metrics["cv_r2_std"] = cv_scores.std()

    # Feature coefficients (in scaled space)
    coefs = model.named_steps["regressor"].coef_
    intercept = model.named_steps["regressor"].intercept_
    coef_table = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
    }).sort_values("coefficient", key=abs, ascending=False)

    # Report
    print("=" * 60)
    print("Linear Regression — House Price Prediction")
    print("=" * 60)
    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Features         : {len(feature_names)}")
    print("-" * 60)
    print("Performance Metrics")
    print("-" * 60)
    print(f"Train R²   : {metrics['train_r2']:.4f}")
    print(f"Test  R²   : {metrics['test_r2']:.4f}")
    print(f"Train RMSE : {metrics['train_rmse']:.4f}")
    print(f"Test  RMSE : {metrics['test_rmse']:.4f}")
    print(f"Train MAE  : {metrics['train_mae']:.4f}")
    print(f"Test  MAE  : {metrics['test_mae']:.4f}")
    print(f"CV R² ({cv_folds}-fold): {metrics['cv_r2_mean']:.4f} "
          f"(± {metrics['cv_r2_std']:.4f})")
    print("-" * 60)
    print(f"Intercept (scaled): {intercept:.4f}")
    print("Feature coefficients (scaled, by |importance|):")
    print(coef_table.to_string(index=False))
    print("=" * 60)

    return {
        "model": model,
        "metrics": metrics,
        "coefficients": coef_table,
        "intercept": intercept,
        "splits": (X_train, X_test, y_train, y_test),
    }


if __name__ == "__main__":
    results = train_house_price_model()