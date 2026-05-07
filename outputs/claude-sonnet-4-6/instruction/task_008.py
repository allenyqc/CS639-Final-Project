```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


def train_and_evaluate_house_price_model(
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> dict:
    """
    Train a Linear Regression model to predict house prices and report performance.

    Best practices followed:
    - Train/test split happens BEFORE any preprocessing.
    - StandardScaler is fit ONLY on training data (via Pipeline).
    - Evaluation uses multiple regression metrics (MAE, RMSE, R², MAPE).
    - Cross-validation is performed on training data only.

    Parameters
    ----------
    test_size    : Fraction of data reserved for the test set.
    random_state : Seed for reproducibility.
    cv_folds     : Number of folds for cross-validation.

    Returns
    -------
    dict with trained pipeline, metrics, and feature importances.
    """

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("  House Price Prediction — Linear Regression")
    print("=" * 60)

    housing = fetch_california_housing(as_frame=True)
    X: pd.DataFrame = housing.frame.drop(columns=["MedHouseVal"])
    y: pd.Series = housing.frame["MedHouseVal"]  # target in $100k units

    print(f"\nDataset shape : {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"Target range  : [{y.min():.2f}, {y.max():.2f}] ($100k)")

    # ------------------------------------------------------------------ #
    # 2. Train / test split — BEFORE any feature engineering
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\nTrain samples : {len(X_train):,}")
    print(f"Test  samples : {len(X_test):,}")

    # ------------------------------------------------------------------ #
    # 3. Build pipeline (scaler fit ONLY on training data)
    # ------------------------------------------------------------------ #
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),   # fit on X_train only
            ("model", LinearRegression()),
        ]
    )

    # ------------------------------------------------------------------ #
    # 4. Cross-validation on training data (no test data leakage)
    # ------------------------------------------------------------------ #
    print(f"\nRunning {cv_folds}-fold cross-validation on training data …")
    cv_r2_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv_folds, scoring="r2", n_jobs=-1
    )
    cv_neg_rmse = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv_folds, scoring="neg_root_mean_squared_error", n_jobs=-1
    )

    print(f"  CV R²   : {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
    print(f"  CV RMSE : {(-cv_neg_rmse).mean():.4f} ± {(-cv_neg_rmse).std():.4f}")

    # ------------------------------------------------------------------ #
    # 5. Final training on full training set
    # ------------------------------------------------------------------ #
    pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # 6. Evaluate on held-out test set
    # ------------------------------------------------------------------ #
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # as %

    # Residual analysis
    residuals = y_test.values - y_pred
    residual_std = residuals.std()

    print("\n" + "=" * 60)
    print("  Test-Set Performance Metrics")
    print("=" * 60)
    print(f"  MAE   (Mean Absolute Error)        : {mae:.4f}  ($100k)")
    print(f"  RMSE  (Root Mean Squared Error)    : {rmse:.4f}  ($100k)")
    print(f"  R²    (Coefficient of Determination): {r2:.4f}")
    print(f"  MAPE  (Mean Abs. Percentage Error) : {mape:.2f} %")
    print(f"  Residual Std Dev                   : {residual_std:.4f}")

    # ------------------------------------------------------------------ #
    # 7. Feature importances (coefficients after scaling)
    # ------------------------------------------------------------------ #
    coef = pipeline.named_steps["model"].coef_
    feature_importance = pd.Series(coef, index=X.columns).sort_values(
        key=abs, ascending=False
    )

    print("\n" + "=" * 60)
    print("  Feature Coefficients (scaled units)")
    print("=" * 60)
    for feat, val in feature_importance.items():
        bar = "█" * int(abs(val) / feature_importance.abs().max() * 20)
        direction = "+" if val >= 0 else "-"
        print(f"  {feat:<15} {direction}{abs(val):6.4f}  {bar}")

    # ------------------------------------------------------------------ #
    # 8. Interpretation summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  Interpretation")
    print("=" * 60)
    if r2 >= 0.7:
        quality = "good"
    elif r2 >= 0.5:
        quality = "moderate"
    else:
        quality = "poor"

    print(
        f"  The model explains {r2 * 100:.1f}% of variance in house prices "
        f"({quality} fit)."
    )
    print(
        f"  On average, predictions are off by ~${mae * 100_000:,.0f} "
        f"(MAE) or ~{mape:.1f}% of the actual price."
    )

    # ------------------------------------------------------------------ #
    # 9. Return artefacts
    # ------------------------------------------------------------------ #
    results = {
        "pipeline": pipeline,
        "metrics": {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mape_pct": mape,
            "residual_std": residual_std,
            "cv_r2_mean": cv_r2_scores.mean(),
            "cv_r2_std": cv_r2_scores.std(),
            "cv_rmse_mean": (-cv_neg_rmse).mean(),
            "cv_rmse_std": (-cv_neg_rmse).std(),
        },
        "feature_coefficients": feature_importance.to_dict(),
        "splits": {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
        "predictions": y_pred,
    }
    return results


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":