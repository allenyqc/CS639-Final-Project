import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_house_price_model(
    data: pd.DataFrame,
    target_col: str = "price",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train a linear regression model to predict house prices.

    Best practices followed:
    - Train/test split is performed BEFORE any feature engineering.
    - Preprocessors (imputers, scalers, encoders) are fit ONLY on training data
      via a Pipeline; the same fitted transformations are applied to test data.
    - Hyperparameter / model selection is done via cross-validation on the
      training set only — the test set is held out and used solely for the
      final unbiased evaluation.
    - For regression we report MAE, RMSE, and R^2 (accuracy is not meaningful).
    """

    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    # 1. Separate features/target and split BEFORE any feature engineering.
    X = data.drop(columns=[target_col])
    y = data[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. Identify column types from TRAINING data only.
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # 3. Build preprocessing + model pipeline (fit only on training data).
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    # 4. Cross-validate on TRAINING data only (no peeking at the test set).
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    cv_neg_rmse = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
    )

    # 5. Fit on full training set, evaluate ONCE on held-out test set.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "cv_rmse_mean": float(-cv_neg_rmse.mean()),
        "cv_rmse_std": float(cv_neg_rmse.std()),
        "test_mae": float(mae),
        "test_rmse": rmse,
        "test_r2": float(r2),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    print("=== House Price Linear Regression ===")
    print(f"Training samples: {metrics['n_train']}, Test samples: {metrics['n_test']}")
    print(
        f"CV R^2 (train):  {metrics['cv_r2_mean']:.4f} "
        f"(+/- {metrics['cv_r2_std']:.4f})"
    )
    print(
        f"CV RMSE (train): {metrics['cv_rmse_mean']:.4f} "
        f"(+/- {metrics['cv_rmse_std']:.4f})"
    )
    print(f"Test MAE:        {metrics['test_mae']:.4f}")
    print(f"Test RMSE:       {metrics['test_rmse']:.4f}")
    print(f"Test R^2:        {metrics['test_r2']:.4f}")

    return model, metrics


if __name__ == "__main__":
    # Demo with the California housing dataset.
    from sklearn.datasets import fetch_california_housing

    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame.rename(columns={"MedHouseVal": "price"})
    fitted_model, results = train_house_price_model(df, target_col="price")