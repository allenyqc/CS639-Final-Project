```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def train_house_price_model(data=None, target_column="price", test_size=0.2, random_state=42):
    """
    Train a linear regression model to predict house prices and report performance.

    Parameters:
        data (pd.DataFrame, optional): DataFrame with features and target column.
                                       If None, synthetic data is generated.
        target_column (str): Name of the target column in the DataFrame.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Dictionary containing the trained model, scaler, metrics, and feature importance.
    """

    # ── 1. Generate synthetic data if none is provided ──────────────────────
    if data is None:
        print("No data provided. Generating synthetic house price data...\n")
        np.random.seed(random_state)
        n_samples = 500

        sqft        = np.random.randint(500, 5000, n_samples)
        bedrooms    = np.random.randint(1, 6, n_samples)
        bathrooms   = np.random.randint(1, 4, n_samples)
        age         = np.random.randint(0, 50, n_samples)
        garage      = np.random.randint(0, 3, n_samples)
        distance_cbd = np.random.uniform(1, 30, n_samples)

        price = (
            150 * sqft
            + 10_000 * bedrooms
            + 15_000 * bathrooms
            - 500 * age
            + 8_000 * garage
            - 3_000 * distance_cbd
            + 50_000
            + np.random.normal(0, 20_000, n_samples)
        )

        data = pd.DataFrame({
            "sqft":         sqft,
            "bedrooms":     bedrooms,
            "bathrooms":    bathrooms,
            "age":          age,
            "garage":       garage,
            "distance_cbd": distance_cbd,
            "price":        price,
        })

    # ── 2. Validate data ─────────────────────────────────────────────────────
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    print(f"Dataset shape : {data.shape}")
    print(f"Features      : {[c for c in data.columns if c != target_column]}")
    print(f"Target        : {target_column}\n")
    print("First 5 rows:")
    print(data.head(), "\n")
    print("Basic statistics:")
    print(data.describe(), "\n")

    # ── 3. Prepare features and target ───────────────────────────────────────
    X = data.drop(columns=[target_column])
    y = data[target_column]

    feature_names = X.columns.tolist()

    # ── 4. Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training samples : {len(X_train)}")
    print(f"Testing  samples : {len(X_test)}\n")

    # ── 5. Feature scaling ───────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── 6. Train the model ───────────────────────────────────────────────────
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # ── 7. Predictions ───────────────────────────────────────────────────────
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test  = model.predict(X_test_scaled)

    # ── 8. Metrics ───────────────────────────────────────────────────────────
    metrics = {
        "train": {
            "R²":   r2_score(y_train, y_pred_train),
            "MAE":  mean_absolute_error(y_train, y_pred_train),
            "MSE":  mean_squared_error(y_train, y_pred_train),
            "RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        },
        "test": {
            "R²":   r2_score(y_test, y_pred_test),
            "MAE":  mean_absolute_error(y_test, y_pred_test),
            "MSE":  mean_squared_error(y_test, y_pred_test),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        },
    }

    # ── 9. Feature importance (coefficients) ─────────────────────────────────
    feature_importance = pd.DataFrame({
        "Feature":     feature_names,
        "Coefficient": model.coef_,
        "Abs_Coeff":   np.abs(model.coef_),
    }).sort_values("Abs_Coeff", ascending=False).reset_index(drop=True)

    # ── 10. Report ────────────────────────────────────────────────────────────
    print("=" * 55)
    print("         LINEAR REGRESSION — PERFORMANCE REPORT")
    print("=" * 55)

    for split, m in metrics.items():
        print(f"\n  [{split.upper()} SET]")
        print(f"    R²   (explained variance) : {m['R²']:.4f}")
        print(f"    MAE  (mean abs error)      : ${m['MAE']:,.2f}")
        print(f"    RMSE (root mean sq error)  : ${m['RMSE']:,.2f}")

    print("\n  [MODEL PARAMETERS]")
    print(f"    Intercept : ${model.intercept_:,.2f}")
    print("\n  [FEATURE COEFFICIENTS (scaled)]")
    print(feature_importance.to_string(index=False))
    print("=" * 55)

    # ── 11. Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Linear Regression — House Price Prediction", fontsize=14, fontweight="bold")

    # (a) Actual vs Predicted
    ax = axes[0]
    ax.scatter(y_test, y_pred_test, alpha=0.5, color="steelblue", edgecolors="white", linewidths=0.3)
    lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Actual vs Predicted")
    ax.legend()

    # (b) Residuals
    residuals = y_test - y_pred_test
    ax = axes[1]
    ax.scatter(y_pred_test, residuals, alpha=0.5, color="darkorange", edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)