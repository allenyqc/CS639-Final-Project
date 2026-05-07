```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def generate_house_data(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic house price data."""
    np.random.seed(random_state)

    size_sqft = np.random.normal(1800, 600, n_samples).clip(500, 5000)
    bedrooms = np.random.randint(1, 7, n_samples)
    bathrooms = np.random.randint(1, 5, n_samples)
    age_years = np.random.randint(0, 50, n_samples)
    garage_spaces = np.random.randint(0, 4, n_samples)
    distance_to_city = np.random.uniform(1, 50, n_samples)

    # Price formula with some noise
    price = (
        150 * size_sqft
        + 10000 * bedrooms
        + 15000 * bathrooms
        - 2000 * age_years
        + 8000 * garage_spaces
        - 3000 * distance_to_city
        + np.random.normal(0, 30000, n_samples)
        + 50000  # base price
    ).clip(50000, 2000000)

    return pd.DataFrame({
        'size_sqft': size_sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age_years,
        'garage_spaces': garage_spaces,
        'distance_to_city_km': distance_to_city,
        'price': price
    })


def train_house_price_model(
    data: pd.DataFrame = None,
    target_column: str = 'price',
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
    plot_results: bool = True
) -> dict:
    """
    Train a linear regression model to predict house prices.

    Args:
        data: DataFrame with features and target. If None, synthetic data is generated.
        target_column: Name of the target column.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.
        scale_features: Whether to standardize features.
        plot_results: Whether to display evaluation plots.

    Returns:
        Dictionary containing the model, metrics, and other artifacts.
    """
    # Use synthetic data if none provided
    if data is None:
        print("No data provided. Generating synthetic house price data...\n")
        data = generate_house_data(random_state=random_state)

    print("=" * 60)
    print("         HOUSE PRICE PREDICTION - LINEAR REGRESSION")
    print("=" * 60)

    # ── Data Overview ──────────────────────────────────────────────
    print("\n📊 Dataset Overview:")
    print(f"   Total samples : {len(data):,}")
    print(f"   Features      : {data.drop(columns=[target_column]).shape[1]}")
    print(f"   Target column : {target_column}")
    print(f"\n{data.describe().round(2)}\n")

    # ── Prepare Features & Target ──────────────────────────────────
    X = data.drop(columns=[target_column])
    y = data[target_column]
    feature_names = X.columns.tolist()

    # ── Train / Test Split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"🔀 Train/Test Split:")
    print(f"   Training samples : {len(X_train):,}")
    print(f"   Testing  samples : {len(X_test):,}\n")

    # ── Feature Scaling ────────────────────────────────────────────
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled  = X_test.values

    # ── Train Model ────────────────────────────────────────────────
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    print("✅ Model trained successfully.\n")

    # ── Predictions ────────────────────────────────────────────────
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred  = model.predict(X_test_scaled)

    # ── Metrics ────────────────────────────────────────────────────
    def compute_metrics(y_true, y_pred, label):
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print(f"📈 {label} Performance:")
        print(f"   R²   (Coeff. of Determination) : {r2:.4f}")
        print(f"   RMSE (Root Mean Squared Error)  : ${rmse:,.2f}")
        print(f"   MAE  (Mean Absolute Error)      : ${mae:,.2f}")
        print(f"   MAPE (Mean Abs. Percentage Err) : {mape:.2f}%\n")
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}

    train_metrics = compute_metrics(y_train, y_train_pred, "Training")
    test_metrics  = compute_metrics(y_test,  y_test_pred,  "Testing")

    # ── Overfitting Check ──────────────────────────────────────────
    r2_gap = train_metrics['r2'] - test_metrics['r2']
    print("🔍 Overfitting Check:")
    if r2_gap < 0.05:
        print(f"   R² gap = {r2_gap:.4f} → Model generalises well ✅")
    elif r2_gap < 0.10:
        print(f"   R² gap = {r2_gap:.4f} → Slight overfitting ⚠️")
    else:
        print(f"   R² gap = {r2_gap:.4f} → Significant overfitting ❌")

    # ── Feature Coefficients ───────────────────────────────────────
    print("\n📌 Feature Coefficients (impact on price):")
    coef_df = pd.DataFrame({
        'Feature'    : feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)

    for _, row in coef_df.iterrows():
        direction = "↑" if row['Coefficient'] > 0 else "↓"
        print(f"   {direction} {row['Feature']:<25} : {row['Coefficient']:>12,.2f}")
    print(f"\n   Intercept: ${model.intercept_:,.2f}")

    # ── Plots ──────────────────────────────────────────────────────
    if plot_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("House