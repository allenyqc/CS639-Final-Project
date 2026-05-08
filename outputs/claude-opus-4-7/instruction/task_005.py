from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def pca_svm_accuracy(X, y, n_components=10, test_size=0.2, random_state=42):
    """
    Reduce features to `n_components` via PCA, train an SVM classifier,
    and return test accuracy.

    Best practices applied:
    - Train/test split is done BEFORE any preprocessing/feature engineering.
    - Scaler and PCA are fit ONLY on training data (via Pipeline).
    - Test data is used only for final evaluation, never for fitting or
      hyperparameter selection.
    """
    # 1. Split BEFORE any feature engineering
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(set(y)) > 1 else None,
    )

    # 2. Pipeline ensures scaler & PCA are fit ONLY on training data
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
        ("svm", SVC(kernel="rbf", random_state=random_state)),
    ])

    pipeline.fit(X_train, y_train)

    # 3. Evaluate on held-out test data
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)