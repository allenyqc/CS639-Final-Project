import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


def build_stacking_ensemble(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Build a two-level stacking ensemble classifier.

    Parameters
    ----------
    X_train : array-like of shape (n_train_samples, n_features)
    X_test  : array-like of shape (n_test_samples, n_features)
    y_train : array-like of shape (n_train_samples,)
    y_test  : array-like of shape (n_test_samples,)
    n_folds : int, number of cross-validation folds for meta-feature generation
    random_state : int, random seed for reproducibility

    Returns
    -------
    results : dict containing ensemble metrics, base model metrics, and fitted objects
    """

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    n_classes = len(np.unique(y_train))
    is_binary = n_classes == 2

    # ------------------------------------------------------------------ #
    # 1. Feature scaling (fit on train, transform both splits)
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------ #
    # 2. Define level-1 base models (wrapped in calibrated pipelines)
    # ------------------------------------------------------------------ #
    base_model_definitions = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            multi_class="auto",
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=random_state,
        ),
    }

    # Wrap each base model with probability calibration (isotonic regression)
    base_models = {
        name: CalibratedClassifierCV(model, method="isotonic", cv=3)
        for name, model in base_model_definitions.items()
    }

    model_names = list(base_models.keys())
    n_models = len(model_names)

    # ------------------------------------------------------------------ #
    # 3. Generate meta-features via out-of-fold (OOF) predictions
    # ------------------------------------------------------------------ #
    # Shape: (n_train_samples, n_models * n_classes)  for multiclass
    #        (n_train_samples, n_models)               for binary (prob of class 1)
    if is_binary:
        meta_train = np.zeros((X_train_scaled.shape[0], n_models))
    else:
        meta_train = np.zeros((X_train_scaled.shape[0], n_models * n_classes))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Store fold-level models for later averaging (optional) or just refit on full data
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_fold_train = X_train_scaled[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train_scaled[val_idx]

        for m_idx, (name, model) in enumerate(base_models.items()):
            # Clone-like: create a fresh calibrated model each fold
            if name == "logistic_regression":
                fold_model = CalibratedClassifierCV(
                    LogisticRegression(
                        max_iter=1000,
                        random_state=random_state,
                        multi_class="auto",
                        solver="lbfgs",
                    ),
                    method="isotonic",
                    cv=3,
                )
            elif name == "random_forest":
                fold_model = CalibratedClassifierCV(
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,
                        min_samples_leaf=5,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                    method="isotonic",
                    cv=3,
                )
            else:  # gradient_boosting
                fold_model = CalibratedClassifierCV(
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        random_state=random_state,
                    ),
                    method="isotonic",
                    cv=3,
                )

            fold_model.fit(X_fold_train, y_fold_train)
            proba = fold_model.predict_proba(X_fold_val)  # (n_val, n_classes)

            if is_binary:
                meta_train[val_idx, m_idx] = proba[:, 1]
            else:
                meta_train[val_idx, m_idx * n_classes : (m_idx + 1) * n_classes] = proba

    # ------------------------------------------------------------------ #
    # 4. Refit base models on the FULL training set
    # ------------------------------------------------------------------ #
    fitted_base_models = {}
    for name, model in base_models.items():
        model.fit(X_train_scaled, y_train)
        fitted_base_models[name] = model

    # ------------------------------------------------------------------ #
    # 5. Generate test meta-features using fully-trained base models
    # ------------------------------------------------------------------ #
    if is_binary:
        meta_test = np.zeros((X_test_scaled.shape[0], n_models))
    else:
        meta_test = np.zeros((X_test_scaled.shape[0], n_models * n_classes))

    for m_idx, (name, model) in enumerate(fitted_base_models.items()):
        proba = model.predict_proba(X_test_scaled)
        if is_binary:
            meta_test[:, m_idx] = proba[:, 1]
        else:
            meta_test[:, m_idx * n_classes : (m_idx + 1) * n_classes] = proba

    # ------------------------------------------------------------------ #
    # 6. Scale meta-features before feeding to meta-learner
    # ------------------------------------------------------------------ #
    meta_scaler = StandardScaler()
    meta_train_scaled = meta_scaler.fit_transform(meta_train)
    meta_test_scaled = meta_scaler.transform(meta_test)

    # ------------------------------------------------------------------ #
    # 7. Train level-2 meta-learner (Logistic Regression)
    # ------------------------------------------------------------------ #
    meta_learner = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=random_state,
        multi_class="auto",
        solver="lbfgs",
    )
    meta_learner.fit(meta_train_scaled, y_train)

    # ------------------------------------------------------------------ #
    # 8. Predict with meta-learner on test meta-features
    # ------------------------------------------------------------------ #
    ensemble_preds = meta_learner.predict(meta_test_scaled)
    ensemble_proba = meta_learner.predict_proba(meta_test_scaled)

    # ------------------------------------------------------------------ #
    # 9. Compute ensemble metrics
    # ------------------------------------------------------------------ #
    def compute_metrics(y_true, y_pred, y_proba, is_binary):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="binary" if is_binary else "weighted")
        if is_binary:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted"
            )
        return {"accuracy": round(acc, 4), "auc": round(auc, 4), "f1": round(f1, 4)}

    ensemble_metrics = compute_metrics(y_test, ensemble_preds, ensemble_proba, is_binary)

    # ------------------------------------------------------------------ #
    # 10. Compute individual base model metrics for comparison
    # ------------------------------------------------------------------ #
    base_model_metrics = {}
    for name, model in fitted_base_models.items():
        preds = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)
        base_model_metrics[name] = compute_metrics(y_test, preds, proba, is_binary)

    # ------------------------------------------------------------------ #
    # 11. Assemble and return results
    # ------------------------------------------------------------------ #
    results = {
        "ensemble_metrics": ensemble_metrics,
        "base_model_metrics": base_model_metrics,
        "fitted_objects": {
            "scaler": scaler,
            "meta_scaler": meta_scaler,
            "base_models": fitted_base_models,
            "meta_learner": meta_learner,
        },
        "meta_features": {
            "meta_train": meta_train,
            "meta_test": meta_test,
        },
    }

    _print_results(results)
    return results


def _print_results(results: dict) -> None:
    """Pretty-print the performance comparison."""
    sep = "-" * 55
    print(sep)
    print("  TWO-LEVEL STACKING ENSEMBLE — PERFORMANCE REPORT")
    print(sep)

    print("\n[Base Model Performance on Test Set]")
    header = f"  {'Model':<25} {'Accuracy':>9} {'AUC':>9} {'F1':>9}"
    print(header)
    print("  " + "-" * 53)
    for name, metrics in results["base_model_metrics"].items():
        print(
            f"  {name:<25} {metrics['accuracy']:>9.4f} "
            f"{metrics['auc']:>9.4f} {metrics['f1']:>9.4f}"
        )

    print("\n[Ensemble (Meta-Learner) Performance on Test Set]")
    em = results["ensemble_metrics"]
    print(
        f"  {'stacking_ensemble':<25} {em['accuracy']:>9.4f} "
        f"{em['auc']:>9.4f} {em['f1']:>9.4f}"
    )
    print(sep + "\n")


# ------------------------------------------------------------------ #
# Demo / smoke test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Binary classification demo
    print("=" * 55)
    print("DEMO: Binary Classification")
    print("=" * 55)
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    binary_results = build_stacking_ensemble(X_tr, X_te, y_tr, y_te)

    # Multiclass classification demo
    print("=" * 55)
    print("DEMO: Multiclass Classification (3 classes)")
    print("=" * 55)
    X_mc, y_mc = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_tr_mc, X_te_mc, y_tr_mc, y_te_mc = train_test_split(
        X_mc, y_mc, test_size=0.2, random_state=42, stratify=y_mc
    )
    mc_results = build_stacking_ensemble(X_tr_mc, X_te_mc, y_tr_mc, y_te_mc)