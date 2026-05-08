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
    X_test  : array-like of shape (n_test_samples,  n_features)
    y_train : array-like of shape (n_train_samples,)
    y_test  : array-like of shape (n_test_samples,)
    n_folds : int, number of cross-validation folds for meta-feature generation
    random_state : int, random seed for reproducibility

    Returns
    -------
    results : dict with keys
        'ensemble'   -> dict(accuracy, auc, f1)
        'base_models'-> dict of per-model dicts(accuracy, auc, f1)
        'meta_learner'        -> fitted meta-learner pipeline
        'base_model_pipelines'-> list of fitted base-model pipelines
    """
    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    n_classes = len(np.unique(y_train))
    multiclass = n_classes > 2

    # ------------------------------------------------------------------
    # 1. Define level-1 base model pipelines (scaling + calibrated model)
    # ------------------------------------------------------------------
    def make_base_pipelines():
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                    solver="lbfgs",
                    multi_class="auto",
                ),
                cv=3,
                method="sigmoid",
            )),
        ])

        rf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=random_state,
                    n_jobs=-1,
                ),
                cv=3,
                method="isotonic",
            )),
        ])

        gb = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=random_state,
                ),
                cv=3,
                method="isotonic",
            )),
        ])

        return [lr, rf, gb]

    base_names = ["LogisticRegression", "RandomForest", "GradientBoosting"]

    # ------------------------------------------------------------------
    # 2. Generate meta-features via out-of-fold (OOF) predictions
    # ------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Each base model contributes n_classes probability columns
    n_meta_cols = n_classes * len(base_names)
    meta_train = np.zeros((X_train.shape[0], n_meta_cols))

    print(f"Generating OOF meta-features with {n_folds}-fold CV ...")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr        = y_train[train_idx]

        fold_pipelines = make_base_pipelines()
        for m_idx, pipe in enumerate(fold_pipelines):
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_val)          # (n_val, n_classes)
            col_start = m_idx * n_classes
            col_end   = col_start + n_classes
            meta_train[val_idx, col_start:col_end] = proba

        print(f"  Fold {fold_idx + 1}/{n_folds} done.")

    # ------------------------------------------------------------------
    # 3. Train base models on the FULL training set
    # ------------------------------------------------------------------
    print("\nTraining base models on full training set ...")
    full_base_pipelines = make_base_pipelines()
    for name, pipe in zip(base_names, full_base_pipelines):
        pipe.fit(X_train, y_train)
        print(f"  {name} trained.")

    # ------------------------------------------------------------------
    # 4. Generate test meta-features using fully-trained base models
    # ------------------------------------------------------------------
    meta_test = np.zeros((X_test.shape[0], n_meta_cols))
    for m_idx, pipe in enumerate(full_base_pipelines):
        proba = pipe.predict_proba(X_test)
        col_start = m_idx * n_classes
        col_end   = col_start + n_classes
        meta_test[:, col_start:col_end] = proba

    # ------------------------------------------------------------------
    # 5. Train level-2 meta-learner on OOF meta-features
    # ------------------------------------------------------------------
    print("\nTraining meta-learner ...")
    meta_scaler = StandardScaler()
    meta_train_scaled = meta_scaler.fit_transform(meta_train)
    meta_test_scaled  = meta_scaler.transform(meta_test)

    meta_learner = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
        multi_class="auto",
        C=1.0,
    )
    meta_learner.fit(meta_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 6. Predict on test set with the meta-learner
    # ------------------------------------------------------------------
    ensemble_preds = meta_learner.predict(meta_test_scaled)
    ensemble_proba = meta_learner.predict_proba(meta_test_scaled)

    # ------------------------------------------------------------------
    # 7. Compute metrics helper
    # ------------------------------------------------------------------
    def compute_metrics(y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="weighted")
        if multiclass:
            auc = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted"
            )
        else:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        return {"accuracy": round(acc, 4), "auc": round(auc, 4), "f1": round(f1, 4)}

    ensemble_metrics = compute_metrics(y_test, ensemble_preds, ensemble_proba)

    # ------------------------------------------------------------------
    # 8. Individual base model performance on test set
    # ------------------------------------------------------------------
    base_model_metrics = {}
    for name, pipe in zip(base_names, full_base_pipelines):
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)
        base_model_metrics[name] = compute_metrics(y_test, preds, proba)

    # ------------------------------------------------------------------
    # 9. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  STACKING ENSEMBLE RESULTS")
    print("=" * 55)
    print(f"{'Model':<25} {'Accuracy':>9} {'AUC':>9} {'F1':>9}")
    print("-" * 55)
    for name, m in base_model_metrics.items():
        print(f"{name:<25} {m['accuracy']:>9.4f} {m['auc']:>9.4f} {m['f1']:>9.4f}")
    print("-" * 55)
    em = ensemble_metrics
    print(f"{'Stacking Ensemble':<25} {em['accuracy']:>9.4f} {em['auc']:>9.4f} {em['f1']:>9.4f}")
    print("=" * 55)

    return {
        "ensemble": ensemble_metrics,
        "base_models": base_model_metrics,
        "meta_learner": meta_learner,
        "meta_scaler": meta_scaler,
        "base_model_pipelines": full_base_pipelines,
        "base_model_names": base_names,
    }


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_iris
    from sklearn.model_selection import train_test_split

    print("=" * 55)
    print("  DEMO 1: Binary classification (Breast Cancer)")
    print("=" * 55)
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    results_binary = build_stacking_ensemble(X_tr, X_te, y_tr, y_te)

    print("\n" + "=" * 55)
    print("  DEMO 2: Multi-class classification (Iris)")
    print("=" * 55)
    data2 = load_iris()
    X2, y2 = data2.data, data2.target
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X2, y2, test_size=0.2, random_state=42, stratify=y2
    )
    results_multi = build_stacking_ensemble(X_tr2, X_te2, y_tr2, y_te2)