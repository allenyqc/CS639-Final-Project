import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    n_classes = len(np.unique(y_train))
    multiclass = n_classes > 2

    base_names = ["LogisticRegression", "RandomForest", "GradientBoosting"]

    def make_base_estimators():
        lr = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
            multi_class="auto",
        )
        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=random_state,
        )
        return [lr, rf, gb]

    def fit_calibrated_pipeline(X_fit, y_fit, base_estimator, calibration_method="sigmoid"):
        """
        Fit a scaler+estimator on a training split, then calibrate using
        a held-out calibration split — all derived from X_fit/y_fit only.
        No test data is ever seen during fitting or calibration.
        """
        # Split X_fit into train portion and calibration portion
        X_fit_tr, X_fit_cal, y_fit_tr, y_fit_cal = train_test_split(
            X_fit, y_fit,
            test_size=0.2,
            random_state=random_state,
            stratify=y_fit,
        )

        # Fit scaler on training portion only
        scaler = StandardScaler()
        X_fit_tr_scaled  = scaler.fit_transform(X_fit_tr)
        X_fit_cal_scaled = scaler.transform(X_fit_cal)

        # Fit base estimator on training portion only
        base_estimator.fit(X_fit_tr_scaled, y_fit_tr)

        # Calibrate using prefit estimator on calibration portion only
        calibrated = CalibratedClassifierCV(
            estimator=base_estimator,
            cv="prefit",
            method=calibration_method,
        )
        calibrated.fit(X_fit_cal_scaled, y_fit_cal)

        return scaler, calibrated

    def predict_proba_with_pipeline(scaler, calibrated_clf, X):
        X_scaled = scaler.transform(X)
        return calibrated_clf.predict_proba(X_scaled)

    def predict_with_pipeline(scaler, calibrated_clf, X):
        X_scaled = scaler.transform(X)
        return calibrated_clf.predict(X_scaled)

    calibration_methods = ["sigmoid", "isotonic", "isotonic"]

    # ------------------------------------------------------------------
    # 2. Generate meta-features via out-of-fold (OOF) predictions
    # ------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    n_meta_cols = n_classes * len(base_names)
    meta_train = np.zeros((X_train.shape[0], n_meta_cols))

    print(f"Generating OOF meta-features with {n_folds}-fold CV ...")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        # Only use fold training data for fitting; fold val data is never used for fitting
        X_tr_fold = X_train[train_idx]
        y_tr_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]

        fold_estimators = make_base_estimators()
        for m_idx, (base_est, cal_method) in enumerate(zip(fold_estimators, calibration_methods)):
            # fit_calibrated_pipeline only uses X_tr_fold, y_tr_fold
            scaler, calibrated_clf = fit_calibrated_pipeline(
                X_tr_fold, y_tr_fold, base_est, cal_method
            )
            # predict on val fold (not used for fitting)
            proba = predict_proba_with_pipeline(scaler, calibrated_clf, X_val_fold)
            col_start = m_idx * n_classes
            col_end   = col_start + n_classes
            meta_train[val_idx, col_start:col_end] = proba

        print(f"  Fold {fold_idx + 1}/{n_folds} done.")

    # ------------------------------------------------------------------
    # 3. Train base models on the FULL training set only
    # ------------------------------------------------------------------
    print("\nTraining base models on full training set ...")
    full_base_estimators = make_base_estimators()
    full_scalers = []
    full_calibrated_clfs = []

    for name, base_est, cal_method in zip(base_names, full_base_estimators, calibration_methods):
        # fit_calibrated_pipeline uses only X_train, y_train
        scaler, calibrated_clf = fit_calibrated_pipeline(
            X_train, y_train, base_est, cal_method
        )
        full_scalers.append(scaler)
        full_calibrated_clfs.append(calibrated_clf)
        print(f"  {name} trained and calibrated.")

    # ------------------------------------------------------------------
    # 4. Generate test meta-features using fully-trained base models
    #    (X_test is only used for prediction, never for fitting)
    # ------------------------------------------------------------------
    meta_test = np.zeros((X_test.shape[0], n_meta_cols))
    for m_idx, (scaler, calibrated_clf) in enumerate(zip(full_scalers, full_calibrated_clfs)):
        proba = predict_proba_with_pipeline(scaler, calibrated_clf, X_test)
        col_start = m_idx * n_classes
        col_end   = col_start + n_classes
        meta_test[:, col_start:col_end] = proba

    # ------------------------------------------------------------------
    # 5. Train level-2 meta-learner on OOF meta-features (training data only)
    # ------------------------------------------------------------------
    print("\nTraining meta-learner ...")
    # Scaler fit only on meta_train (derived from training data OOF predictions)
    meta_scaler = StandardScaler()
    meta_train_scaled = meta_scaler.fit_transform(meta_train)
    # Transform meta_test using scaler fit on meta_train only
    meta_test_scaled  = meta_scaler.transform(meta_test)

    meta_learner = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
        multi_class="auto",
        C=1.0,
    )
    # Meta-learner fit only on training meta-features and training labels
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
    #    (X_test only used for prediction, never fitting)
    # ------------------------------------------------------------------
    base_model_metrics = {}
    for name, scaler, calibrated_clf in zip(base_names, full_scalers, full_calibrated_clfs):
        preds = predict_with_pipeline(scaler, calibrated_clf, X_test)
        proba = predict_proba_with_pipeline(scaler, calibrated_clf, X_test)
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
        "base_scalers": full_scalers,
        "base_calibrated_clfs": full_calibrated_clfs,
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
    # Split FIRST, then pass to build_stacking_ensemble
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    results_binary = build_stacking_ensemble(X_tr, X_te, y_tr, y_te)

    print("\n" + "=" * 55)
    print("  DEMO 2: Multi-class classification (Iris)")
    print("=" * 55)
    data2 = load_iris()
    X2, y2 = data2.data, data2.target
    # Split FIRST, then pass to build_stacking_ensemble
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X2, y2, test_size=0.2, random_state=42, stratify=y2
    )
    results_multi = build_stacking_ensemble(X_tr2, X_te2, y_tr2, y_te2)