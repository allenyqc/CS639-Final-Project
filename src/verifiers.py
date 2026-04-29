# =============================================================================
# src/verifiers.py
#
# PURPOSE:
#   Rule-based, static (no execution) checks that scan LLM-generated code for
#   common ML protocol violations. Each checker returns a violation message
#   string if a problem is found, or None if the code looks correct.
#
# TODO:
#   - Implement each checker using regex / AST pattern matching
#   - Handle edge cases: aliased imports, multi-line expressions, comments
#   - Consider using Python's `ast` module for more robust parsing
#   - Add more violation types as the task set expands
# =============================================================================

from __future__ import annotations

import re


def check_data_leakage(code: str) -> str | None:
    """
    Detect whether a scaler or encoder is fit on test or full data before
    the train/test split (a.k.a. data leakage).

    Heuristic: look for `.fit(` or `.fit_transform(` on a variable that
    appears to include test data (e.g., X, X_test, dataset).

    Parameters
    ----------
    code : str
        Raw Python source code as a string.

    Returns
    -------
    str | None
        A violation description string, or None if no leakage detected.

    TODO:
        - Distinguish fit-on-train (correct) from fit-on-full/test (violation)
        - Handle cases where the split happens after the fit call
    """
    raise NotImplementedError("TODO: implement check_data_leakage()")


def check_test_misuse(code: str) -> str | None:
    """
    Detect whether the test set is used more than once — e.g., for
    hyperparameter tuning, threshold selection, or early stopping.

    Heuristic: look for GridSearchCV / cross_val_score / threshold selection
    code that references X_test or y_test directly.

    Parameters
    ----------
    code : str
        Raw Python source code as a string.

    Returns
    -------
    str | None
        A violation description string, or None if no misuse detected.

    TODO:
        - Detect GridSearchCV(... , X_test, y_test) pattern
        - Detect manual loops over thresholds evaluated on test data
    """
    raise NotImplementedError("TODO: implement check_test_misuse()")


def check_metric_misuse(code : str) -> str | None:
    """
    Detect whether the code reports only accuracy on what appears to be an
    imbalanced classification task, without any complementary metric.

    Heuristic: if the task prompt mentions "imbalanced" and the code only
    calls accuracy_score (no f1_score, roc_auc_score, etc.), flag it.

    Parameters
    ----------
    code : str
        Raw Python source code as a string.

    Returns
    -------
    str | None
        A violation description string, or None if metrics look appropriate.

    TODO:
        - Parse which metrics are imported and which are called
        - Avoid false positives when accuracy is reported alongside other metrics
    """
    raise NotImplementedError("TODO: implement check_metric_misuse()")


def run_verifiers(code: str) -> list[str]:
    """
    Run all verifier checks and collect any violations found.

    Parameters
    ----------
    code : str
        Raw Python source code as a string.

    Returns
    -------
    list[str]
        A (possibly empty) list of violation description strings.
        Empty list means the code passed all checks.

    TODO: add new checker functions here as they are implemented.
    """
    checks = [
        check_data_leakage,
        check_test_misuse,
        check_metric_misuse,
    ]
    violations = []
    for check in checks:
        result = check(code)
        if result is not None:
            violations.append(result)
    return violations
