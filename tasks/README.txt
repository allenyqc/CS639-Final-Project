TASK DESIGN GUIDE
=================

Each task in tasks.json (and tasks_hard.json) represents a realistic ML coding
prompt that contains hidden pitfalls — subtle but consequential mistakes an LLM
is likely to make.

FIELD DEFINITIONS
-----------------
  id               : Unique string identifier (e.g. "task_001")
  category         : High-level violation family (see categories below)
  prompt           : The exact prompt sent to the LLM. Should sound like a
                     normal homework or interview question — do NOT hint at
                     the pitfall in the prompt text.
  hidden_pitfall   : Human-readable description of the mistake we are testing for.
  violation_type   : Machine-readable label used by verifiers (must match a
                     checker name in src/verifiers.py).
  expected_behavior: What correct code should do instead.

==============================================================================
STANDARD BENCHMARK — tasks.json (16 tasks, ~20-80 lines each)
==============================================================================

VIOLATION CATEGORIES
------------------------------------------------------
  data_leakage        — test/validation statistics influence training transforms
    task_001  StandardScaler fit on full data (train + test)
    task_004  Target encoding means computed on full dataset
    task_005  PCA fit on full dataset before split

  test_misuse         — test set used more than once
    task_003  GridSearchCV evaluated directly on test set
    task_006  Classification threshold selected by optimizing test F1
    task_007  Mutual-information feature selection uses test labels

  metric_misuse       — misleading or incomplete evaluation metric
    task_002  Accuracy only on heavily imbalanced binary dataset
    task_008  Only training-set metric reported (no held-out evaluation)
    task_009  Micro-accuracy only on multi-class with class imbalance

  preprocessing_order — transforms applied to the full dataset before splitting
    task_010  SMOTE oversampling applied before train/test split
    task_011  Mean imputation fit on full dataset before split
    task_012  MinMaxScaler fit on full dataset before split

  coding              — general Python correctness pitfalls (non-ML)
    task_013  Mutable default argument causes state to persist across calls
    task_014  Bare except clause silently swallows unexpected exceptions
    task_015  Broad except in retry logic
    task_016  Multiple mutable default arguments in pipeline builder

==============================================================================
HARD BENCHMARK — tasks_hard.json (30 tasks, ~230 lines each)
==============================================================================

Designed to challenge frontier models (e.g. Claude Opus 4.7 thinking xhigh)
that achieve ~0% violation rate on the standard benchmark. Inspired by:
  - CodeMorph (ICSE 2025): semantic-preserving perturbation to expose true
    capability vs memorised patterns.
  - Agent Skills in the Wild (arXiv 2601.10338): security vulnerability
    patterns in AI agent code.

KEY DIFFICULTY DIFFERENCES FROM STANDARD TASKS
  - Multi-step pipelines requiring ~230 lines with pitfalls at non-obvious points
  - Composite violations: 2-3 interleaved pitfalls per task
  - Domain-specific complexity: time series, NLP, survival analysis, ranking,
    fairness, federated learning, security
  - Novel leakage forms not covered by simple fit-before-split patterns

HARD VIOLATION CATEGORIES (30 tasks)
------------------------------------------------------

  A. Temporal/Sequential Data Leakage (5 tasks)
     task_101  Time series forecasting with lag features + random split
     task_102  User click-stream prediction with future session features
     task_103  Walk-forward CV with full-data scaler and KFold
     task_104  Stock technical indicators computed on full price history
     task_105  Customer LTV with future transaction features

  B. Advanced Data Leakage (5 tasks)
     task_106  Stacking ensemble with in-sample meta-features
     task_107  TF-IDF fitted on full corpus before split
     task_108  Medical data without patient-level splitting
     task_109  Word2Vec trained on full corpus including test docs
     task_110  Data augmentation applied before train/test split

  C. Sophisticated Test Misuse (5 tasks)
     task_111  Early stopping on test set, best-epoch reporting
     task_112  Probability calibration fitted on test data
     task_113  AutoML model selection without nested CV
     task_114  Bayesian optimisation using test set as objective
     task_115  SHAP-based feature selection using test predictions

  D. Advanced Metric/Evaluation Errors (5 tasks)
     task_116  Survival analysis with accuracy instead of C-index
     task_117  Multi-label classification with single-label metrics
     task_118  Learning-to-rank with classification accuracy
     task_119  Fairness thresholds optimised on test data
     task_120  Anomaly detection evaluated with accuracy at <1% rate

  E. Statistical Protocol Violations (5 tasks)
     task_121  A/B/C/D test without multiple comparison correction
     task_122  Bootstrap model comparison with unpaired resampling
     task_123  Sequential experiment monitoring without alpha spending
     task_124  Feature importance on full data + no nested evaluation
     task_125  Post-hoc power analysis with observed effect size

  F. Security-Aware ML and System Pitfalls (5 tasks)
     task_126  Unsafe pickle deserialization + no input validation
     task_127  Hardcoded credentials + broad exception handling
     task_128  Federated learning with global statistics leakage
     task_129  Experiment tracker with mutable defaults + broad except
     task_130  Data anonymisation without k-anonymity verification

NEW VIOLATION TYPES (added for hard tasks)
------------------------------------------------------
  temporal_leakage      — random shuffle on time-indexed data, or missing
                           TimeSeriesSplit / date-based cutoff
  stacking_leakage      — base model in-sample predictions used as meta-features
                           without cross_val_predict
  group_leakage         — KFold / train_test_split without group-aware splitting
                           when data has patient/user/subject structure
  tfidf_leakage         — TF-IDF vectorizer fit on full corpus before split
  embedding_leakage     — Word2Vec / FastText trained on full corpus before split
  early_stopping_leak   — test data used for early stopping / validation callbacks
  calibration_leakage   — calibration model (Platt / isotonic) fit on test data
  nested_cv_missing     — GridSearchCV best_score_ reported as final result without
                           separate holdout evaluation
  augmentation_leakage  — data augmentation applied before train/test split
  multiple_testing      — multiple statistical tests without Bonferroni / BH correction
  survival_metric_misuse — classification accuracy on survival / censored data
  multilabel_metric_misuse — single-label metrics on multi-label data
  ranking_metric_misuse — classification / regression metrics on ranking task
  unsafe_deserialization — pickle.load / joblib.load without integrity verification
  credential_exposure   — hardcoded API keys / passwords / connection strings

DESIGN PRINCIPLES (both benchmarks)
-----------------
  1. Prompts must be natural and pitfall-neutral — do not hint at the mistake.
  2. The pitfall must be subtle enough that a capable LLM plausibly commits it.
  3. The violation must produce a statically detectable signal in the generated
     code (e.g., a specific call pattern) so verifiers can check without running it.
  4. Hard tasks have multi-step prompts generating ~230 lines, with 2-3
     interleaved pitfalls per task.

ADDING NEW TASKS
----------------
  - Follow the same JSON schema (all 6 fields required).
  - Choose a violation_type that corresponds to an existing checker in
    src/verifiers.py, or add a new checker first.
  - Verify the prompt does not accidentally guide the model toward the correct
    solution by mentioning the pitfall directly.
  - Standard tasks use ids task_001 - task_016.
  - Hard tasks use ids task_101 - task_130.

STATIC DETECTABILITY NOTES
---------------------------
  Original checkers (tasks 001-016):

  data_leakage      — look for .fit( / .fit_transform( on a variable that is
                       not clearly restricted to training data.
  test_misuse       — look for GridSearchCV / SelectKBest / threshold loops
                       that reference X_test or y_test.
  metric_misuse     — look for accuracy_score as the only metric, or
                       score()/predict() called only on training data.
  preprocessing_order — scaler/imputer/SMOTE .fit( before train_test_split().
  mutable_default_arg — def f(..., x=[]) or def f(..., x={}).
  broad_exception     — bare `except:` or `except Exception:`.

  New checkers (tasks 101-130):

  temporal_leakage    — train_test_split without shuffle=False in time-series
                         context; absence of TimeSeriesSplit.
  stacking_leakage    — .predict(X_train) used for meta-features without
                         cross_val_predict.
  group_leakage       — KFold/train_test_split without GroupKFold when
                         patient_id/user_id present.
  tfidf_leakage       — TfidfVectorizer.fit() before train_test_split().
  embedding_leakage   — Word2Vec() training before corpus split.
  early_stopping_leak — X_test in validation_data or EarlyStopping config.
  calibration_leakage — calibration .fit(X_test).
  nested_cv_missing   — best_score_ reported without separate test evaluation.
  augmentation_leakage — augmentation keywords before train_test_split().
  multiple_testing    — >=2 statistical test calls without multipletests/Bonferroni.
  survival_metric_misuse  — accuracy_score in survival context without C-index.
  multilabel_metric_misuse — accuracy_score in multi-label context without
                              hamming_loss or sample/macro-averaged metrics.
  ranking_metric_misuse   — accuracy_score/MSE in ranking context without NDCG/MAP.
  unsafe_deserialization  — pickle.load/joblib.load without hash/signature check.
  credential_exposure     — hardcoded API keys or passwords in string literals.
