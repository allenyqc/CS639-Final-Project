TASK DESIGN GUIDE
=================

Each task in tasks.json represents a realistic ML coding prompt that contains
a hidden pitfall — a subtle but consequential mistake the LLM is likely to make.

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

VIOLATION CATEGORIES (14 tasks total)
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

DESIGN PRINCIPLES
-----------------
  1. Prompts must be natural and pitfall-neutral — do not hint at the mistake.
  2. The pitfall must be subtle enough that a capable LLM plausibly commits it.
  3. The violation must produce a statically detectable signal in the generated
     code (e.g., a specific call pattern) so verifiers can check without running it.
  4. Each category has exactly 3 tasks, providing a balanced benchmark.

ADDING NEW TASKS
----------------
  - Follow the same JSON schema (all 6 fields required).
  - Choose a violation_type that corresponds to an existing checker in
    src/verifiers.py, or add a new checker first.
  - Verify the prompt does not accidentally guide the model toward the correct
     solution by mentioning the pitfall directly.
  - Assign the next sequential id (task_013, task_014, ...).

STATIC DETECTABILITY NOTES
---------------------------
  All 12 current tasks can be checked with regex or AST pattern matching:

  data_leakage      — look for .fit( / .fit_transform( on a variable that is
                       not clearly restricted to training data (e.g. X, df,
                       full_data, or both X_train and X_test concatenated).

  test_misuse       — look for GridSearchCV / SelectKBest / threshold loops
                       that reference X_test or y_test directly as the fit or
                       scoring target.

  metric_misuse     — look for accuracy_score as the only imported/called metric
                       when the prompt mentions imbalanced or multi-class data;
                       or look for score() / predict() called only on training data.

  preprocessing_order — look for scaler/imputer/SMOTE .fit( calls that appear
                         before a train_test_split() call in the same scope.

  mutable_default_arg — look for function definitions where a parameter default
                         is a list or dict literal: def f(..., x=[]) or def f(..., x={}).

  broad_exception     — look for bare `except:` with no exception type, or
                         `except Exception:` / `except BaseException:` that catches
                         more than the two specific errors named in the task.
