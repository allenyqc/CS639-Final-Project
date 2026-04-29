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
                     verifier function name in src/verifiers.py).
  expected_behavior: What correct code should do instead.

VIOLATION CATEGORIES
--------------------
  data_leakage       — test/validation data influences training (e.g. fit
                       scaler on full dataset, use future data in features)
  test_misuse        — test set used more than once (hyperparameter tuning,
                       threshold selection, early stopping on test)
  metric_misuse      — misleading or wrong evaluation metric (accuracy on
                       imbalanced data, reporting train-set score only)
  preprocessing_order — transforms applied before the train/test split

DESIGN PRINCIPLES
-----------------
  1. Prompts should be natural and unambiguous about the task goal.
  2. The pitfall must be subtle enough that a capable LLM plausibly falls for it.
  3. There should be a clear, detectable signal in the generated code that
     reveals the violation (so verifiers can check it without executing code).
  4. Each (category, sub-type) pair should appear at least twice in the
     benchmark for statistical reliability.

TODO
----
  - Define 10–15 total tasks across all categories above
  - Ensure at least 3 tasks per category
  - Validate that each hidden_pitfall is statically detectable (or document
    why it requires execution-based checking)
  - Add a "difficulty" field (easy / medium / hard) once pilot results are in
