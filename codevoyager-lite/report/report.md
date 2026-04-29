# CodeVoyager-Lite: Can Simple Verifiers Improve the Trustworthiness of LLM-Generated ML Code?

**CS 639 Final Project**
Authors: [TODO: your names]
Date: [TODO: submission date]

---

## 1. Introduction

<!-- TODO:
  - Motivate the problem: LLMs generate code that looks right but violates ML best practices.
  - State the research question: does adding a verifier reduce such violations?
  - Summarize findings in 2–3 sentences (fill in after experiments).
-->

Large language models (LLMs) are increasingly used to generate machine learning code,
yet they can produce subtle protocol violations that are difficult to detect by inspection alone.
This project investigates whether lightweight rule-based verifiers can act as a safety net,
reducing the frequency of such violations without requiring execution of the generated code.

---

## 2. Method

<!-- TODO:
  - Describe the three-mode pipeline (baseline → instruction → verifier).
  - Diagram or pseudocode of the verifier + repair loop.
  - State which LLM is used and how it is prompted (temperature, max_tokens, etc.).
-->

### 2.1 Pipeline Overview

### 2.2 Modes

| Mode | Description |
|---|---|
| Baseline | Plain prompt, no additional guidance |
| Instruction | Prompt augmented with a list of best-practice warnings |
| Verifier | Baseline generation + static checks + conditional repair call |

### 2.3 LLM Configuration

---

## 3. Tasks

<!-- TODO:
  - Describe the benchmark: how many tasks, which violation categories.
  - Explain how tasks were designed (pitfall-neutral prompts).
  - Table: task_id | category | prompt summary | violation type
-->

---

## 4. Verifiers

<!-- TODO:
  - Describe each rule-based check (check_data_leakage, check_test_misuse, check_metric_misuse).
  - Explain the detection heuristic (regex, AST, keyword patterns).
  - Discuss limitations (false positives, aliased imports, runtime-only violations).
-->

---

## 5. Experiments

<!-- TODO:
  - Describe the experimental setup: how many runs per task, randomness controls.
  - Which LLM version / API endpoint used.
  - How outputs were saved and evaluated.
-->

---

## 6. Results

<!-- TODO:
  - Table: per-mode violation rate and functional success rate.
  - Bar charts from results/plots/.
  - Statistical significance test if n is large enough.
-->

---

## 7. Discussion

<!-- TODO:
  - Which mode performed best and why?
  - Were there tasks where verifier made things worse (over-repair)?
  - Qualitative examples of caught vs. missed violations.
-->

---

## 8. Limitations

<!-- TODO:
  - Small benchmark size (10–15 tasks).
  - Static analysis misses runtime violations.
  - Single LLM evaluated — results may not generalize.
  - Human evaluation of functional success is subjective.
-->

---

## References

<!-- TODO: add citations for related work on LLM code generation, ML reproducibility, etc. -->
