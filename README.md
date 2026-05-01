# codevoyager-lite

A lightweight study of how simple rule-based verifiers improve the trustworthiness of LLM-generated machine learning code.

## Project Goal

Large language models can generate plausible-looking code that subtly violates ML best practices — e.g., leaking test data into training, misusing evaluation metrics, or fitting on the wrong split. This project evaluates whether adding a lightweight verifier layer (rule-based checks + optional repair) reduces such violations compared to a plain prompt baseline.

Three modes are compared:

| Mode | Description |
|---|---|
| `baseline` | Raw LLM output, no verification |
| `instruction` | Prompt-engineered to warn the model about pitfalls |
| `verifier` | LLM output passed through rule-based checkers; violations trigger a repair call |

## Repository Structure

```
codevoyager-lite/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── .gitignore
│
├── tasks/
│   ├── tasks.json          # Benchmark task definitions
│   └── README.txt          # Task design guidelines
│
├── src/
│   ├── __init__.py         # Package marker
│   ├── agent.py            # Main agent loop (run tasks in each mode)
│   ├── llm.py             # LLM API abstraction (OpenAI / Anthropic)
│   ├── prompts.py          # Prompt templates
│   ├── verifiers.py        # Rule-based code checkers
│   ├── repair.py           # LLM-based repair loop
│   ├── evaluate.py         # Scoring and metrics
│   └── utils.py            # Shared helpers
│
├── outputs/
│   ├── baseline/           # Raw LLM outputs
│   ├── instruction/        # Instruction-prompted outputs
│   └── verifier/           # Post-verification outputs
│
├── results/
│   ├── results.csv         # Aggregated evaluation results
│   └── plots/              # Generated figures
│
└── report/
    └── report.md           # Final project report
```

## TODO

- [x] Expand `tasks/tasks.json` to 10–15 diverse ML tasks
- [x] Integrate LLM API calls in `src/agent.py`
- [x] Implement rule-based checks in `src/verifiers.py`
- [x] Implement repair loop in `src/repair.py`
- [ ] Define and compute metrics in `src/evaluate.py`
- [ ] Generate result plots in `results/plots/`
- [ ] Write full report in `report/report.md`

## Quick Start

```bash
pip install -r requirements.txt

# Set your API key (pick one)
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."

# Optionally override model name
export LLM_MODEL="gpt-4o"

# Run the full benchmark (baseline + instruction + verifier)
python -m src.agent --mode all

# Or run a single mode
python -m src.agent --mode baseline
```
