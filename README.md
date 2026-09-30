# RegressionPlanner

## Installation

Requires Python 3.10+.

With [uv](https://docs.astral.sh/uv/):

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or with plain pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all scripts from the repository root (e.g. `python src/real_time_plan_SNN_recontr.py`), since data, model and figure paths are relative to it. The `data/`, `models/` and `figs/` directories are not tracked and must be created/populated locally.
