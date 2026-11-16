# AN2DL Challenge

Shared repo for the Artificial Neural Networks & Deep Learning challenge (Polimi).

## Layout
- `src/` – shared code (lead only): `an2dl_toolkit.py`, `metric.py`, `data.py`, `submit.py`
- `shared/` – `train_folds.csv`, `cleaning_ids.csv` (frozen on Day 3, tag `freeze-day3`)
- `eda/` – EDA and cleaning notebooks
- `families/` – one folder per person/family
- `runs/` – one small JSON per run (`{family}_{model}_s{seed}.json`)
- `ensemble/` – blending and post-processing
- `tests/` – `smoke_test.py`
- `docs/` – Playbook: [Part I – Strategy](docs/AN2DL_Playbook_Part1_Strategy.md), [Part II – Code](docs/AN2DL_Playbook_Part2_Code.md)

## Setup
```bash
pip install -r requirements.txt
python tests/smoke_test.py   # must pass on Colab/GPU before anything else
```

## Rules
1. Only the lead edits `src/` and `shared/`; others open issues/PRs.
2. One branch per person (`famA/...`); merge to `main` only what others need.
3. Every run logs its commit hash; dirty runs are excluded from the final ensemble.
4. Hyperparameters live in YAML/CLI flags, not notebook cells.
5. Never commit data, weights, `.npy` logits or tokens (use the shared drive / Colab Secrets).
