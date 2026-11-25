# AN2DL Challenge

Shared repo for the Artificial Neural Networks & Deep Learning challenge (Polimi), A.Y. 2026-27: **"Flying SPAGHETTI"** — 10-class classification of 256×256 RGB telescope images, metric weighted F1, Nov 25 → Dec 8 2026 ([Kaggle](https://www.kaggle.com/competitions/an2dl2627)).

## Layout
- `src/` – shared code (lead only): `an2dl_toolkit.py`, `metric.py`, `data.py`, `submit.py`
- `shared/` – `train_folds.csv`, `cleaning_ids.csv` (frozen on Day 3, tag `freeze-day3`)
- `eda/` – EDA and cleaning notebooks
- `families/` – one folder per person/family
- `runs/` – one small JSON per run (`{family}_{model}_s{seed}.json`)
- `ensemble/` – blending and post-processing
- `tests/` – `smoke_test.py`
- `docs/` – Playbook: [Part 0 – Challenge 2026-27](docs/AN2DL_Playbook_Part0_Challenge_2627.md) (read first), [Part I – Strategy](docs/AN2DL_Playbook_Part1_Strategy.md), [Part II – Code](docs/AN2DL_Playbook_Part2_Code.md)

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
