"""Build the two team-wide files from the EDA cleaning proposal.

    python shared/make_shared_files.py

- shared/train_folds.csv  (id, label, group, fold): ALL 4,328 train images, same order as
  train.csv. 5 folds, stratified on the label, grouped by `dup_group`, so a grainy copy and its
  clean source (and near duplicates) always land in the same fold: no leakage through copies.
- shared/cleaning_ids.csv (id, category, reason, action): the 1,333 images to drop
  (grainy label-swapped copies, aliens, blank, broken, wrong-label duplicates).

How training uses them: drop the `cleaning_ids` rows from the TRAINING part of every fold.
On validation, report F1 on the kept images (main number, their labels are trustworthy) and on
the full fold (reference, as the playbook asks).

Both files are frozen at the Day 3 gate (Nov 28); changing them after that needs the whole team.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from an2dl_toolkit import create_and_save_folds  # noqa: E402

PROPOSAL = ROOT / 'eda' / 'outputs' / 'davide' / 'csv' / 'cleaning_proposal_davide.csv'
TRAIN_CSV = ROOT / 'data' / 'train.csv'
N_SPLITS = 5


def main():
    prop = pd.read_csv(PROPOSAL)
    train = pd.read_csv(TRAIN_CSV)
    assert prop['id'].tolist() == train['id'].tolist(), 'proposal and train.csv are not aligned'
    assert (prop['label'].values == train['label'].values).all(), 'labels differ from train.csv'

    df = train.assign(group=prop['dup_group'].values)
    folds = create_and_save_folds(df, target_col='label', group_col='group', n_splits=N_SPLITS)
    folds = folds[['id', 'label', 'group', 'fold']]
    assert folds['id'].tolist() == train['id'].tolist()
    assert (folds.groupby('group')['fold'].nunique() == 1).all(), 'a group spans two folds'

    drop = prop.loc[prop['action'] == 'drop', ['id', 'category', 'reason', 'action']]

    folds.to_csv(ROOT / 'shared' / 'train_folds.csv', index=False)
    drop.to_csv(ROOT / 'shared' / 'cleaning_ids.csv', index=False)

    kept = folds[~folds['id'].isin(drop['id'])]
    print(f'train_folds.csv: {len(folds)} rows, cleaning_ids.csv: {len(drop)} drops, '
          f'{len(kept)} kept')
    print('\nKept images per fold and class:')
    print(pd.crosstab(kept['fold'], kept['label'], margins=True).to_string())
    print('\nAll images per fold:', folds['fold'].value_counts().sort_index().tolist())


if __name__ == '__main__':
    np.random.seed(0)
    main()
