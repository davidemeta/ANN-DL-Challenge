# families/B_effnetv2

Family B: `tf_efficientnetv2_s.in21k_ft_in1k` fine-tuned on the team's shared split (Elia's
`folds.csv`, 2,995 clean images). Study, results and reasons: `DECISIONS.md`.

| file | what it is |
|---|---|
| `B_effnetv2_kaggle.ipynb` | run 1 notebook: augmentation study on fold 0, then 5 folds (30 epochs, hue-view TTA) |
| `B_effnetv2_run2_kaggle.ipynb` | run 2 notebook: 20 epochs, no hue TTA, tests resolution 320 / label cleaning / MixUp |
| `DECISIONS.md` | EDA summary, test-shift analysis, decisions, run 1 results, run 2 plan |
| `experiments_log.csv` | run 1 rows in Elia's log format (fold-0 study, 5 folds, OOF) |
| `ablation_run1.csv` | run 1 fold-0 augmentation study (kept / rejected steps) |
| `label_noise_run1.csv` | 128 training images confidently contradicted by the run-1 OOF models (likely mislabelled) |
| `submission_run1_D4.csv` | **run 1 submission** (the `submission.csv` of the run): average of the 5 fold models, D4 TTA (8 rotations/flips). OOF F1 0.827 clean / 0.8285 shifted |
| `submission_run1_D4_hue_weighted.csv` | run 1 variant: D4 TTA + hue-view TTA (original 0.5, ±120° channel shifts 0.25 each). OOF 0.8299 / 0.8303, within noise of the D4 one |
