# chess_analysis

Predicts the result of a Lichess game from information available at move 20, using a
dataset I built myself from the Lichess public database rather than a cleaned Kaggle
file. The interesting part is not the model, it is where the signal actually lives.

**Holdout accuracy: 0.749 (log loss 0.505)** against two baselines: always pick white
(0.514) and pick whoever has the higher rating (0.634).

## Result

The held out test set is the last 20% of games by date, so nothing from the future
leaks into training.

| Model | Accuracy | Log loss |
| --- | --- | --- |
| Ensemble of four families | **0.7486** | **0.5052** |
| CatBoost, one family only | 0.7491 | 0.5011 |
| Logistic regression, one family only | 0.7485 | 0.5085 |
| Rating heuristic (`elo_diff > 0`) | 0.6337 | |
| Always white | 0.5135 | |

CatBoost on its own matches the ensemble. Once the engine evaluation is a feature,
the model families stop disagreeing, and that is the honest conclusion of the repo.

## Data

Lichess standard rated dump, `lichess_db_standard_rated_2017-04.pgn.zst`, streamed
directly with `zstandard` so the 20 GB archive never lands on disk in full. Games are
parsed with `python-chess`, one pass each, and kept only if they reach the feature
snapshot with a legal position.

After filtering (`base_sec > 0`, non terminal positions, infinities removed) the
training side of the split has **38,148 games**, and the test side is **3,799**.

The extracted CSVs are gitignored because they are large and reproducible. The
notebook rebuilds them from the URL.

## Features, all taken at move 20

Nothing after move 20 is visible to the model, so a prediction is a mid game
prediction, not a post mortem.

- **Clock state**: seconds left for each side, the difference, both normalised by the
  base time, and seconds spent per move.
- **Clock trend**: each side's clock midpoint at move 10 minus clock at move 20, which
  captures who is burning time in the second half.
- **Time control**: base seconds, increment, `has_increment`, `log1p(base_sec)`.
- **Position, white minus black**: material (standard piece values), number of developed
  knights and bishops, queen out flag, legal move count (mobility), castling flags.
- **Rating**: `elo_diff`, `elo_mean`, and the interactions `eval_end * level` and
  `eval_end * elo_gap`, which let the engine score mean something different at 1200
  than at 2400.
- **Engine evaluation**: Stockfish over the final FEN, run across worker threads, then
  clipped to +-1000 to stop a single mate score from dominating the scale.

Feature selection was done by iterative dropping with feature importances, and the kept
columns are written to `final_columns.csv` so the same list is reused everywhere.

## Method

1. Stream and parse games, extract features at move 20 (move 15 in an earlier pass).
2. Sort by date and time, split 80/20 chronologically.
3. Build a `ColumnTransformer` preprocessor and tune four model families with
   `RandomizedSearchCV` over 5 fold CV: XGBoost, CatBoost, RandomForest and Logistic
   Regression.
4. Keep the best configuration per family in `best_models_per_family.csv`.
5. Average the predicted probabilities of the family winners, threshold at 0.5.

## What the experiments showed

- **Ablation by move count.** Snapshots at 5, 10, 15 and 20 moves were compared with
  the same model. Accuracy improves as the game progresses, which is expected, and the
  move 20 numbers are the ones reported above.
- **Learning curve.** Logistic regression at 5 fold CV scores 0.6610 on 3,814 rows and
  only 0.6676 on 38,148 rows. Ten times the data buys half a point, so the ceiling here
  is the feature set, not the sample size.
- **Sigmoid on the engine eval.** Wrapping the evaluation in `1 / (1 + exp(-k * eval))`
  made logistic regression slightly worse (0.7462 versus 0.7485), because logistic
  regression already applies a sigmoid in its link function.
- **Increment and base time as separate features.** Adding `has_increment` gave CatBoost
  its best log loss (0.4998) with accuracy 0.7501, a small but repeatable gain.

## Run it

```bash
pip install pandas numpy scikit-learn xgboost catboost python-chess zstandard requests matplotlib
cd chess_analysis
jupyter notebook chess_dataset.ipynb
```

The notebook expects the CSVs it caches locally (`games_2017-04m20_*.csv`). Set
`SF_PATH` at the top of the feature section to your Stockfish binary before running the
evaluation step, or skip that cell and drop `eval_end` from `final_columns.csv`.

## Layout

```
chess_dataset.ipynb        the whole pipeline, in order
catboost_info/             CatBoost training logs from the tuned run
```

## Notes

- The dataset is one month of 2017 Lichess games, so treat the absolute accuracy as a
  property of that population. Blitz and rapid are pooled; splitting by time control is
  the obvious next experiment.
- `fen_end` is kept in the frame for the evaluation step but never fed to the models.
