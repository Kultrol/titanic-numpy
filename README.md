# Titanic logistic regression with NumPy

A small, readable implementation of binary logistic regression for the Kaggle
Titanic data. NumPy performs the model calculations; pandas reads and prepares
the CSV files, and Rich prints progress. Typer handles the command line, and
Pydantic models hold preprocessing statistics and training results.
There is no scikit-learn dependency.

Read the [mathematics and pipeline guide](docs/titanic-mathematics.pdf) for the
probability model, loss and gradient derivation, preprocessing, validation, and
final prediction workflow. The repository includes the compiled PDF.

The standalone [Kaggle notebook](titanic_numpy_kaggle.ipynb) presents the same
workflow as explanatory sections and runnable cells. Attach the Titanic
competition data and run its cells from top to bottom.

Data source: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data).
The supplied training and test CSVs are included so the example and tests can
run locally. `gender_submission.csv` is Kaggle's sample submission;
`submission.csv` contains this model's predictions.

## Run

From this project folder, with Python 3.14 or later and `uv` installed:

```sh
git clone https://github.com/Kultrol/titanic-numpy.git
cd titanic-numpy
uv sync
uv run titanic
```

Keep the supplied `train.csv` and `test.csv` in the project folder. The command
evaluates a local model, trains a fresh final model, and writes `submission.csv`.
Upload that file to Kaggle; the program does not submit it automatically.

Use `uv run titanic --help` to see the typed command-line options and defaults.

You can also choose file locations and training settings:

```sh
uv run titanic --train train.csv --test test.csv --output submission.csv \
  --learning-rate 0.1 --epochs 20000 --tolerance 1e-8 --patience 20 \
  --log-every 2000 --validation-fraction 0.2 --seed 42
```

## Data flow

1. Split the raw labeled rows into approximately 80% training and 20% validation.
   The NumPy split is reproducible and stratified to preserve both class ratios.
2. Learn preprocessing from the local training rows: fill missing `Age` and
   `Fare` with their medians, then standardize both using their means and
   population standard deviations. A constant feature uses a scale of 1.
3. Train the local model from zero weights and bias. Freeze its parameters and
   preprocessing, then report validation loss, accuracy, and a confusion matrix.
4. Learn fresh preprocessing from **all** labeled `train.csv` rows and train a
   new model from zero. Validation rows can now contribute to final training.
5. Apply those full-training statistics to `test.csv`, predict survival, and
   save `PassengerId,Survived` with no index column, preserving test row order.

Validation and Kaggle test rows never determine their own medians or scaling.
Kaggle's `test.csv` has no survival labels, so accuracy is measured only on the
local validation split. Repeated tuning against that split can make its score
optimistic; it is a development estimate, not a guaranteed Kaggle score.

## The model

The original feature order is preserved:
`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`.
`Sex` uses the fixed mapping `male = 0`, `female = 1`. `Pclass` retains the
original ordinal numeric assumption. `SibSp` and `Parch` are counts; these three
columns and `Sex` remain unscaled. Only `Age` and `Fare` are continuous features.

For `m` passengers, the core calculations are:

```text
Z = X @ W + b
p_hat = sigmoid(Z)
J = mean(logaddexp(0, (1 - 2 * Y) * Z))
dJ_w = X.T @ (p_hat - Y) / m
dJ_b = mean(p_hat - Y)
W, b = W - nu * dJ_w, b - nu * dJ_b
```

`z_cal`, `sig_cal`, `L_cal`, and `training` retain the original mathematical
steps. **`L_cal(Y, Z)` takes logits, not probabilities.** Its loss is binary
cross-entropy computed directly from logits. A branched sigmoid and `logaddexp`
avoid overflow and the need to clip probabilities.

An epoch is one full batch gradient update. Training logs its loss at the chosen
interval and stops after `patience` consecutive improvements below `tolerance`,
or at the epoch limit. A nonfinite or materially increasing loss raises an
error so an unsuitable learning rate does not silently produce a submission.
Predicted probabilities at least `0.5` become `Survived = 1`.

The confusion matrix has **actual labels as rows** and **predictions as columns**,
both ordered `0, 1`: `[[true negatives, false positives],
[false negatives, true positives]]`.

## Check

From the project folder, run the pytest suite:

```sh
uv run pytest -q
```

`pytest` is a development dependency installed by `uv sync` or `uv run`.
Tests use plain assertions, parameterized cases, and pytest's `tmp_path` and
`monkeypatch` fixtures. Run one focused check with:

```sh
uv run pytest tests/test_main.py::test_manual_gradients_match_finite_differences -q
```

## Verified run

With the supplied CSV files and default settings (seed 42):

- Local training: 712 passengers; validation: 179 passengers.
- Validation accuracy: **80.45%** (144/179); validation loss: **0.424898**.
- Confusion matrix: `[[95, 15], [20, 49]]`.
- Local training converged after 3,776 epochs; final training after 3,850.
- Final training used all 891 labeled passengers and produced 418 predictions.

`submission.csv` is ready to upload. `training.log` records this run.

Reference: NumPy's [logaddexp documentation](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html)
describes the stable logarithm operation used by the loss.
