"""Train logistic regression with NumPy and create a Titanic submission."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

console = Console()
app = typer.Typer(help=__doc__, add_completion=False)

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
continuous_features = ["Age", "Fare"]


class Preprocessing(BaseModel):
    # Keep pandas Series as Series, with Pydantic checking their container types.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    medians: pd.Series
    means: pd.Series
    scales: pd.Series


class TrainingResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, allow_inf_nan=False)

    W: np.ndarray
    b: float
    epochs: int = Field(ge=1)
    loss: float = Field(ge=0)
    converged: bool


def encode_features(data):
    """Select the same six columns in the same order without changing data."""
    missing = set(features) - set(data.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    X = data[features].copy()
    if not X["Sex"].isin(["male", "female"]).all():
        raise ValueError("Sex must contain only 'male' and 'female', with no missing values.")
    X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
    X = X.astype(float)

    if np.isinf(X.to_numpy()).any():
        raise ValueError("Features cannot contain infinity.")
    if X.drop(columns=continuous_features).isna().any().any():
        raise ValueError("Only Age and Fare may contain missing feature values.")
    return X


def fit_preprocessing(training_data):
    """Learn imputation and scaling from the current training rows only."""
    X = encode_features(training_data)
    medians = X[continuous_features].median()
    if medians.isna().any():
        raise ValueError("Training data needs at least one observed Age and Fare.")

    continuous = X[continuous_features].fillna(medians)
    means = continuous.mean()
    # ddof=0 uses the population standard deviation. A constant column stays safe.
    scales = continuous.std(ddof=0).replace(0.0, 1.0)
    return Preprocessing(medians=medians, means=means, scales=scales)


def transform_features(data, preprocessing):
    """Reuse learned statistics; never fit new statistics on validation/test."""
    X = encode_features(data)
    continuous = X[continuous_features].fillna(preprocessing.medians)
    X[continuous_features] = (continuous - preprocessing.means) / preprocessing.scales
    X = X.to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Preprocessing produced missing or non-finite values.")
    return X


def split_labeled_data(data, validation_fraction=0.2, seed=42):
    """Make a reproducible split with both survival classes in each part."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if "Survived" not in data or not data["Survived"].isin([0, 1]).all():
        raise ValueError("train.csv must contain Survived labels equal to 0 or 1.")

    rng = np.random.default_rng(seed)
    train_indices, validation_indices = [], []
    Y = data["Survived"].to_numpy()
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(Y == label))
        if len(indices) < 2:
            raise ValueError("Each survival class needs at least two labeled rows.")
        # Round up per class; keep at least one row of each class for training.
        n_validation = min(int(np.ceil(len(indices) * validation_fraction)), len(indices) - 1)
        validation_indices.extend(indices[:n_validation])
        train_indices.extend(indices[n_validation:])

    train = data.iloc[rng.permutation(train_indices)].reset_index(drop=True)
    validation = data.iloc[rng.permutation(validation_indices)].reset_index(drop=True)
    return train, validation


def z_cal(x, w, b):
    return x @ w + b


def sig_cal(z):
    """Compute sigmoid without exponentiating a large positive number."""
    z = np.asarray(z, dtype=float)
    p_hat = np.empty_like(z)
    positive = z >= 0
    p_hat[positive] = 1 / (1 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    p_hat[~positive] = exp_z / (1 + exp_z)
    return p_hat


def L_cal(y, z):
    """Mean binary cross-entropy from logits z, NOT probabilities.

    For binary y this equals mean(log(1 + exp(z)) - y*z).
    The signed-logit form also avoids subtracting nearly equal large values.
    """
    return float(np.mean(np.logaddexp(0, (1 - 2 * y) * z)))


def calculate_gradients(X, Y, p_hat):
    error = p_hat - Y
    dJ_w = X.T @ error / len(Y)
    dJ_b = float(np.mean(error))
    return dJ_w, dJ_b


def training(dJ_w, dJ_b, W, b, nu):
    return W - nu * dJ_w, b - nu * dJ_b


def train_model(
    X, Y, nu=0.1, max_epochs=20000, tolerance=1e-8,
    patience=20, log_every=2000, verbose=True,
):
    """Full-batch gradient descent: one update uses every training passenger."""
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 1 or len(X) != len(Y) or len(Y) == 0:
        raise ValueError("X must be a nonempty matrix and Y one label per row.")
    if not np.isfinite(X).all() or not np.isin(Y, [0, 1]).all():
        raise ValueError("X must be finite and Y must contain binary labels.")
    if not np.isfinite(nu) or nu <= 0 or max_epochs < 1:
        raise ValueError("Learning rate and epoch limit must be positive.")
    if not np.isfinite(tolerance) or tolerance < 0 or patience < 1 or log_every < 1:
        raise ValueError("Tolerance must be nonnegative; patience and log interval positive.")

    W = np.zeros(X.shape[1])
    b = 0.0
    Z = z_cal(X, W, b)
    previous_loss = L_cal(Y, Z)
    small_improvements = 0
    converged = False
    if verbose:
        console.print(f"Epoch {0:>6}: loss={previous_loss:.8f}")

    for epoch in range(1, max_epochs + 1):
        p_hat = sig_cal(Z)
        dJ_w, dJ_b = calculate_gradients(X, Y, p_hat)
        W, b = training(dJ_w, dJ_b, W, b, nu)

        # Calculate loss AFTER the update so it describes the returned W and b.
        Z = z_cal(X, W, b)
        loss = L_cal(Y, Z)
        if not np.isfinite(loss) or not np.isfinite(W).all() or not np.isfinite(b):
            raise ValueError("Training became non-finite. Try a smaller learning rate.")
        if loss > previous_loss + 1e-12:
            raise ValueError("Training loss increased. Try a smaller learning rate.")

        improvement = previous_loss - loss
        if 0 <= improvement < tolerance:
            small_improvements += 1
        else:
            small_improvements = 0
        converged = small_improvements >= patience

        if verbose and (epoch % log_every == 0 or converged or epoch == max_epochs):
            console.print(f"Epoch {epoch:>6}: loss={loss:.8f}, improvement={improvement:.2e}")
        if converged:
            break
        previous_loss = loss

    if verbose:
        reason = "Converged" if converged else "Reached epoch limit"
        console.print(f"{reason} after {epoch} epochs. Final loss={loss:.8f}")
    return TrainingResult(W=W, b=b, epochs=epoch, loss=loss, converged=converged)


def predict(X, W, b):
    return (sig_cal(z_cal(X, W, b)) >= 0.5).astype(int)


def evaluate_model(X, Y, W, b):
    """Evaluate frozen parameters; rows are actual, columns are predicted."""
    predictions = predict(X, W, b)
    confusion = np.array([
        [np.sum((Y == 0) & (predictions == 0)), np.sum((Y == 0) & (predictions == 1))],
        [np.sum((Y == 1) & (predictions == 0)), np.sum((Y == 1) & (predictions == 1))],
    ])
    return {
        "accuracy": float(np.mean(predictions == Y)),
        "loss": L_cal(Y, z_cal(X, W, b)),
        "confusion_matrix": confusion,
    }


@app.command()
def run(
    train: Annotated[Path, typer.Option(
        exists=True, dir_okay=False, readable=True, help="Labeled training CSV.",
    )] = Path("train.csv"),
    test: Annotated[Path, typer.Option(
        exists=True, dir_okay=False, readable=True, help="Kaggle test CSV.",
    )] = Path("test.csv"),
    output: Annotated[Path, typer.Option(help="Where to save predictions.")] = Path("submission.csv"),
    learning_rate: Annotated[float, typer.Option(min=0.0, help="Gradient descent step size.")] = 0.1,
    epochs: Annotated[int, typer.Option(min=1, help="Maximum training epochs.")] = 20000,
    tolerance: Annotated[float, typer.Option(min=0.0, help="Training loss improvement threshold.")] = 1e-8,
    patience: Annotated[int, typer.Option(min=1, help="Consecutive small improvements to stop.")] = 20,
    log_every: Annotated[int, typer.Option(min=1, help="Epochs between progress messages.")] = 2000,
    validation_fraction: Annotated[float, typer.Option(min=0.0, max=1.0, help="Fraction held out locally.")] = 0.2,
    seed: Annotated[int, typer.Option(min=0, help="Random seed for the split.")] = 42,
):
    """Validate locally, retrain on all labeled passengers, and save predictions."""
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise typer.BadParameter("Must be finite and greater than zero.", param_hint="--learning-rate")
    if not np.isfinite(tolerance):
        raise typer.BadParameter("Must be finite.", param_hint="--tolerance")
    if not 0 < validation_fraction < 1:
        raise typer.BadParameter("Must be strictly between zero and one.", param_hint="--validation-fraction")
    if output.resolve() in {train.resolve(), test.resolve()}:
        raise typer.BadParameter("Must differ from the input CSV paths.", param_hint="--output")

    labeled = pd.read_csv(train)
    test_data = pd.read_csv(test)
    if test_data.empty or "PassengerId" not in test_data:
        raise ValueError("test.csv must contain passengers and a PassengerId column.")
    passenger_ids = test_data["PassengerId"]
    if passenger_ids.isna().any() or passenger_ids.duplicated().any():
        raise ValueError("Test PassengerId values must be present and unique.")

    # 1. Split RAW labeled rows, then learn statistics only from local training.
    local_train, validation = split_labeled_data(labeled, validation_fraction, seed)
    local_preprocessing = fit_preprocessing(local_train)
    X_train = transform_features(local_train, local_preprocessing)
    X_validation = transform_features(validation, local_preprocessing)
    Y_train = local_train["Survived"].to_numpy(dtype=float)
    Y_validation = validation["Survived"].to_numpy(dtype=float)
    settings = dict(
        nu=learning_rate, max_epochs=epochs, tolerance=tolerance,
        patience=patience, log_every=log_every,
    )

    console.print(f"Local training: {len(local_train)} passengers; validation: {len(validation)}")
    local_model = train_model(X_train, Y_train, **settings)

    # 2. Freeze W, b and preprocessing; validation never updates them.
    metrics = evaluate_model(X_validation, Y_validation, local_model.W, local_model.b)
    console.print(f"Validation accuracy: {metrics['accuracy']:.2%}; loss: {metrics['loss']:.6f}")
    console.print("Confusion matrix: rows=actual, columns=predicted; order=0, 1")
    console.print(pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"],
    ).to_string())

    # 3. Start fresh: all labeled rows can now teach the final pipeline.
    final_preprocessing = fit_preprocessing(labeled)
    X_all = transform_features(labeled, final_preprocessing)
    X_test = transform_features(test_data, final_preprocessing)
    Y_all = labeled["Survived"].to_numpy(dtype=float)
    console.print(f"\nFinal training: all {len(labeled)} labeled passengers")
    final_model = train_model(X_all, Y_all, **settings)

    # 4. Kaggle supplies no labels for test.csv. Make predictions and save them.
    submission = pd.DataFrame({
        "PassengerId": passenger_ids.to_numpy(),
        "Survived": predict(X_test, final_model.W, final_model.b),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    console.print(f"\nSaved {len(submission)} predictions to {output.resolve()}")


def main():
    app()


if __name__ == "__main__":
    main()
