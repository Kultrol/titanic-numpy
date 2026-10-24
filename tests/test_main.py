"""Focused pytest checks for the manual model and submission workflow."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
import pytest
from typer.testing import CliRunner


# titanic.__init__ also exports a function called main, so import the module
# explicitly to keep access to its individual learning functions.
model = importlib.import_module("titanic.main")
project = Path(__file__).resolve().parents[1]


def test_sigmoid_handles_extreme_logits_without_overflow():
    z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        probabilities = model.sig_cal(z)

    np.testing.assert_allclose(
        probabilities,
        [0.0, 0.2689414213699951, 0.5, 0.7310585786300049, 1.0],
    )
    assert np.isfinite(probabilities).all()


def test_loss_uses_logits_and_stays_finite_for_confident_errors():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    z = np.array([-1000.0, 1000.0, 1000.0, -1000.0])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        loss = model.L_cal(y, z)

    assert np.isfinite(loss)
    assert loss == pytest.approx(500.0)
    assert model.L_cal(y, np.zeros(4)) == pytest.approx(np.log(2.0))


def test_manual_gradients_match_finite_differences():
    X = np.array([[1.2, -0.3], [0.7, 1.4], [-1.1, 0.2], [0.0, -0.5]])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    W = np.array([0.2, -0.4])
    b = 0.15
    epsilon = 1e-6

    p_hat = model.sig_cal(model.z_cal(X, W, b))
    dJ_w, dJ_b = model.calculate_gradients(X, y, p_hat)

    numerical_weights = np.zeros_like(W)
    for feature_index in range(len(W)):
        step = np.zeros_like(W)
        step[feature_index] = epsilon
        loss_above = model.L_cal(y, model.z_cal(X, W + step, b))
        loss_below = model.L_cal(y, model.z_cal(X, W - step, b))
        numerical_weights[feature_index] = (loss_above - loss_below) / (2 * epsilon)

    loss_above = model.L_cal(y, model.z_cal(X, W, b + epsilon))
    loss_below = model.L_cal(y, model.z_cal(X, W, b - epsilon))
    numerical_bias = (loss_above - loss_below) / (2 * epsilon)

    np.testing.assert_allclose(dJ_w, numerical_weights, rtol=1e-6, atol=1e-8)
    assert dJ_b == pytest.approx(numerical_bias, rel=0, abs=1e-7)


def test_training_step_updates_weights_and_bias():
    W = np.array([0.5, -0.25])

    new_W, new_b = model.training(np.array([0.4, -0.8]), -0.3, W, 0.2, 0.1)

    np.testing.assert_allclose(new_W, [0.46, -0.17])
    assert new_b == pytest.approx(0.23)


def test_unseen_rows_use_training_medians_means_and_scales():
    training_rows = pd.DataFrame(
        {
            "Pclass": [1, 2, 3, 2],
            "Sex": ["female", "male", "female", "male"],
            "Age": [10.0, 20.0, np.nan, 40.0],
            "SibSp": [0, 1, 0, 2],
            "Parch": [0, 0, 1, 1],
            "Fare": [10.0, np.nan, 30.0, 50.0],
        }
    )
    unseen_rows = pd.DataFrame(
        {
            "Pclass": [3, 1],
            "Sex": ["male", "female"],
            "Age": [1000.0, np.nan],
            "SibSp": [0, 1],
            "Parch": [0, 2],
            "Fare": [np.nan, 10000.0],
        }
    )
    original_unseen_rows = unseen_rows.copy(deep=True)

    stats = model.fit_preprocessing(training_rows)
    age_scale = np.sqrt(118.75)
    fare_scale = np.sqrt(200.0)

    assert isinstance(stats, model.Preprocessing)
    assert isinstance(stats, BaseModel)
    for values in (stats.medians, stats.means, stats.scales):
        assert isinstance(values, pd.Series)
        assert list(values.index) == ["Age", "Fare"]
    np.testing.assert_allclose(stats.medians, [20.0, 30.0])
    np.testing.assert_allclose(stats.means, [22.5, 30.0])
    np.testing.assert_allclose(stats.scales, [age_scale, fare_scale])

    transformed = model.transform_features(unseen_rows, stats)
    expected = np.array(
        [
            [3, 0, (1000.0 - 22.5) / age_scale, 0, 0, 0],
            [1, 1, (20.0 - 22.5) / age_scale, 1, 2, (10000.0 - 30.0) / fare_scale],
        ]
    )
    np.testing.assert_allclose(transformed, expected)
    pd.testing.assert_frame_equal(unseen_rows, original_unseen_rows)


def test_constant_continuous_features_have_safe_unit_scales():
    rows = pd.DataFrame(
        {
            "Pclass": [1, 3],
            "Sex": ["male", "female"],
            "Age": [33.0, 33.0],
            "SibSp": [0, 1],
            "Parch": [0, 0],
            "Fare": [15.0, 15.0],
        }
    )

    stats = model.fit_preprocessing(rows)
    X = model.transform_features(rows, stats)

    np.testing.assert_allclose(stats.scales, [1.0, 1.0])
    np.testing.assert_allclose(X[:, [2, 5]], 0.0)
    assert np.isfinite(X).all()


def test_optimization_reduces_loss_and_reports_returned_models_loss():
    # These labels cannot be separated perfectly by a single threshold.
    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    initial_loss = model.L_cal(y, np.zeros(len(y)))

    result = model.train_model(X, y, max_epochs=10000, tolerance=1e-10, verbose=False)
    final_loss = model.L_cal(y, model.z_cal(X, result.W, result.b))

    assert isinstance(result, model.TrainingResult)
    assert isinstance(result, BaseModel)
    assert final_loss < initial_loss - 0.03
    assert result.loss == pytest.approx(final_loss, rel=0, abs=1e-12)
    assert 0 < result.epochs <= 10000
    assert np.isfinite(result.W).all()
    assert np.isfinite(result.b)


def test_convergence_stops_when_the_parameters_are_already_optimal():
    X = np.zeros((4, 2))
    y = np.array([0.0, 1.0, 0.0, 1.0])

    result = model.train_model(X, y, max_epochs=200, patience=3, verbose=False)

    assert result.converged
    assert result.epochs < 200
    assert result.loss == pytest.approx(np.log(2.0))


def test_epoch_limit_does_not_claim_convergence():
    X = np.array([[-1.0], [1.0]])
    y = np.array([0.0, 1.0])

    result = model.train_model(X, y, max_epochs=1, patience=20, verbose=False)

    assert result.epochs == 1
    assert not result.converged


def test_confusion_matrix_has_actual_rows_and_predicted_columns():
    X = np.array([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    W = np.array([1.0])

    result = model.evaluate_model(X, y, W, 0.0)

    # [[true negatives, false positives],
    #  [false negatives, true positives]]
    np.testing.assert_array_equal(result["confusion_matrix"], [[1, 1], [2, 2]])
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["loss"] == pytest.approx(model.L_cal(y, model.z_cal(X, W, 0.0)))


def test_prediction_includes_probability_exactly_one_half_as_survived():
    X = np.array([[-1e-10], [0.0], [1e-10]])

    predictions = model.predict(X, np.array([1.0]), 0.0)

    np.testing.assert_array_equal(predictions, [0, 1, 1])
    assert np.issubdtype(predictions.dtype, np.integer)


def test_split_is_reproducible_disjoint_and_stratified():
    labeled = pd.read_csv(project / "train.csv")

    training_rows, validation_rows = model.split_labeled_data(labeled)
    repeated_training, repeated_validation = model.split_labeled_data(labeled)
    different_training, _ = model.split_labeled_data(labeled, seed=43)

    pd.testing.assert_frame_equal(training_rows, repeated_training)
    pd.testing.assert_frame_equal(validation_rows, repeated_validation)
    training_ids = set(training_rows["PassengerId"])
    validation_ids = set(validation_rows["PassengerId"])
    assert not training_ids & validation_ids
    assert training_ids | validation_ids == set(labeled["PassengerId"])
    assert training_ids != set(different_training["PassengerId"])
    assert (len(training_rows), len(validation_rows)) == (712, 179)
    assert validation_rows["Survived"].value_counts().to_dict() == {0: 110, 1: 69}
    assert set(training_rows["Survived"]) == {0, 1}


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ([0, 0, 0, 0], "Each survival class needs at least two labeled rows"),
        ([0, 0, 1], "Each survival class needs at least two labeled rows"),
        ([0, 0, 1, 2], "Survived labels equal to 0 or 1"),
    ],
    ids=["one-class", "singleton-class", "nonbinary-label"],
)
def test_split_rejects_labels_that_cannot_form_both_classes(labels, message):
    with pytest.raises(ValueError, match=message):
        model.split_labeled_data(pd.DataFrame({"Survived": labels}))


@pytest.mark.parametrize("field", ["medians", "means", "scales"])
def test_preprocessing_requires_series_for_each_statistic(field):
    statistics = {
        "medians": pd.Series([20.0, 30.0], index=["Age", "Fare"]),
        "means": pd.Series([22.5, 30.0], index=["Age", "Fare"]),
        "scales": pd.Series([10.0, 15.0], index=["Age", "Fare"]),
    }

    with pytest.raises(ValidationError) as error:
        model.Preprocessing(**{**statistics, field: [1.0, 2.0]})

    assert len(error.value.errors()) == 1
    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == "is_instance_of"


@pytest.mark.parametrize(
    ("field", "invalid_value", "error_type"),
    [
        ("W", [0.1, -0.2], "is_instance_of"),
        ("b", float("nan"), "finite_number"),
        ("epochs", 0, "greater_than_equal"),
        ("loss", -0.1, "greater_than_equal"),
        ("loss", float("inf"), "finite_number"),
    ],
    ids=["list-weights", "nan-bias", "zero-epochs", "negative-loss", "infinite-loss"],
)
def test_training_result_rejects_invalid_containers_and_scalars(
    field, invalid_value, error_type
):
    values = {
        "W": np.array([0.1, -0.2]),
        "b": 0.0,
        "epochs": 1,
        "loss": 0.4,
        "converged": False,
    }

    with pytest.raises(ValidationError) as error:
        model.TrainingResult(**{**values, field: invalid_value})

    assert len(error.value.errors()) == 1
    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == error_type


def test_help_describes_options_without_reading_data_or_training(monkeypatch):
    read_csv = Mock()
    train_model = Mock()
    monkeypatch.setattr(model.pd, "read_csv", read_csv)
    monkeypatch.setattr(model, "train_model", train_model)

    result = CliRunner().invoke(model.app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--train" in result.output
    assert "--epochs" in result.output
    assert "--output" in result.output
    read_csv.assert_not_called()
    train_model.assert_not_called()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--epochs", "0"],
        ["--epochs", "two"],
        ["--learning-rate", "-0.1"],
        ["--validation-fraction", "1.2"],
    ],
    ids=["zero-epochs", "nonnumeric-epochs", "negative-learning-rate", "invalid-fraction"],
)
def test_invalid_options_are_rejected_before_reading_data_or_training(
    arguments, monkeypatch
):
    read_csv = Mock()
    train_model = Mock()
    monkeypatch.setattr(model.pd, "read_csv", read_csv)
    monkeypatch.setattr(model, "train_model", train_model)

    result = CliRunner().invoke(model.app, arguments)

    assert result.exit_code == 2, result.output
    assert "Invalid value" in result.output
    read_csv.assert_not_called()
    train_model.assert_not_called()


def test_command_writes_a_kaggle_submission_in_original_passenger_order(tmp_path):
    expected_ids = pd.read_csv(project / "test.csv")["PassengerId"]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(project / "src"), environment.get("PYTHONPATH")])
    )
    output = tmp_path / "submission.csv"

    completed = subprocess.run(
        [sys.executable, "-m", "titanic", "--output", str(output)],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

    submission = pd.read_csv(output)
    with output.open() as saved_file:
        assert saved_file.readline().strip() == "PassengerId,Survived"
    assert list(submission.columns) == ["PassengerId", "Survived"]
    assert len(submission) == len(expected_ids)
    np.testing.assert_array_equal(submission["PassengerId"], expected_ids)
    assert submission["Survived"].isin([0, 1]).all()
    assert pd.api.types.is_integer_dtype(submission["Survived"])
    assert "Validation accuracy:" in completed.stdout
    assert "Confusion matrix:" in completed.stdout
    assert "Final training: all 891 labeled passengers" in completed.stdout


def test_command_fits_local_and_full_pipelines_with_fresh_zero_parameters(
    tmp_path, monkeypatch
):
    arguments = [
        "--train", str(project / "train.csv"),
        "--test", str(project / "test.csv"),
        "--output", str(tmp_path / "submission.csv"),
        "--epochs", "1",
    ]
    fit_preprocessing = Mock(wraps=model.fit_preprocessing)
    train_model = Mock(wraps=model.train_model)
    training_step = Mock(wraps=model.training)
    monkeypatch.setattr(model.console, "print", Mock())
    monkeypatch.setattr(model, "fit_preprocessing", fit_preprocessing)
    monkeypatch.setattr(model, "train_model", train_model)
    monkeypatch.setattr(model, "training", training_step)

    result = CliRunner().invoke(model.app, arguments)

    assert result.exit_code == 0, result.output
    assert fit_preprocessing.call_count == 2
    assert train_model.call_count == 2
    assert training_step.call_count == 2
    assert [len(call.args[0]) for call in fit_preprocessing.call_args_list] == [712, 891]
    assert [len(call.args[0]) for call in train_model.call_args_list] == [712, 891]
    for call in training_step.call_args_list:
        np.testing.assert_array_equal(call.args[2], np.zeros(len(model.features)))
        assert call.args[3] == 0.0
