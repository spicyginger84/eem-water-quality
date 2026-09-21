import copy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from eem_water_quality.cli import main
from eem_water_quality.data import BOD, EC, SS
from eem_water_quality.neural import (
    ImageScaler,
    build_model,
    construct_images,
    make_loader,
    seed_everything,
    train_model,
)
from eem_water_quality.predict import predict_saved


def test_fft_channel_order_and_train_scaling():
    eem = np.ones((5, 8, 8), dtype=np.float32)
    images = construct_images(eem)
    np.testing.assert_array_equal(images[:, 0], eem)
    assert images[0, 1, 4, 4] == pytest.approx(np.log1p(64))
    scaler = ImageScaler().fit(images[:3])
    before = scaler.mean_.copy()
    assert np.isfinite(scaler.transform(images + 100)).all()
    np.testing.assert_array_equal(scaler.mean_, before)


@pytest.mark.parametrize("name", ["cnn", "resnet10", "resnet18"])
@pytest.mark.parametrize("tab_count", [0, 2])
def test_architectures(name, tab_count):
    torch.set_num_threads(1)
    seed_everything(42)
    model = build_model(name, 2, tab_count)
    prediction = model(torch.ones(2, 2, 8, 16), torch.ones(2, tab_count))
    assert prediction.shape == (2,)
    prediction.sum().backward()


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, image, tab):
        return self.weight.expand(len(image))


def test_early_stopping_restores_first_best_checkpoint():
    # Training pushes predictions positive; validation prefers negative predictions.
    images, tab = np.zeros((4, 1, 2, 2)), np.empty((4, 0))
    train = make_loader(images, tab, np.ones(4), np.arange(4), batch_size=4)
    val = make_loader(images, tab, -np.ones(4), np.arange(4), batch_size=4)
    original = ScalarModel()
    one, _, _ = train_model(copy.deepcopy(original), train, val, "cpu", epochs=1, lr=0.1)
    stopped, history, best_epoch = train_model(
        original, train, val, "cpu", epochs=20, patience=2, lr=0.1
    )
    assert best_epoch == 1
    assert len(history) == 3
    torch.testing.assert_close(one.weight, stopped.weight)


def test_seed_reproduces_initial_weights():
    seed_everything(9)
    first = build_model("cnn", 1, 0).state_dict()
    seed_everything(9)
    second = build_model("cnn", 1, 0).state_dict()
    for key in first:
        torch.testing.assert_close(first[key], second[key])


def test_neural_cli_reload(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(4)
    eem = rng.uniform(0, 1, (30, 8, 16)).astype(np.float32)
    samples = pd.DataFrame(
        {
            "Point": np.repeat(np.arange(10), 3),
            BOD: rng.uniform(1, 2, 30),
            EC: rng.normal(size=30),
            SS: rng.normal(size=30),
        }
    )
    data = tmp_path / "data"
    data.mkdir()
    np.save(data / "eem.npy", eem)
    samples.to_parquet(data / "samples.parquet")
    output = tmp_path / "run"
    main(
        [
            "neural",
            "--data",
            str(data),
            "--output",
            str(output),
            "--models",
            "cnn",
            "--feature-sets",
            "EEM_only",
            "EC_SS",
            "--epochs",
            "2",
            "--device",
            "cpu",
        ]
    )
    destination = output / "BOD"
    predictions = pd.read_csv(destination / "test_predictions.csv")
    indices = predictions.row_position.to_numpy()
    prediction = predict_saved(destination, eem[indices], samples.iloc[indices])
    np.testing.assert_allclose(prediction, predictions.y_pred, atol=1e-6)
