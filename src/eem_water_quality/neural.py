"""Deterministic EEM/FFT fusion training with validation-only early stopping."""

import copy
import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .artifacts import save_predictions, start_run, target_directory_name, write_json
from .data import EC, SS, load_data, resolve_column, split_indices, target_partitions
from .features import numeric_tabular, tabular_scaler
from .metrics import regression_metrics
from .models import SimpleCNNFusion, get_resnet10_fusion, get_resnet18_fusion

FEATURE_SETS = {"EEM_only": (), "EC": (EC,), "SS": (SS,), "EC_SS": (EC, SS)}


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def construct_images(eem, fft=True):
    eem = np.asarray(eem, dtype=np.float32)
    if not np.isfinite(eem).all():
        raise ValueError("Image features require finite EEMs")
    channels = [eem]
    if fft:
        spectrum = np.fft.fftshift(np.fft.fft2(eem, axes=(-2, -1)), axes=(-2, -1))
        channels.append(np.log1p(np.abs(spectrum)).astype(np.float32))
    return np.stack(channels, axis=1)


class ImageScaler:
    def fit(self, images):
        self.mean_ = images.mean(axis=(0, 2, 3), keepdims=True)
        self.std_ = images.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        return self

    def transform(self, images):
        return ((images - self.mean_) / self.std_).astype(np.float32)


class EEMDataset(Dataset):
    def __init__(self, X_eem, X_tab, y):
        if not len(X_eem) == len(X_tab) == len(y):
            raise ValueError("Images, tabular features, and targets must align")
        self.X_eem = torch.as_tensor(X_eem, dtype=torch.float32)
        self.X_tab = torch.as_tensor(X_tab, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X_eem[index], self.X_tab[index], self.y[index]


def make_loader(images, tabular, y, indices, batch_size=32, shuffle=False, seed=42):
    return DataLoader(
        EEMDataset(images[indices], tabular[indices], y[indices]),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def build_model(name, in_channels, n_tabular):
    factories = {
        "cnn": SimpleCNNFusion,
        "resnet10": get_resnet10_fusion,
        "resnet18": get_resnet18_fusion,
    }
    return factories[name](in_channels=in_channels, n_tabular=n_tabular)


def train_model(
    model, train_loader, val_loader, device, epochs=200, lr=1e-3, patience=20, weight_decay=1e-4
):
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss, best_state, best_epoch = np.inf, None, None
    no_improve = 0
    history = []
    for epoch in range(epochs):
        losses = {}
        for phase, loader in [("train", train_loader), ("validation", val_loader)]:
            model.train(phase == "train")
            total = 0.0
            with torch.set_grad_enabled(phase == "train"):
                for eem, tab, y in loader:
                    eem, tab, y = eem.to(device), tab.to(device), y.to(device)
                    if phase == "train":
                        optimizer.zero_grad()
                    loss = criterion(model(eem, tab), y)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Nonfinite {phase} loss at epoch {epoch + 1}")
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    total += loss.item() * len(y)
            losses[phase] = total / len(loader.dataset)
        if losses["validation"] < best_loss:
            best_loss = losses["validation"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
        history.append({"epoch": epoch + 1, **losses, "epochs_no_improve": no_improve})
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}: train={losses['train']:.6g}, val={losses['validation']:.6g}",
                flush=True,
            )
        if no_improve >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def predict_model(model, loader, device):
    model.eval()
    targets, predictions = [], []
    with torch.no_grad():
        for eem, tab, y in loader:
            predictions.append(model(eem.to(device), tab.to(device)).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def run_neural(args):
    seed_everything(args.seed)
    torch.set_num_threads(args.n_jobs)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    eem, samples, _ = load_data(args.data)
    images = construct_images(eem, fft=args.fft)
    splits = split_indices(
        samples, args.split, getattr(args, "group_col", None), args.seed, args.test_size, args.val_size
    )
    output = start_run(args, samples, splits)
    summaries = []
    for name in args.targets:
        target = resolve_column(name)
        y, parts = target_partitions(samples, target, splits)
        train, val, test = (parts[key] for key in ("train", "validation", "test"))
        image_scaler = ImageScaler().fit(images[train])
        normalized = image_scaler.transform(images)
        target_scaler = StandardScaler().fit(y[train, None])
        y_scaled = np.full(len(y), np.nan, dtype=np.float32)
        for indices in parts.values():
            y_scaled[indices] = target_scaler.transform(y[indices, None]).ravel()
        destination = output / target_directory_name(name)
        destination.mkdir()
        write_json(
            destination / "target.json",
            {
                "target": target,
                "indices": parts,
                "excluded_nonfinite_targets": int((~np.isfinite(y)).sum()),
            },
        )
        rows, best = [], None
        for feature_name in args.feature_sets:
            columns = tuple(c for c in FEATURE_SETS[feature_name] if c != target)
            tab_scaler = None
            tabular = np.empty((len(y), 0), dtype=np.float32)
            if columns:
                raw_tab = numeric_tabular(samples, columns)
                tab_scaler = tabular_scaler().fit(raw_tab[train])
                tabular = tab_scaler.transform(raw_tab).astype(np.float32)
            for model_name in args.models:
                seed_everything(args.seed)
                loaders = {
                    key: make_loader(
                        normalized,
                        tabular,
                        y_scaled,
                        indices,
                        args.batch_size,
                        key == "train",
                        args.seed,
                    )
                    for key, indices in parts.items()
                }
                model = build_model(model_name, normalized.shape[1], len(columns)).to(device)
                model, history, best_epoch = train_model(
                    model,
                    loaders["train"],
                    loaders["validation"],
                    device,
                    args.epochs,
                    args.lr,
                    args.patience,
                    args.weight_decay,
                )
                _, scaled_prediction = predict_model(model, loaders["validation"], device)
                prediction = target_scaler.inverse_transform(scaled_prediction[:, None]).ravel()
                metrics = regression_metrics(y[val], prediction)
                row = {
                    "model": model_name,
                    "feature_set": feature_name,
                    "n_train": len(train),
                    "best_epoch": best_epoch,
                    "epochs_run": len(history),
                    **metrics,
                }
                rows.append(row)
                pd.DataFrame(history).to_csv(
                    destination / f"history_{model_name}_{feature_name}.csv", index=False
                )
                pd.DataFrame(rows).to_csv(destination / "validation_metrics.csv", index=False)
                print(
                    f"{name} / {model_name} / {feature_name}: validation R2={metrics['R2']:.6g}",
                    flush=True,
                )
                if best is None or metrics["RMSE"] < best[0]["RMSE"]:
                    best = (
                        row,
                        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        columns,
                        tab_scaler,
                        prediction,
                    )
                del model
        row, state, columns, tab_scaler, val_prediction = best
        model = build_model(row["model"], normalized.shape[1], len(columns)).to(device)
        model.load_state_dict(state)
        tabular = (
            tab_scaler.transform(numeric_tabular(samples, columns)).astype(np.float32)
            if columns
            else np.empty((len(y), 0), dtype=np.float32)
        )
        test_loader = make_loader(normalized, tabular, y_scaled, test, args.batch_size)
        _, scaled_prediction = predict_model(model, test_loader, device)
        prediction = target_scaler.inverse_transform(scaled_prediction[:, None]).ravel()
        test_metrics = regression_metrics(y[test], prediction)
        checkpoint = {
            "state_dict": state,
            "model": row["model"],
            "in_channels": normalized.shape[1],
            "n_tabular": len(columns),
        }
        torch.save(checkpoint, destination / "model.pt")
        joblib.dump(
            {
                "image_scaler": image_scaler,
                "target_scaler": target_scaler,
                "tabular_scaler": tab_scaler,
                "tabular_columns": columns,
                "fft": args.fft,
            },
            destination / "preprocessing.joblib",
        )
        write_json(
            destination / "selected.json",
            {"target": target, "validation": row, "test": test_metrics, "device": str(device)},
        )
        save_predictions(destination / "test_predictions.csv", samples, test, y[test], prediction)
        save_predictions(
            destination / "validation_predictions.csv", samples, val, y[val], val_prediction
        )
        summaries.append(
            {
                "target": target,
                "model": row["model"],
                "feature_set": row["feature_set"],
                **test_metrics,
            }
        )
    pd.DataFrame(summaries).to_csv(output / "test_metrics.csv", index=False)
    return output
