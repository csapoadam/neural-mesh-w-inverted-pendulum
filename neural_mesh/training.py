"""
Training and evaluation utilities for the Neural Mesh architecture.

Provides a ``train`` function (Adam optimiser by default) and an ``evaluate``
function that reports RMSE and R² for regression or accuracy + cross-entropy
for classification.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from neural_mesh.model import NeuralMeshModel


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error (we report RMSE externally)."""
    return F.mse_loss(pred.squeeze(-1), target.squeeze(-1))


def _classification_loss(log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood (cross-entropy) for integer class labels."""
    return F.nll_loss(log_probs, target.long())


def _r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination R² = 1 - SS_res / SS_tot.

    Returns ``nan`` when the target has zero variance (SS_tot == 0).
    """
    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    if ss_tot.item() == 0.0:
        return float("nan")
    return (1.0 - ss_res / ss_tot).item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: NeuralMeshModel,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    optimiser: str = "adam",
    clip_grad_norm: float | None = 5.0,
    restore_best: bool = True,
    verbose: bool = True,
    print_every: int = 20,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    recorder=None,
) -> Dict[str, list]:
    """Train *model* on the given data using an adaptive-moment SGD method.

    Parameters
    ----------
    model : NeuralMeshModel
    X_train, y_train : Tensors
        Training features (batch, n) and targets.
    epochs : int
    batch_size : int
    lr : float
        Initial learning rate.
    weight_decay : float
        L2 regularisation coefficient.
    optimiser : {"adam", "adamw", "sgd"}
    clip_grad_norm : float or None
        If set, clip the global gradient norm to this value each step.
        Steps with non-finite loss or gradients are skipped entirely
        (they would otherwise poison Adam's moment estimates -> NaN).
    restore_best : bool
        If True (default) and validation data is given, the model
        parameters from the best-validation-loss epoch are restored at
        the end of training (guards against late-training divergence).
    verbose : bool
    print_every : int
        How often (in epochs) to print progress.
    X_val, y_val : optional Tensors
        If provided, validation metrics are tracked each epoch.
    recorder : ActivationRecorder or None
        If provided, ``recorder.maybe_record(epoch, X_train)`` is called
        at the end of each epoch.

    Returns
    -------
    dict with keys "train_loss" and optionally "val_loss", each a list of
    per-epoch floats.
    """
    # Coerce numpy arrays to tensors
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if not isinstance(y_train, torch.Tensor):
        dtype = torch.long if model.task == "classification" else torch.float32
        y_train = torch.tensor(y_train, dtype=dtype)
    if X_val is not None and not isinstance(X_val, torch.Tensor):
        X_val = torch.tensor(X_val, dtype=torch.float32)
    if y_val is not None and not isinstance(y_val, torch.Tensor):
        dtype = torch.long if model.task == "classification" else torch.float32
        y_val = torch.tensor(y_val, dtype=dtype)

    # Select optimiser
    opt_cls = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }[optimiser.lower()]
    opt = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Select loss function
    loss_fn: Callable
    if model.task == "regression":
        loss_fn = _regression_loss
    else:
        loss_fn = _classification_loss

    # DataLoader
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history: Dict[str, list] = {"train_loss": []}
    if X_val is not None:
        history["val_loss"] = []
        if model.task == "classification":
            history["val_acc"] = []

    best_val = float("inf")
    best_state = None
    best_epoch = None
    n_skipped = 0

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            loss.backward()
            if clip_grad_norm is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad_norm
                )
                if not torch.isfinite(total_norm):
                    # Non-finite gradient: skip this step so it cannot
                    # poison the optimiser state.
                    opt.zero_grad()
                    n_skipped += 1
                    continue
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validation
        if X_val is not None:
            val_metrics = evaluate(model, X_val, y_val)
            history["val_loss"].append(val_metrics["loss"])
            if "accuracy" in val_metrics:
                history["val_acc"].append(val_metrics["accuracy"])
            if restore_best and val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_epoch = epoch
                best_state = {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                }
            model.train()

        # Activation recording
        if recorder is not None:
            recorder.maybe_record(epoch, X_train)

        if verbose and (epoch % print_every == 0 or epoch == 1):
            msg = f"Epoch {epoch:4d}/{epochs}  train_loss={avg_loss:.6f}"
            if model.task == "regression":
                msg += f"  RMSE={math.sqrt(avg_loss):.6f}"
                with torch.no_grad():
                    model.eval()
                    train_r2 = _r2_score(model(X_train), y_train)
                    model.train()
                msg += f"  R2={train_r2:.4f}"
            if X_val is not None:
                v = history["val_loss"][-1]
                msg += f"  val_loss={v:.6f}"
                if model.task == "regression":
                    msg += f"  val_R2={val_metrics['r2']:.4f}"
                if model.task == "classification":
                    msg += f"  val_acc={val_metrics['accuracy']:.2%}"
            print(msg)

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"Restored best validation model from epoch {best_epoch} "
                  f"(val_loss={best_val:.6f})")
    history["best_epoch"] = best_epoch
    if n_skipped > 0 and verbose:
        print(f"WARNING: skipped {n_skipped} optimiser steps due to "
              f"non-finite loss/gradients.")

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: NeuralMeshModel,
    X: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, float]:
    """Compute metrics on a dataset.

    Returns
    -------
    dict
        For regression:      {"loss": MSE, "rmse": RMSE, "r2": R²}
        For classification:  {"loss": CE, "accuracy": float}
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        dtype = torch.long if model.task == "classification" else torch.float32
        y = torch.tensor(y, dtype=dtype)

    model.eval()
    pred = model(X)

    if model.task == "regression":
        mse = F.mse_loss(pred.squeeze(-1), y.squeeze(-1)).item()
        return {"loss": mse, "rmse": math.sqrt(mse), "r2": _r2_score(pred, y)}
    else:
        ce = F.nll_loss(pred, y.long()).item()
        acc = (pred.argmax(dim=-1) == y.long()).float().mean().item()
        return {"loss": ce, "accuracy": acc}
