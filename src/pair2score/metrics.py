from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import cohen_kappa_score


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    return float(np.abs(y_pred_arr - y_true_arr).mean())


def quadratic_weighted_kappa(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    min_grade: float | None = None,
    max_grade: float | None = None,
    step: float = 0.5,
) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    if min_grade is None:
        min_grade = float(np.nanmin(y_true_arr))
    if max_grade is None:
        max_grade = float(np.nanmax(y_true_arr))
    y_pred_arr = np.clip(y_pred_arr, min_grade, max_grade)
    bins = np.arange(min_grade, max_grade + step, step)
    yt = np.digitize(y_true_arr, bins) - 1
    yp = np.digitize(y_pred_arr, bins) - 1
    if yt.size == 0 or len(np.unique(yt)) <= 1 or len(np.unique(yp)) <= 1:
        return 0.0
    return float(cohen_kappa_score(yt, yp, weights="quadratic"))


def compute_regression_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    min_grade: float | None = None,
    max_grade: float | None = None,
    step: float = 0.5,
) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "qwk": quadratic_weighted_kappa(y_true, y_pred, min_grade=min_grade, max_grade=max_grade, step=step),
    }
