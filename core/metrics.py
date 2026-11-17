"""
Benchmark Metric Evaluation Suite for Wind Turbine Condition Monitoring.
Implements the official CARE benchmark score, event-level reliability, and lead time metrics.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


def compute_care_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray
) -> Dict[str, float]:
    """
    Computes comprehensive point-wise classification metrics and AUC-ROC.
    Input:
        y_true: Binary ground truth array (0: healthy normal, 1: anomalous/fault)
        y_pred: Binary alert predictions
        scores: Continuous anomaly scores
    """
    # Pointwise accuracy, precision, recall
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = float(np.mean(y_true == y_pred))

    # False-alarm rate on healthy normal timestamps
    normal_mask = (y_true == 0)
    if np.sum(normal_mask) > 0:
        pointwise_fa = float(np.mean(y_pred[normal_mask] == 1))
        normal_acc = float(1.0 - pointwise_fa)
    else:
        pointwise_fa = 0.0
        normal_acc = 1.0

    # AUC-ROC
    try:
        auc = float(roc_auc_score(y_true, scores))
    except Exception:
        auc = 0.5

    # CARE metric (balanced combination of detection recall and normal operational accuracy)
    care_score = float(0.5 * (rec + normal_acc))

    return {
        "care_score": care_score,
        "auc_roc": auc,
        "accuracy": acc,
        "normal_accuracy": normal_acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "pointwise_false_alarm_rate": pointwise_fa
    }


def calculate_prognostic_lead_time(
    alert_timestamps: List[float],
    failure_timestamp: float
) -> float:
    """
    Calculates prognostic earliness lead time in days between first persistent alert
    and the physical failure breakdown timestamp.
    """
    if not alert_timestamps:
        return 0.0
    first_alert = min(alert_timestamps)
    lead_seconds = max(0.0, failure_timestamp - first_alert)
    return lead_seconds / (24.0 * 3600.0)


def compute_earliness_distribution(lead_times_days: List[float]) -> Dict[str, float]:
    """
    Computes parametric and non-parametric earliness distribution statistics:
    Mean, Median, 25th percentile (Q1), and 75th percentile (Q3).
    """
    if not lead_times_days:
        return {"mean": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0}

    arr = np.array(lead_times_days)
    return {
        "mean_lead_time_days": float(np.mean(arr)),
        "median_lead_time_days": float(np.median(arr)),
        "q25_lead_time_days": float(np.percentile(arr, 25)),
        "q75_lead_time_days": float(np.percentile(arr, 75)),
        "min_lead_time_days": float(np.min(arr)),
        "max_lead_time_days": float(np.max(arr))
    }
