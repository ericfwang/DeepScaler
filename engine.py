"""
DeepScaler Engine — Predicting Peak CPU Utilization in Cloud Workloads

Implements the trained LightGBM + CatBoost ensemble from the OIT 367 final report
(Wang, Lee, Shankar, Maheshwari, 2026).

Two operating modes:
  - PRODUCTION: loads trained model artifacts from models/ directory
  - DEMO: uses report-derived heuristics for simulation (clearly flagged)
"""

from __future__ import annotations

import json
import math
import pickle
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional heavy dependencies — graceful degradation
# ---------------------------------------------------------------------------
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import catboost as cb
    _HAS_CB = True
except ImportError:
    _HAS_CB = False


# ═══════════════════════════════════════════════════════════════════════════
# Constants from the report
# ═══════════════════════════════════════════════════════════════════════════

SAFETY_BUFFER = 0.10  # 10 % buffer above predicted peak
UNDER_PENALTY = 10.0  # α = 10 in AsymRMSE
OPTIMAL_QUANTILE = 10.0 / 11.0  # τ ≈ 0.9091 (newsvendor solution)

# Default model directory (relative to this file)
DEFAULT_MODEL_DIR = Path(__file__).parent / "models"

# Scheduling classes (Table 1)
CLASS_NAMES = {
    0: "Best-effort",
    1: "Batch / burst",
    2: "Standard production",
    3: "High-priority production",
}
CLASS_DATASET_SHARE = {0: 0.08, 1: 0.28, 2: 0.14, 3: 0.50}
CLASS_ASYMRMSE = {0: 18.97, 1: 31.88, 2: 25.55, 3: 27.55}

# Savings constants (report Section 5)
VCPU_HOUR_PRICE = 0.0316
RECLAMATION_RATE = 0.28
ADDRESSABLE_FRACTION = 0.266
DEPLOYMENT_COVERAGE = 0.30
DEPLOYMENT_EFFECTIVENESS = 0.80
FLEET_VCPU_HOURS_YEAR = 134e9

# 72 feature names organized by group (for display/documentation)
FEATURE_GROUPS: dict[str, list[str]] = {
    "job_metadata": [
        "requested_cpus", "requested_memory", "priority",
        "scheduling_class", "job_type", "task_count", "duration_at_15min",
    ],
    "user_history": [
        "user_avg_cpu_util", "user_p50_cpu_util", "user_p90_cpu_util",
        "user_p95_cpu_util", "user_max_cpu_util", "user_std_cpu_util",
        "user_avg_memory_util", "user_p90_memory_util", "user_job_count",
        "user_burst_rate", "user_avg_burst_magnitude", "user_p90_burst_magnitude",
        "user_max_burst_magnitude", "user_avg_duration", "user_p90_duration",
        "user_class1_fraction", "user_recent_avg_cpu",
    ],
    "scalar_timeseries": [
        "cpu_avg_w0", "cpu_max_w0", "mem_avg_w0", "mem_max_w0",
        "cpu_avg_w1", "cpu_max_w1", "mem_avg_w1", "mem_max_w1",
        "cpu_avg_w2", "cpu_max_w2", "mem_avg_w2", "mem_max_w2",
    ],
    "histogram_stats": [
        "hist_mean_w0", "hist_std_w0", "hist_skew_w0", "hist_p75_w0", "hist_p95_w0",
        "hist_mean_w1", "hist_std_w1", "hist_skew_w1", "hist_p75_w1", "hist_p95_w1",
        "hist_mean_w2", "hist_std_w2", "hist_skew_w2", "hist_p75_w2", "hist_p95_w2",
    ],
    "trend_acceleration": [
        "cpu_slope_w01", "cpu_slope_w12", "cpu_slope_w02", "cpu_acceleration",
        "mem_slope_w01", "mem_slope_w12", "mem_slope_w02", "mem_acceleration",
    ],
    "interaction": [
        "burst_ratio_w0", "burst_ratio_w1", "burst_ratio_w2",
        "job_vs_user_p90_ratio", "cpu_memory_ratio", "peak_window_cpu_over_user_p90",
        "max_burst_ratio", "trend_x_burst", "user_history_percentile",
        "cpu_util_trajectory", "hist_tail_ratio", "cross_window_variance",
        "cpu_request_efficiency",
    ],
}
ALL_FEATURES: list[str] = [f for group in FEATURE_GROUPS.values() for f in group]


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

class Decision(str, Enum):
    RIGHT_SIZE = "right_size"
    REFUSE = "refuse"


@dataclass
class ReasoningStep:
    step: str
    detail: str


@dataclass
class PredictionResult:
    decision: Decision
    scheduling_class: int
    class_name: str
    predicted_peak_utilization: float | None = None
    safety_buffer_pct: float = SAFETY_BUFFER
    recommended_cpu_ceiling: float | None = None
    original_request: float = 1.0
    cpu_freed_pct: float | None = None
    reasoning: list[ReasoningStep] = field(default_factory=list)
    refusal_reason: str | None = None
    mode: str = "demo"
    actual_peak: float | None = None

    @property
    def estimated_savings_fraction(self) -> float | None:
        if self.cpu_freed_pct is not None:
            return self.cpu_freed_pct / 100.0
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def asym_rmse(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = UNDER_PENALTY) -> float:
    residuals = y_true - y_pred
    weights = np.where(residuals > 0, alpha, 1.0)
    return float(np.sqrt(np.mean(weights * residuals ** 2)))


# ═══════════════════════════════════════════════════════════════════════════
# DeepScaler Agent
# ═══════════════════════════════════════════════════════════════════════════

class DeepScalerAgent:
    """
    Agentic right-sizing system for cloud CPU workloads.

    Loads trained LightGBM + CatBoost models and the pre-computed sample of
    100 real jobs (with features) for interactive analysis.
    """

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._lgb_model = None
        self._cb_model = None
        self._feature_names: list[str] = []
        self._ensemble_weights: dict[str, float] = {}
        self._metadata: dict[str, Any] = {}
        self.mode = "demo"
        self._init_errors: list[str] = []

        # Sample data for the dashboard
        self.sample_jobs: pd.DataFrame | None = None
        self.sample_features: pd.DataFrame | None = None

        # Try multiple possible model directories
        candidates = [self.model_dir]
        # On Streamlit Cloud, __file__ may resolve differently
        candidates.append(Path(__file__).resolve().parent / "models")
        # Also try cwd
        candidates.append(Path.cwd() / "models")

        for candidate in candidates:
            if candidate.exists() and any(candidate.iterdir()):
                self.model_dir = candidate
                self._load_models()
                break
        else:
            self._init_errors.append(
                f"No model directory found. Tried: {[str(c) for c in candidates]}"
            )

    def _load_models(self) -> None:
        import logging
        logger = logging.getLogger("deepscaler")
        loaded = 0

        logger.info(f"Loading models from {self.model_dir}")
        logger.info(f"Model dir exists: {self.model_dir.exists()}")
        if self.model_dir.exists():
            logger.info(f"Model dir contents: {list(self.model_dir.iterdir())}")

        # Metadata (feature names, weights)
        meta_path = self.model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)
            self._feature_names = self._metadata.get("feature_names", [])
            self._ensemble_weights = self._metadata.get("ensemble_weights", {})

        # LightGBM
        lgb_path = self.model_dir / "lightgbm_q9091.txt"
        logger.info(f"LGB path exists: {lgb_path.exists()}, _HAS_LGB: {_HAS_LGB}")
        if lgb_path.exists() and _HAS_LGB:
            self._lgb_model = lgb.Booster(model_file=str(lgb_path))
            loaded += 1

        # CatBoost
        cb_path = self.model_dir / "catboost_q90.cbm"
        logger.info(f"CB path exists: {cb_path.exists()}, _HAS_CB: {_HAS_CB}")
        if cb_path.exists() and _HAS_CB:
            self._cb_model = cb.CatBoostRegressor()
            self._cb_model.load_model(str(cb_path))
            loaded += 1

        # Sample jobs + pre-computed features (load even if models fail)
        sj_path = self.model_dir / "sample_jobs.parquet"
        sf_path = self.model_dir / "sample_features.parquet"
        logger.info(f"Sample jobs exists: {sj_path.exists()}, features exists: {sf_path.exists()}")
        if sj_path.exists() and sf_path.exists():
            self.sample_jobs = pd.read_parquet(sj_path)
            self.sample_features = pd.read_parquet(sf_path)
            logger.info(f"Loaded {len(self.sample_jobs)} sample jobs")

        logger.info(f"Models loaded: {loaded}")
        if loaded >= 1:
            self.mode = "production"
        elif self.sample_jobs is not None:
            # Sample data loaded but no ML models — still useful for demo
            self.mode = "demo"

    def predict_from_features(self, X: np.ndarray, job_idx: int | None = None) -> float:
        """Run the trained ensemble on a pre-computed feature vector (1, n_features).
        Falls back to stored ensemble_prediction if models aren't available."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds: dict[str, float] = {}
        if self._lgb_model:
            preds["lgb"] = float(np.expm1(self._lgb_model.predict(X)[0]))
        if self._cb_model:
            preds["cb"] = float(np.expm1(self._cb_model.predict(X)[0]))
        if not preds:
            # No models loaded — use stored prediction from training
            if (job_idx is not None and self.sample_jobs is not None
                    and "ensemble_prediction" in self.sample_jobs.columns):
                return float(self.sample_jobs.iloc[job_idx]["ensemble_prediction"])
            return float(np.mean(X[0, :5]))

        w = self._ensemble_weights
        total_w = sum(w.get(k, 0.5) for k in preds)
        return sum(preds[k] * w.get(k, 0.5) / total_w for k in preds)

    def predict_job(self, job_idx: int) -> PredictionResult:
        """
        Predict for a sample job by index. Uses real models on real features.
        """
        if self.sample_jobs is None or self.sample_features is None:
            raise RuntimeError("No sample data loaded")

        job = self.sample_jobs.iloc[job_idx]
        X = self.sample_features.iloc[job_idx].values.astype(np.float64)
        sched_class = int(job["scheduling_class"])
        class_name = CLASS_NAMES.get(sched_class, f"Unknown ({sched_class})")
        requested_cpus = float(job["requested_cpus"])
        actual_peak = float(job["actual_peak"])
        n_feats = len(self._feature_names) if self._feature_names else X.shape[0]

        reasoning: list[ReasoningStep] = []
        reasoning.append(ReasoningStep(
            "Feature engineering",
            f"Loaded {n_feats} pre-computed features for job {job['collection_id']} "
            f"(instance {job['instance_index']}). "
            f"Scheduling class = {sched_class} ({class_name}). "
            f"Requested CPUs = {requested_cpus:.4f}.",
        ))

        # Class gate — refuse burst/best-effort jobs
        if sched_class == 1:
            reasoning.append(ReasoningStep(
                "Class 1 refusal — structural error floor",
                "Class 1 (Batch/burst) jobs exhibit CPU spikes of 10-300x after "
                "the 15-minute observation window. This creates a structural error "
                "floor (AsymRMSE = 31.88) that no early-window predictor can overcome. "
                "Two burst users alone account for ~36% of total squared error. "
                "Right-sizing would cause throttling failures. [Report Section 5]",
            ))
            return PredictionResult(
                decision=Decision.REFUSE,
                scheduling_class=sched_class,
                class_name=class_name,
                original_request=requested_cpus,
                reasoning=reasoning,
                actual_peak=actual_peak,
                refusal_reason=(
                    "REFUSED: Class 1 burst jobs have a structural error floor. "
                    "Burst events occur after the 15-minute observation window, "
                    "beyond what any early-window predictor can capture (report finding). "
                    "Premature right-sizing risks throttling failures."
                ),
                mode=self.mode,
            )

        if sched_class == 0:
            reasoning.append(ReasoningStep(
                "Class 0 refusal — unpredictable preemptible workload",
                "Class 0 (Best-effort) jobs are low-priority and preemptible with "
                "unpredictable CPU usage. Right-sizing provides minimal savings "
                "and risks disrupting already unstable workloads.",
            ))
            return PredictionResult(
                decision=Decision.REFUSE,
                scheduling_class=sched_class,
                class_name=class_name,
                original_request=requested_cpus,
                reasoning=reasoning,
                actual_peak=actual_peak,
                refusal_reason=(
                    "REFUSED: Class 0 best-effort jobs have unpredictable, preemptible "
                    "CPU usage. The report recommends right-sizing only for production "
                    "jobs (classes 2-3, ~64% of tasks)."
                ),
                mode=self.mode,
            )

        # Predict peak CPU utilization using real models (or stored predictions)
        # predicted_peak is a ratio: peak_usage / requested_cpus
        predicted_peak_ratio = self.predict_from_features(X, job_idx=job_idx)

        # Individual model predictions for reasoning
        lgb_pred = float(np.expm1(self._lgb_model.predict(X.reshape(1, -1))[0])) if self._lgb_model else None
        cb_pred = float(np.expm1(self._cb_model.predict(X.reshape(1, -1))[0])) if self._cb_model else None

        model_detail = []
        if lgb_pred is not None:
            model_detail.append(f"LightGBM (q=0.9091): {lgb_pred:.4f}")
        if cb_pred is not None:
            model_detail.append(f"CatBoost (q=0.90): {cb_pred:.4f}")
        w = self._ensemble_weights
        model_detail.append(f"Weights: LGB={w.get('lgb', 0):.1%}, CB={w.get('cat', 0):.1%}")

        # Convert ratio to absolute CPU: predicted_abs = ratio * requested
        predicted_abs = predicted_peak_ratio * requested_cpus

        reasoning.append(ReasoningStep(
            "Ensemble prediction",
            f"Predicted peak utilization ratio = {predicted_peak_ratio:.4f} "
            f"(i.e., {predicted_peak_ratio:.1f}x the requested {requested_cpus:.6f} CPUs "
            f"= {predicted_abs:.6f} absolute CPUs). "
            + " | ".join(model_detail)
            + f". Mode: {self.mode}.",
        ))

        # Safety: predicted absolute CPU + 10% buffer, capped at original request
        buffered_abs = predicted_abs * (1 + SAFETY_BUFFER)
        recommended_ceiling = min(buffered_abs, requested_cpus)

        reasoning.append(ReasoningStep(
            "Safety buffer applied",
            f"Predicted absolute CPU ({predicted_abs:.6f}) + 10% buffer = {buffered_abs:.6f}. "
            f"Capped at original request ({requested_cpus:.6f}). "
            f"Recommended CPU ceiling = {recommended_ceiling:.6f}.",
        ))

        cpu_freed_pct = max(0.0, (requested_cpus - recommended_ceiling) / requested_cpus * 100)

        reasoning.append(ReasoningStep(
            "Capacity reclamation",
            f"{cpu_freed_pct:.1f}% of requested CPU can be returned to the shared pool. "
            f"Actual peak ratio was {actual_peak:.4f} "
            f"({actual_peak * requested_cpus:.6f} absolute CPUs).",
        ))

        return PredictionResult(
            decision=Decision.RIGHT_SIZE,
            scheduling_class=sched_class,
            class_name=class_name,
            predicted_peak_utilization=predicted_peak_ratio,
            recommended_cpu_ceiling=recommended_ceiling,
            original_request=requested_cpus,
            cpu_freed_pct=cpu_freed_pct,
            reasoning=reasoning,
            actual_peak=actual_peak,
            mode=self.mode,
        )

    # ------------------------------------------------------------------
    # Savings calculator (report Section 5)
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_savings(
        fleet_vcpu_hours: float = FLEET_VCPU_HOURS_YEAR,
        addressable_fraction: float = ADDRESSABLE_FRACTION,
        vcpu_price: float = VCPU_HOUR_PRICE,
        coverage: float = DEPLOYMENT_COVERAGE,
        effectiveness: float = DEPLOYMENT_EFFECTIVENESS,
    ) -> dict[str, float]:
        addressable = fleet_vcpu_hours * addressable_fraction
        upper_bound = addressable * vcpu_price
        realistic = upper_bound * coverage * effectiveness
        return {
            "fleet_vcpu_hours_year": fleet_vcpu_hours,
            "addressable_fraction": addressable_fraction,
            "addressable_vcpu_hours": addressable,
            "reclamation_rate": RECLAMATION_RATE,
            "vcpu_hour_price": vcpu_price,
            "upper_bound_dollars": upper_bound,
            "coverage": coverage,
            "effectiveness": effectiveness,
            "realistic_savings_dollars": realistic,
        }
