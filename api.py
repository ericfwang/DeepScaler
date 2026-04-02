"""
DeepScaler API — FastAPI backend for cloud workload right-sizing.

Wraps the DeepScalerAgent engine and exposes prediction, reasoning,
and savings estimation endpoints.

Run:  uvicorn api:app --reload --port 8000
"""

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine import (
    ALL_FEATURES,
    CLASS_ASYMRMSE,
    CLASS_DATASET_SHARE,
    CLASS_NAMES,
    FEATURE_GROUPS,
    DeepScalerAgent,
    Decision,
)

# ═══════════════════════════════════════════════════════════════════════════
# App setup
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="DeepScaler API",
    version="2.0.0",
    description=(
        "Predicts peak CPU utilization for cloud workloads from the first "
        "15-minute observation window. Uses trained LightGBM + CatBoost "
        "ensemble from Wang, Lee, Shankar & Maheshwari (2026)."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = DeepScalerAgent()


# ═══════════════════════════════════════════════════════════════════════════
# Request / Response models
# ═══════════════════════════════════════════════════════════════════════════

class ReasoningStepResponse(BaseModel):
    step: str
    detail: str


class PredictionResponse(BaseModel):
    decision: str
    scheduling_class: int
    class_name: str
    predicted_peak_utilization: Optional[float]
    safety_buffer_pct: float
    recommended_cpu_ceiling: Optional[float]
    original_request: float
    cpu_freed_pct: Optional[float]
    actual_peak: Optional[float]
    reasoning: list[ReasoningStepResponse]
    refusal_reason: Optional[str]
    mode: str


class SavingsInput(BaseModel):
    fleet_vcpu_hours_year: float = Field(134e9, description="Total fleet vCPU-hours/year")
    addressable_fraction: float = Field(0.266, ge=0, le=1)
    vcpu_hour_price: float = Field(0.0316, ge=0)
    coverage: float = Field(0.30, ge=0, le=1)
    effectiveness: float = Field(0.80, ge=0, le=1)


class SavingsResponse(BaseModel):
    fleet_vcpu_hours_year: float
    addressable_fraction: float
    addressable_vcpu_hours: float
    reclamation_rate: float
    vcpu_hour_price: float
    upper_bound_dollars: float
    coverage: float
    effectiveness: float
    realistic_savings_dollars: float


class ModelInfoResponse(BaseModel):
    mode: str
    ensemble_models: list[str]
    feature_count: int
    feature_groups: dict[str, int]
    report_metrics: dict[str, Any]


class SampleJobResponse(BaseModel):
    index: int
    collection_id: int
    scheduling_class: int
    class_name: str
    requested_cpus: float
    actual_peak: float
    ensemble_prediction: float


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    return {"status": "healthy", "mode": agent.mode}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    n_feats = len(agent._feature_names) if agent._feature_names else len(ALL_FEATURES)
    return ModelInfoResponse(
        mode=agent.mode,
        ensemble_models=["LightGBM (quantile q=0.9091)", "CatBoost (quantile q=0.90)"],
        feature_count=n_feats,
        feature_groups={group: len(feats) for group, feats in FEATURE_GROUPS.items()},
        report_metrics={
            "best_asymrmse": 23.64,
            "baseline_asymrmse": 72.11,
            "improvement_pct": 67.2,
            "under_prediction_rate_pct": 6.4,
            "class_asymrmse": CLASS_ASYMRMSE,
            "class_dataset_share": CLASS_DATASET_SHARE,
            "local_model_metrics": {
                k: v for k, v in agent._metadata.items()
                if k in ("lgb_test_asymrmse", "cat_test_asymrmse",
                         "ensemble_test_asymrmse", "under_prediction_rate")
            } if agent._metadata else None,
        },
    )


@app.get("/sample-jobs", response_model=list[SampleJobResponse])
def list_sample_jobs():
    """Return the 100 pre-loaded sample jobs available for analysis."""
    if agent.sample_jobs is None:
        raise HTTPException(status_code=404, detail="No sample jobs loaded. Run train.py first.")
    results = []
    for i, row in agent.sample_jobs.iterrows():
        results.append(SampleJobResponse(
            index=i,
            collection_id=int(row["collection_id"]),
            scheduling_class=int(row["scheduling_class"]),
            class_name=CLASS_NAMES.get(int(row["scheduling_class"]), "Unknown"),
            requested_cpus=float(row["requested_cpus"]),
            actual_peak=float(row["actual_peak"]),
            ensemble_prediction=float(row["ensemble_prediction"]),
        ))
    return results


@app.post("/predict/{job_index}", response_model=PredictionResponse)
def predict_job(job_index: int):
    """
    Predict peak CPU utilization for a sample job by index.

    Uses the trained ensemble on real pre-computed features.
    For Class 0-1: REFUSES to right-size, citing the structural error floor.
    For Class 2-3: returns predicted peak + 10% safety buffer.
    """
    if agent.sample_jobs is None:
        raise HTTPException(status_code=404, detail="No sample jobs loaded.")
    if job_index < 0 or job_index >= len(agent.sample_jobs):
        raise HTTPException(status_code=400, detail=f"Job index must be 0-{len(agent.sample_jobs)-1}")

    result = agent.predict_job(job_index)
    return PredictionResponse(
        decision=result.decision.value,
        scheduling_class=result.scheduling_class,
        class_name=result.class_name,
        predicted_peak_utilization=result.predicted_peak_utilization,
        safety_buffer_pct=result.safety_buffer_pct,
        recommended_cpu_ceiling=result.recommended_cpu_ceiling,
        original_request=result.original_request,
        cpu_freed_pct=result.cpu_freed_pct,
        actual_peak=result.actual_peak,
        reasoning=[ReasoningStepResponse(step=r.step, detail=r.detail) for r in result.reasoning],
        refusal_reason=result.refusal_reason,
        mode=result.mode,
    )


@app.post("/savings", response_model=SavingsResponse)
def estimate_savings(params: SavingsInput):
    result = DeepScalerAgent.estimate_savings(
        fleet_vcpu_hours=params.fleet_vcpu_hours_year,
        addressable_fraction=params.addressable_fraction,
        vcpu_price=params.vcpu_hour_price,
        coverage=params.coverage,
        effectiveness=params.effectiveness,
    )
    return SavingsResponse(**result)


@app.get("/classes")
def scheduling_classes():
    return [
        {
            "class": cls,
            "name": CLASS_NAMES[cls],
            "dataset_share_pct": CLASS_DATASET_SHARE[cls] * 100,
            "asymrmse": CLASS_ASYMRMSE[cls],
            "right_sizeable": cls >= 2,
            "note": (
                "Structural error floor -- burst events after 15-min window"
                if cls == 1 else
                "Unpredictable, preemptible" if cls == 0 else
                "Eligible for right-sizing"
            ),
        }
        for cls in sorted(CLASS_NAMES)
    ]
