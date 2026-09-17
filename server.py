"""
SupportHR Remote Classifier Microservice
---------------------------------------
Standalone FastAPI service for CV Industry Classification (24 classes).
Deployable on Google Colab, Kaggle Notebooks, Docker, or Cloud VPS.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("classifier_service")

DEFAULT_LABELS = [
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS", "AUTOMOBILE",
    "AVIATION", "BANKING", "BPO", "BUSINESS-DEVELOPMENT", "CHEF", "CONSTRUCTION",
    "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA", "ENGINEERING", "FINANCE", "FITNESS",
    "HEALTHCARE", "HR", "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER"
]

app = FastAPI(
    title="SupportHR Classifier Microservice",
    version="1.0.0",
    description="Dedicated microservice for CV Industry Classification",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Any | None = None
_model_source: str = os.getenv("MODEL_SOURCE", "remote://classifier-service")
_model_error: str | None = None


class ClassifyRequest(BaseModel):
    cv_text: str = Field(..., description="Extracted CV text")
    top_k: int = Field(default=3, ge=1, le=24, description="Top-k predictions")


class TopPrediction(BaseModel):
    label: str
    score: float


class ClassifyResponse(BaseModel):
    predicted_label: str
    confidence: float
    top_predictions: list[TopPrediction]
    model_source: str


class ClassifierStatusResponse(BaseModel):
    ready: bool
    model_source: str
    label_count: int
    labels: list[str]
    error: str | None = None


def clean_text(text: str) -> str:
    normalized = str(text).lower()
    normalized = re.sub(r"[^\w\s]+", " ", normalized, flags=re.UNICODE)
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    if total <= 0:
        return [0.0 for _ in values]
    return [v / total for v in exps]


def init_model(model_path: str | Path | None = None) -> None:
    global _model, _model_error

    import joblib

    resolved_path = None
    if model_path:
        candidate = Path(model_path)
        if candidate.is_file():
            resolved_path = candidate

    if not resolved_path:
        default_candidates = [
            Path("text_classifier_model.pkl"),
            Path(__file__).resolve().parent / "text_classifier_model.pkl",
            Path(__file__).resolve().parent.parent / "cv-match-api" / "api_server" / "app" / "models" / "text_classifier_model.pkl",
            Path("/kaggle/working/text_classifier_model.pkl"),
            Path("/content/text_classifier_model.pkl"),
        ]
        for candidate in default_candidates:
            if candidate.is_file():
                resolved_path = candidate
                break

    if resolved_path:
        try:
            logger.info("Loading classifier model from %s", resolved_path)
            _model = joblib.load(resolved_path)
            _model_error = None
            logger.info("Model loaded successfully with classes: %s", getattr(_model, "classes_", []))
            return
        except Exception as err:
            logger.warning("Failed to load model from %s: %s", resolved_path, err)
            _model_error = str(err)

    logger.info("Building in-memory pipeline for 24 SupportHR industry categories...")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC

        seed_data = [
            ("accounting finance ledger tax audit balance sheet accountant bookkeeping", "ACCOUNTANT"),
            ("legal advocate court lawyer litigation counsel compliance contract", "ADVOCATE"),
            ("agriculture crop farming agronomy harvesting soil seeds livestock", "AGRICULTURE"),
            ("apparel fashion textile garment clothing designer merchandising pattern", "APPAREL"),
            ("fine arts painting sculpture gallery illustrator artist visual exhibition", "ARTS"),
            ("automobile vehicle automotive mechanic engine car repair transport diagnostic", "AUTOMOBILE"),
            ("aviation pilot flight aircraft airline aeronautical aerospace avionics", "AVIATION"),
            ("banking loan credit deposit investment branch banking banker mortgage", "BANKING"),
            ("bpo customer support call center outsourcing telemarketing agent technical", "BPO"),
            ("business development partnership growth strategy b2b sales lead enterprise", "BUSINESS-DEVELOPMENT"),
            ("chef culinary cooking cuisine kitchen restaurant pastry food recipe", "CHEF"),
            ("civil engineering construction architect building site contractor structural", "CONSTRUCTION"),
            ("consulting advisory management consultant strategy operational roadmap", "CONSULTANT"),
            ("graphic designer ui ux figma product layout typography visual wireframe", "DESIGNER"),
            ("digital media social content video creator broadcast marketing journalism", "DIGITAL-MEDIA"),
            ("electrical mechanical engineering firmware hardware maintenance cad plc", "ENGINEERING"),
            ("financial analyst equity wealth investment banking portfolio risk treasury", "FINANCE"),
            ("fitness trainer gym coach exercise workout health sports athletics", "FITNESS"),
            ("doctor nurse medical hospital clinical patient healthcare therapy pharmacy", "HEALTHCARE"),
            ("human resources talent acquisition recruitment payroll hr screening onboarding", "HR"),
            ("python fastapi react typescript software engineer developer backend frontend cloud devops", "INFORMATION-TECHNOLOGY"),
            ("public relations communications press release media spokesperson crisis branding", "PUBLIC-RELATIONS"),
            ("sales executive quota pipeline negotiation account manager cold calling revenue", "SALES"),
            ("teacher tutor education curriculum classroom school professor pedagogy lecturing", "TEACHER"),
        ]
        X = [item[0] for item in seed_data]
        y = [item[1] for item in seed_data]

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LinearSVC(random_state=42, dual="auto")),
        ])
        pipeline.fit(X, y)
        _model = pipeline
        _model_error = None
        logger.info("In-memory baseline model initialized successfully.")
    except Exception as err:
        logger.error("Failed to initialize baseline model: %s", err)
        _model_error = str(err)


@app.on_event("startup")
def startup_event():
    init_model()


@app.get("/health")
def health():
    return {"status": "ok", "ready": _model is not None}


@app.get("/api/cv/classifier-status", response_model=ClassifierStatusResponse)
@app.get("/api/classifier-status", response_model=ClassifierStatusResponse)
def classifier_status():
    if _model is None:
        return ClassifierStatusResponse(
            ready=False,
            model_source=_model_source,
            label_count=0,
            labels=[],
            error=_model_error or "Model not loaded",
        )
    classes = [str(c) for c in getattr(_model, "classes_", DEFAULT_LABELS)]
    return ClassifierStatusResponse(
        ready=True,
        model_source=_model_source,
        label_count=len(classes),
        labels=classes,
        error=None,
    )


@app.post("/api/cv/classify-industry", response_model=ClassifyResponse)
@app.post("/api/classify-industry", response_model=ClassifyResponse)
def classify_industry(payload: ClassifyRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Classifier model is not ready.")

    cleaned = clean_text(payload.cv_text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="CV text is empty after cleaning.")

    classes = [str(c) for c in getattr(_model, "classes_", DEFAULT_LABELS)]

    if hasattr(_model, "predict_proba"):
        probabilities = _model.predict_proba([cleaned])[0]
        scored = [
            {"label": label, "score": round(float(score), 4)}
            for label, score in zip(classes, probabilities, strict=False)
        ]
    elif hasattr(_model, "decision_function"):
        raw_scores = _model.decision_function([cleaned])
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        if isinstance(raw_scores, list) and raw_scores and isinstance(raw_scores[0], list):
            raw_scores = raw_scores[0]
        if not isinstance(raw_scores, list):
            raw_scores = [float(raw_scores)]
        probabilities = _softmax([float(v) for v in raw_scores])
        scored = [
            {"label": label, "score": round(float(score), 4)}
            for label, score in zip(classes, probabilities, strict=False)
        ]
    else:
        pred = str(_model.predict([cleaned])[0])
        scored = [{"label": pred, "score": 1.0}]

    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)[:payload.top_k]
    top_label = ranked[0]["label"] if ranked else "UNKNOWN"
    top_score = ranked[0]["score"] if ranked else 0.0

    return ClassifyResponse(
        predicted_label=top_label,
        confidence=top_score,
        top_predictions=[TopPrediction(label=item["label"], score=item["score"]) for item in ranked],
        model_source=_model_source,
    )


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="SupportHR Classifier Microservice")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--model-path", help="Path to model file")
    parser.add_argument("--model-source", default="remote://classifier-service", help="Model source identifier")
    args = parser.parse_args()

    global _model_source
    _model_source = args.model_source
    init_model(args.model_path)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
