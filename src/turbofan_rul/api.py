"""FastAPI service: send an engine's recent sensor readings, get back RUL + risk.

Start it with:  uvicorn turbofan_rul.api:app --reload
Then open http://127.0.0.1:8000/docs to try it in the browser.

Endpoints
- GET  /health                 is the model loaded, is Ollama reachable
- GET  /model/info             model name, metrics, sensors, risk rules
- GET  /sample/{engine_id}     recent readings of an FD001 test engine (handy for demos)
- POST /predict                readings in -> predicted RUL + risk level
- POST /explain                same, plus sensor impacts and a plain-English note
"""

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .config import MODEL_PATH, RISK_HIGH, RISK_MEDIUM, RUL_CAP, SENSOR_COLS, SETTING_COLS
from .data import load_test
from .explain import ollama_available, write_note
from .model import RULModel, risk_level


class Reading(BaseModel):
    """One cycle of data: the cycle number, 3 operating settings, 21 sensors."""

    cycle: int = Field(..., ge=1)
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    s7: float
    s8: float
    s9: float
    s10: float
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s16: float
    s17: float
    s18: float
    s19: float
    s20: float
    s21: float


class EngineReadings(BaseModel):
    engine_id: str = Field("engine-1", description="Any label you like; it's echoed back.")
    readings: list[Reading] = Field(..., min_length=1, description="Oldest first. 30+ cycles gives the best results.")

    @field_validator("readings")
    @classmethod
    def cycles_increase(cls, v: list[Reading]) -> list[Reading]:
        cycles = [r.cycle for r in v]
        if any(b <= a for a, b in zip(cycles, cycles[1:])):
            raise ValueError("readings must be in order of increasing cycle, with no repeats")
        return v

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame([r.model_dump() for r in self.readings])
        return df.assign(unit=0)[["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]]


class Prediction(BaseModel):
    engine_id: str
    predicted_rul: float = Field(..., description="Predicted remaining useful life, in cycles")
    risk_level: str
    cycles_used: int
    model: str


class Factor(BaseModel):
    feature: str
    description: str
    impact_cycles: float = Field(..., description="Cycles of life this sensor's state costs vs a healthy engine")
    current_value: float
    healthy_value: float
    trend_per_cycle: float | None


class Explanation(Prediction):
    factors: list[Factor]
    note: str
    note_source: str


@lru_cache(maxsize=1)
def get_model() -> RULModel:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No model at {MODEL_PATH}. Run `python -m turbofan_rul.train` first.")
    return RULModel.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _test_data() -> pd.DataFrame:
    return load_test()[0]


app = FastAPI(
    title="Turbofan RUL Predictor",
    version=__version__,
    description="Remaining useful life for NASA C-MAPSS FD001 turbofan engines, with explanations.",
)


def _predict(body: EngineReadings) -> tuple[RULModel, pd.DataFrame, float]:
    model = get_model()
    x = model.latest_features(body.to_frame())
    rul = float(model.predict_features(x)[0])
    return model, x, rul


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists(), "ollama_available": ollama_available()}


@app.get("/model/info")
def model_info() -> dict:
    m = get_model()
    return {
        "model": m.name,
        "params": m.params,
        "metrics": {k: round(v, 3) for k, v in m.metrics.items()},
        "sensors_used": m.sensors,
        "n_features": len(m.feature_cols),
        "rul_cap": RUL_CAP,
        "risk_levels": {"high": f"<= {RISK_HIGH} cycles", "medium": f"<= {RISK_MEDIUM} cycles", "low": f"> {RISK_MEDIUM} cycles"},
    }


@app.get("/sample/{engine_id}")
def sample(engine_id: int, last_n: int = 30) -> dict:
    test = _test_data()
    rows = test[test["unit"] == engine_id]
    if rows.empty:
        raise HTTPException(404, f"FD001 test engines are numbered 1-{test['unit'].max()}")
    rows = rows.tail(max(1, last_n)).drop(columns="unit")
    return {"engine_id": f"FD001-test-{engine_id}", "readings": rows.to_dict(orient="records")}


@app.post("/predict", response_model=Prediction)
def predict(body: EngineReadings) -> Prediction:
    model, _, rul = _predict(body)
    return Prediction(
        engine_id=body.engine_id,
        predicted_rul=round(rul, 1),
        risk_level=risk_level(rul),
        cycles_used=len(body.readings),
        model=model.name,
    )


@app.post("/explain", response_model=Explanation)
def explain(body: EngineReadings, use_llm: bool = True, top_n: int = 5) -> Explanation:
    model, x, rul = _predict(body)
    risk = risk_level(rul)
    factors = model.explain_features(x)
    note, source = write_note(body.engine_id, rul, risk, factors, use_llm=use_llm)
    return Explanation(
        engine_id=body.engine_id,
        predicted_rul=round(rul, 1),
        risk_level=risk,
        cycles_used=len(body.readings),
        model=model.name,
        factors=[Factor(**f) for f in factors[: max(1, top_n)]],
        note=note,
        note_source=source,
    )
