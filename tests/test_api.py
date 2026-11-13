"""Tests for the FastAPI service and the explanation note."""

import pytest
from fastapi.testclient import TestClient

from turbofan_rul import api, explain


@pytest.fixture()
def client(small_model, test_data, monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda: small_model)
    # Make sure no test ever waits on a real Ollama.
    monkeypatch.setattr(explain, "llm_note", lambda prompt: None)
    monkeypatch.setattr(api, "ollama_available", lambda: False)
    return TestClient(api.app)


@pytest.fixture()
def engine_payload(client):
    return client.get("/sample/1", params={"last_n": 30}).json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client, small_model):
    info = client.get("/model/info").json()
    assert info["model"] == "test_model"
    assert info["sensors_used"] == small_model.sensors


def test_sample_returns_readings(engine_payload):
    assert engine_payload["engine_id"] == "FD001-test-1"
    assert len(engine_payload["readings"]) == 30


def test_sample_unknown_engine(client):
    assert client.get("/sample/999").status_code == 404


def test_predict(client, engine_payload, small_model, test_data):
    r = client.post("/predict", json=engine_payload)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["predicted_rul"] <= 125
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["cycles_used"] == 30

    # The API must agree with the model used directly on the same readings.
    test, _ = test_data
    direct = small_model.predict(test[test["unit"] == 1].tail(30))[0]
    assert body["predicted_rul"] == pytest.approx(direct, abs=0.05)


def test_predict_single_reading_works(client, engine_payload):
    payload = {"engine_id": "x", "readings": engine_payload["readings"][-1:]}
    assert client.post("/predict", json=payload).status_code == 200


def test_predict_rejects_empty(client):
    assert client.post("/predict", json={"engine_id": "x", "readings": []}).status_code == 422


def test_predict_rejects_missing_sensor(client, engine_payload):
    reading = dict(engine_payload["readings"][0])
    del reading["s7"]
    assert client.post("/predict", json={"readings": [reading]}).status_code == 422


def test_predict_rejects_out_of_order_cycles(client, engine_payload):
    payload = {"readings": list(reversed(engine_payload["readings"]))}
    assert client.post("/predict", json=payload).status_code == 422


def test_explain_falls_back_to_template(client, engine_payload):
    r = client.post("/explain", json=engine_payload, params={"top_n": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["note_source"] == "template"
    assert len(body["factors"]) == 3
    assert f"{body['predicted_rul']:.0f} cycles" in body["note"]
    impacts = [abs(f["impact_cycles"]) for f in body["factors"]]
    assert impacts == sorted(impacts, reverse=True)


def test_explain_uses_llm_when_available(client, engine_payload, monkeypatch):
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return "LLM note."

    monkeypatch.setattr(explain, "llm_note", fake_llm)
    body = client.post("/explain", json=engine_payload).json()
    assert body["note"] == "LLM note."
    assert body["note_source"].startswith("ollama:")
    # The prompt carries the real numbers, so the LLM has nothing to make up.
    assert str(round(body["predicted_rul"])) in seen["prompt"]


def test_explain_can_skip_llm(client, engine_payload, monkeypatch):
    monkeypatch.setattr(explain, "llm_note", lambda p: pytest.fail("LLM should not be called"))
    body = client.post("/explain", json=engine_payload, params={"use_llm": False}).json()
    assert body["note_source"] == "template"


def test_template_note_mentions_action():
    note = explain.template_note(
        "E1", 20.0, "high",
        [{"description": "HPC outlet temperature", "impact_cycles": 12.0}],
    )
    assert "high risk" in note and "HPC outlet temperature" in note and "inspection" in note
