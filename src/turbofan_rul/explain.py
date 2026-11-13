"""Turn a prediction + its sensor impacts into a short note for a maintenance manager.

If Ollama is running locally we ask a small open model to write the note.
If it isn't (or it's slow, or it errors), we fall back to a fixed template,
so the API always returns something useful and never needs a paid key.
"""

import json
import os
import urllib.error
import urllib.request

from .config import RISK_HIGH, RISK_MEDIUM

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "60"))

def recommended_action(risk: str, rul: float) -> str:
    if risk == "high":
        # Leave a safety margin: inspect by roughly half the predicted life.
        return (f"Schedule an inspection within about {max(1, round(rul / 2))} cycles "
                "and line up parts for a possible engine removal.")
    if risk == "medium":
        return "Add the engine to the watch list and plan an inspection at the next scheduled maintenance slot."
    return "No action needed now. Keep monitoring as normal."


def ollama_available(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _top_factors(factors: list[dict], n: int = 3) -> list[dict]:
    """The sensors pulling predicted life down the most (positive impact = life lost)."""
    return [f for f in factors if f["impact_cycles"] > 0][:n]


def template_note(engine_id: str, rul: float, risk: str, factors: list[dict]) -> str:
    top = _top_factors(factors)
    lines = [f"Engine {engine_id}: about {rul:.0f} cycles of useful life left ({risk} risk)."]
    if top:
        parts = [f"{f['description']} (-{f['impact_cycles']:.0f} cycles)" for f in top]
        lines.append("Main drivers compared with a healthy engine: " + ", ".join(parts) + ".")
    else:
        lines.append("Sensor readings look close to a healthy engine.")
    lines.append(f"Recommended action: {recommended_action(risk, rul)}")
    return " ".join(lines)


def build_prompt(engine_id: str, rul: float, risk: str, factors: list[dict]) -> str:
    facts = {
        "engine": engine_id,
        "predicted_remaining_cycles": round(rul),
        "risk_level": risk,
        "risk_rules": f"high <= {RISK_HIGH} cycles, medium <= {RISK_MEDIUM} cycles, low above",
        "top_factors": [
            {
                "what": f["description"],
                "cycles_of_life_it_costs": f["impact_cycles"],
                "current_reading": f["current_value"],
                "healthy_engine_reading": f["healthy_value"],
            }
            for f in _top_factors(factors, 4)
        ],
        "suggested_action": recommended_action(risk, rul),
    }
    return (
        "You write short maintenance notes for an aircraft engine maintenance manager.\n"
        "Use only the facts in the JSON below. Don't invent numbers, sensors or causes.\n"
        "Write 3-4 plain sentences, no lists, no headings: the remaining life and risk level, "
        "the two or three sensors driving it and what they suggest, then the recommended action.\n\n"
        f"Facts:\n{json.dumps(facts, indent=2)}\n\nNote:"
    )


def llm_note(prompt: str) -> str | None:
    body = json.dumps(
        {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as r:
            text = json.loads(r.read().decode()).get("response", "").strip()
            return text or None
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def write_note(engine_id: str, rul: float, risk: str, factors: list[dict], use_llm: bool = True) -> tuple[str, str]:
    """Returns (note, source) where source is 'ollama:<model>' or 'template'."""
    if use_llm:
        note = llm_note(build_prompt(engine_id, rul, risk, factors))
        if note:
            return note, f"ollama:{OLLAMA_MODEL}"
    return template_note(engine_id, rul, risk, factors), "template"
