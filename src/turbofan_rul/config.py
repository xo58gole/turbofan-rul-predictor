"""Settings shared across the project: paths, column names and a few constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "model.joblib"

SUBSET = "FD001"

# The raw files have no header, so these are the 26 columns in order.
ID_COLS = ["unit", "cycle"]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
RAW_COLS = ID_COLS + SETTING_COLS + SENSOR_COLS

# What each sensor measures (from the C-MAPSS documentation, Saxena et al. 2008).
# We use these names in the plain-English explanations.
SENSOR_NAMES = {
    "s1": "fan inlet temperature",
    "s2": "LPC outlet temperature",
    "s3": "HPC outlet temperature",
    "s4": "LPT outlet temperature",
    "s5": "fan inlet pressure",
    "s6": "bypass-duct pressure",
    "s7": "HPC outlet pressure",
    "s8": "physical fan speed",
    "s9": "physical core speed",
    "s10": "engine pressure ratio",
    "s11": "HPC outlet static pressure",
    "s12": "fuel flow to static pressure ratio",
    "s13": "corrected fan speed",
    "s14": "corrected core speed",
    "s15": "bypass ratio",
    "s16": "burner fuel-air ratio",
    "s17": "bleed enthalpy",
    "s18": "demanded fan speed",
    "s19": "demanded corrected fan speed",
    "s20": "HPT coolant bleed",
    "s21": "LPT coolant bleed",
}

# RUL labels get capped here. Early in an engine's life there's no visible
# wear, so asking the model to tell "300 cycles left" from "200 cycles left"
# is just noise. 125 is the usual choice for FD001.
RUL_CAP = 125

# Rolling windows (in cycles) for the time-series features.
SHORT_WINDOW = 10
LONG_WINDOW = 30

# Risk levels on predicted RUL (cycles). <= HIGH is high risk, <= MEDIUM is medium.
RISK_HIGH = 30
RISK_MEDIUM = 75

RANDOM_STATE = 42
