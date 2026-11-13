"""Train, compare and save the RUL models, logging every run to MLflow.

Run it with:  python -m turbofan_rul.train

What happens:
1. Load FD001, build features, cap RUL labels at 125.
2. 5-fold cross-validation grouped by engine, so no engine is ever in both
   the training and the validation fold.
3. Compare a mean-only floor, a linear baseline and a small grid of
   gradient boosting models. Pick the winner on cross-validation RMSE
   (never on the test set).
4. Refit the winner on all training engines, score it once on NASA's
   official 100 test engines, and save the model + results.
"""

import json
import os
import time
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import LONG_WINDOW, MODEL_PATH, RANDOM_STATE, REPORT_DIR, ROOT, RUL_CAP, SUBSET
from .data import load_test, load_train
from .features import build_features, feature_columns, find_constant_sensors, useful_sensors
from .model import RULModel, nasa_score, risk_level, rmse

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

N_FOLDS = 5
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
EXPERIMENT = f"turbofan-rul-{SUBSET}"


def candidates() -> list[tuple[str, str, object, dict]]:
    """(run name, family, estimator, params) for everything we compare."""
    out = [
        ("mean_only", "floor", DummyRegressor(strategy="mean"), {}),
        # Baseline: linear regression on the current sensor values only, no history.
        ("linear_current_sensors", "baseline", make_pipeline(StandardScaler(), LinearRegression()), {"features": "current"}),
    ]
    for lr in (0.05, 0.1):
        for leaves in (15, 31):
            for min_leaf in (20, 50):
                params = {
                    "learning_rate": lr,
                    "max_leaf_nodes": leaves,
                    "min_samples_leaf": min_leaf,
                    "max_iter": 300,
                    "l2_regularization": 1.0,
                }
                est = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
                out.append((f"gbm_lr{lr}_leaves{leaves}_minleaf{min_leaf}", "gradient_boosting", est, params))
    return out


def columns_for(family: str, params: dict, sensors: list[str]) -> list[str]:
    if params.get("features") == "current":
        return list(sensors)
    return feature_columns(sensors)


def cross_validate(est, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    fold_rmse, fold_score = [], []
    for tr_idx, va_idx in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = np.clip(est.predict(X.iloc[va_idx]), 0, RUL_CAP)
        fold_rmse.append(rmse(y.iloc[va_idx], pred))
        # In CV every row is a valid "now", so we average the NASA score per row
        # to keep folds comparable. (The official score sums one row per test engine.)
        fold_score.append(nasa_score(y.iloc[va_idx], pred) / len(va_idx))
    return {
        "cv_rmse": float(np.mean(fold_rmse)),
        "cv_rmse_std": float(np.std(fold_rmse)),
        "cv_nasa_score_per_row": float(np.mean(fold_score)),
    }


def main() -> dict:
    t0 = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_train()
    test, truth = load_test()
    sensors = useful_sensors(train)
    dropped = find_constant_sensors(train)

    feats = build_features(train, sensors)
    y = train.loc[feats.index, "rul"]
    groups = train.loc[feats.index, "unit"]

    test_feats = build_features(test, sensors)
    last_idx = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1).index
    test_last = test_feats.loc[last_idx]

    # "Healthy" reference values for explanations: early-life rows with full
    # rolling windows, where the (uncapped) RUL is still above the cap.
    healthy_mask = (y >= RUL_CAP) & (train.loc[feats.index, "cycle"] >= LONG_WINDOW)
    healthy_reference = feats.loc[healthy_mask].median().to_dict()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    results = []
    for run_name, family, est, params in candidates():
        cols = columns_for(family, params, sensors)
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"family": family, "subset": SUBSET})
            mlflow.log_params({**params, "n_features": len(cols), "rul_cap": RUL_CAP, "cv_folds": N_FOLDS})
            cv = cross_validate(est, feats[cols], y, groups)
            # Also score every candidate on the test set, for the record. It's
            # logged, but it plays no part in choosing the winner.
            est.fit(feats[cols], y)
            pred = np.clip(est.predict(test_last[cols]), 0, RUL_CAP)
            test_metrics = {"test_rmse": rmse(truth, pred), "test_nasa_score": nasa_score(truth, pred)}
            mlflow.log_metrics({**cv, **test_metrics})
            results.append({"run": run_name, "family": family, "params": params, "cols": cols, **cv, **test_metrics, "pred": pred})
            print(f"{run_name:40s} cv_rmse={cv['cv_rmse']:.2f}  test_rmse={test_metrics['test_rmse']:.2f}  nasa={test_metrics['test_nasa_score']:.0f}")

    gbms = [r for r in results if r["family"] == "gradient_boosting"]
    best = min(gbms, key=lambda r: r["cv_rmse"])
    best_est = next(e for n, _, e, _ in candidates() if n == best["run"])
    best_est.fit(feats[best["cols"]], y)

    model = RULModel(
        estimator=best_est,
        sensors=sensors,
        healthy_reference={k: float(v) for k, v in healthy_reference.items()},
        name=best["run"],
        params=best["params"],
    )
    test_pred = model.predict_features(test_last[model.feature_cols])
    assert np.allclose(test_pred, best["pred"]), "refit model should reproduce the logged test predictions"

    # Explanations for every test engine (used for the global importance chart).
    attributions = []
    for i, idx in enumerate(last_idx):
        for f in model.explain_features(test_last.loc[[idx], model.feature_cols]):
            attributions.append({"unit": int(test.loc[idx, "unit"]), **f})
    attr = pd.DataFrame(attributions)

    errors = test_pred - truth
    preds = pd.DataFrame(
        {
            "unit": test.loc[last_idx, "unit"].to_numpy(),
            "cycles_observed": test.loc[last_idx, "cycle"].to_numpy(),
            "true_rul": truth,
            "predicted_rul": np.round(test_pred, 2),
            "baseline_rul": np.round(next(r for r in results if r["run"] == "linear_current_sensors")["pred"], 2),
            "error": np.round(errors, 2),
            "risk_predicted": [risk_level(p) for p in test_pred],
            "risk_true": [risk_level(t) for t in truth],
        }
    )
    preds.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    attr.to_csv(REPORT_DIR / "test_attributions.csv", index=False)

    model.metrics = {
        "test_rmse": rmse(truth, test_pred),
        "test_nasa_score": nasa_score(truth, test_pred),
        "cv_rmse": best["cv_rmse"],
    }

    with mlflow.start_run(run_name=f"final_{best['run']}"):
        mlflow.set_tags({"family": "gradient_boosting", "subset": SUBSET, "final": "true"})
        mlflow.log_params({**best["params"], "n_features": len(model.feature_cols), "selected_by": "cv_rmse"})
        mlflow.log_metrics({**model.metrics, "cv_rmse_std": best["cv_rmse_std"]})
        model.save(MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(REPORT_DIR / "test_predictions.csv"))

    summary = {
        "subset": SUBSET,
        "n_train_engines": int(train["unit"].nunique()),
        "n_train_rows": int(len(train)),
        "n_test_engines": int(test["unit"].nunique()),
        "n_test_rows": int(len(test)),
        "rul_cap": RUL_CAP,
        "sensors_used": sensors,
        "sensors_dropped": dropped,
        "n_features": len(model.feature_cols),
        "selected_model": best["run"],
        "selected_params": best["params"],
        "models": [{k: v for k, v in r.items() if k not in ("pred", "cols")} for r in results],
        "final": model.metrics,
        "train_seconds": round(time.time() - t0, 1),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSelected {best['run']} on CV RMSE. Test RMSE {model.metrics['test_rmse']:.2f}, "
          f"NASA score {model.metrics['test_nasa_score']:.0f}. Saved to {MODEL_PATH}")
    return summary


if __name__ == "__main__":
    main()
