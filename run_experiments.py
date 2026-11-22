"""
Run every experiment once and save the results, so the dashboard loads instantly.

1. Hold-out test (train 2015-2022, test 2023-2024) for both tasks, all models.
2. Walk-forward validation by year (train up to year Y-1, test on year Y,
   for Y = 2022..2024) to get mean +/- std of each metric.
3. SARIMA vs RF vs SARIMA+RF hybrid for monthly rainfall in every city.

Output: results/results.pkl
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import ml

OUT = Path(__file__).parent / "results"


def main():
    OUT.mkdir(exist_ok=True)
    daily = ml.load_daily()
    f = ml.build_features(daily)
    results = {"n_rows": len(f), "tasks": {}}

    for task in ml.TASKS:
        print(f"\n=== Task: {task} ===")
        X, y = ml.make_xy(f, task)
        X_tr, X_te, y_tr, y_te = ml.time_split(f, X, y)
        models = ml.get_models(task, **ml.DEFAULT_PARAMS)

        holdout, preds, probas, importances = [], {}, {}, {}
        for name, model in models.items():
            print(f"  hold-out  {name}")
            res, pred, proba = ml.evaluate(model, X_tr, X_te, y_tr, y_te, task)
            holdout.append({"model": name, "family": ml.FAMILY[name], **res})
            preds[name] = pred
            if proba is not None:
                probas[name] = proba
            imp = ml.feature_importance(model, X.columns)
            if imp is not None:
                importances[name] = imp

        # walk-forward yearly validation
        cv_rows = []
        for year in range(2022, 2025):
            tr = f["date"] < f"{year}-01-01"
            te = (f["date"] >= f"{year}-01-01") & (f["date"] < f"{year + 1}-01-01")
            for name, model in ml.get_models(task, **ml.DEFAULT_PARAMS).items():
                print(f"  fold {year}  {name}")
                res, _, _ = ml.evaluate(model, X[tr], X[te], y[tr], y[te], task)
                cv_rows.append({"model": name, "fold": year, **res})

        results["tasks"][task] = {
            "holdout": pd.DataFrame(holdout),
            "cv": pd.DataFrame(cv_rows),
            "y_test": y_te.to_numpy(),
            "test_meta": f.loc[X_te.index, ["date", "city", "season", "climate_zone"]].reset_index(drop=True),
            "preds": preds,
            "probas": probas,
            "importances": importances,
            "n_train": len(X_tr),
            "n_test": len(X_te),
        }

    print("\n=== Time series hybrids ===")
    ts = {}
    for city in daily["city"].unique():
        print(f"  {city}")
        s = ml.monthly_rainfall(daily, city)
        actual, preds, metrics = ml.timeseries_hybrid(s)
        ts[city] = {"series": s, "actual": actual, "preds": preds, "metrics": metrics}
    results["timeseries"] = ts

    with open(OUT / "results.pkl", "wb") as fh:
        pickle.dump(results, fh)
    print("\nSaved results/results.pkl")

    for task, r in results["tasks"].items():
        print(f"\n{task} hold-out:")
        print(r["holdout"].drop(columns=["family"]).round(4).to_string(index=False))
    summary = pd.concat([v["metrics"].assign(city=k) for k, v in ts.items()])
    print("\nTime series mean RMSE by model:")
    print(summary.groupby("model")[["rmse", "mae", "r2"]].mean().round(2))


if __name__ == "__main__":
    np.random.seed(ml.RANDOM_STATE)
    main()
