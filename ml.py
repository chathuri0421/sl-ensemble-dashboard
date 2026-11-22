"""
Core ML code: feature engineering, model zoo (bagging / boosting / stacking /
hybrid), evaluation, and the ARIMA + RF time-series hybrid.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sl_weather_daily.csv"
RANDOM_STATE = 42

TASKS = {
    "rain": {
        "label": "Will it rain tomorrow? (classification, > 1 mm)",
        "type": "classification",
    },
    "temp": {
        "label": "Tomorrow's max temperature in °C (regression)",
        "type": "regression",
    },
}

# Model families, used for colouring charts and grouping in the report
FAMILY = {
    "Decision Tree (baseline)": "Baseline",
    "Linear / Logistic (baseline)": "Baseline",
    "Bagging (trees)": "Bagging",
    "Random Forest": "Bagging",
    "AdaBoost": "Boosting",
    "XGBoost": "Boosting",
    "LightGBM": "Boosting",
    "CatBoost": "Boosting",
    "Stacking (RF+XGB+LGBM)": "Stacking",
    "Hybrid RF + ANN": "Hybrid",
}


# --------------------------------------------------------------------------
# Data + features
# --------------------------------------------------------------------------
def monsoon_season(month):
    """Sri Lanka's four climate seasons (Dept. of Meteorology)."""
    if month in (12, 1, 2):
        return "NE Monsoon"
    if month in (3, 4):
        return "1st Inter-monsoon"
    if month in (5, 6, 7, 8, 9):
        return "SW Monsoon"
    return "2nd Inter-monsoon"  # Oct, Nov


def load_daily():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(monsoon_season)
    return df


def build_features(df):
    """One row per city-day. Features use only today and earlier; targets are tomorrow."""
    out = []
    for _, g in df.groupby("city", sort=False):
        g = g.copy()
        p = g["precipitation_sum"]
        g["rain_lag1"] = p.shift(1)
        g["rain_lag2"] = p.shift(2)
        g["rain_3day"] = p.rolling(3).sum()
        g["rain_7day"] = p.rolling(7).sum()
        g["rain_30day"] = p.rolling(30).sum()
        g["tmax_lag1"] = g["temperature_2m_max"].shift(1)
        g["pressure_change"] = g["surface_pressure_mean"].diff()
        g["humidity_change"] = g["relative_humidity_2m_mean"].diff()
        # targets (tomorrow)
        g["target_rain"] = (p.shift(-1) > 1.0).astype(float)
        g["target_temp"] = g["temperature_2m_max"].shift(-1)
        g.loc[p.shift(-1).isna(), "target_rain"] = np.nan
        out.append(g)
    f = pd.concat(out).dropna().reset_index(drop=True)
    doy = f["date"].dt.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["wind_dir_sin"] = np.sin(np.deg2rad(f["wind_direction_10m_dominant"]))
    f["wind_dir_cos"] = np.cos(np.deg2rad(f["wind_direction_10m_dominant"]))
    return f


NUMERIC_FEATURES = [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "precipitation_hours", "relative_humidity_2m_mean", "surface_pressure_mean",
    "cloud_cover_mean", "wind_speed_10m_max", "shortwave_radiation_sum",
    "rain_lag1", "rain_lag2", "rain_3day", "rain_7day", "rain_30day",
    "tmax_lag1", "pressure_change", "humidity_change",
    "doy_sin", "doy_cos", "wind_dir_sin", "wind_dir_cos", "month",
]
CATEGORICAL_FEATURES = ["city", "climate_zone", "season"]


def make_xy(f, task):
    X = pd.get_dummies(f[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
                       columns=CATEGORICAL_FEATURES, dtype=float)
    y = f["target_rain"].astype(int) if task == "rain" else f["target_temp"]
    return X, y


def time_split(f, X, y, test_from="2023-01-01"):
    """Chronological split: train on 2015-2022, test on 2023-2024 (no future leakage)."""
    is_test = f["date"] >= test_from
    return X[~is_test], X[is_test], y[~is_test], y[is_test]


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------
DEFAULT_PARAMS = {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05}


def get_models(task, n_estimators=300, max_depth=8, learning_rate=0.05):
    rs, n, d, lr = RANDOM_STATE, n_estimators, max_depth, learning_rate
    if TASKS[task]["type"] == "classification":
        rf = RandomForestClassifier(n_estimators=n, max_depth=d + 6, n_jobs=-1, random_state=rs)
        xgb = XGBClassifier(n_estimators=n, max_depth=d // 2 + 2, learning_rate=lr,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                            random_state=rs, eval_metric="logloss", verbosity=0)
        lgbm = LGBMClassifier(n_estimators=n, num_leaves=2 ** (d // 2 + 2) - 1, learning_rate=lr,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=rs, verbose=-1, n_jobs=-1)
        ann = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
            max_iter=300, random_state=rs))
        return {
            "Decision Tree (baseline)": DecisionTreeClassifier(max_depth=d, random_state=rs),
            "Linear / Logistic (baseline)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "Bagging (trees)": BaggingClassifier(DecisionTreeClassifier(max_depth=d + 6),
                                                 n_estimators=min(n, 100), n_jobs=-1, random_state=rs),
            "Random Forest": rf,
            "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                                           n_estimators=n, learning_rate=lr * 10, random_state=rs),
            "XGBoost": xgb,
            "LightGBM": lgbm,
            "CatBoost": CatBoostClassifier(iterations=n, depth=min(d, 10), learning_rate=lr,
                                           random_seed=rs, verbose=0, thread_count=-1, allow_writing_files=False),
            "Stacking (RF+XGB+LGBM)": StackingClassifier(
                estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
                final_estimator=LogisticRegression(max_iter=2000), cv=5, n_jobs=1),
            "Hybrid RF + ANN": VotingClassifier([("rf", rf), ("ann", ann)], voting="soft"),
        }
    rf = RandomForestRegressor(n_estimators=n, max_depth=d + 6, n_jobs=-1, random_state=rs)
    xgb = XGBRegressor(n_estimators=n, max_depth=d // 2 + 2, learning_rate=lr,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=rs, verbosity=0)
    lgbm = LGBMRegressor(n_estimators=n, num_leaves=2 ** (d // 2 + 2) - 1, learning_rate=lr,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         random_state=rs, verbose=-1, n_jobs=-1)
    ann = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
        max_iter=300, random_state=rs))
    return {
        "Decision Tree (baseline)": DecisionTreeRegressor(max_depth=d, random_state=rs),
        "Linear / Logistic (baseline)": make_pipeline(StandardScaler(), Ridge()),
        "Bagging (trees)": BaggingRegressor(DecisionTreeRegressor(max_depth=d + 6),
                                            n_estimators=min(n, 100), n_jobs=-1, random_state=rs),
        "Random Forest": rf,
        "AdaBoost": AdaBoostRegressor(DecisionTreeRegressor(max_depth=3),
                                      n_estimators=n, learning_rate=lr * 10, random_state=rs),
        "XGBoost": xgb,
        "LightGBM": lgbm,
        "CatBoost": CatBoostRegressor(iterations=n, depth=min(d, 10), learning_rate=lr,
                                      random_seed=rs, verbose=0, thread_count=-1, allow_writing_files=False),
        "Stacking (RF+XGB+LGBM)": StackingRegressor(
            estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
            final_estimator=Ridge(), cv=5, n_jobs=1),
        "Hybrid RF + ANN": VotingRegressor([("rf", rf), ("ann", ann)]),
    }


def evaluate(model, X_train, X_test, y_train, y_test, task):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X_test)
    pred_s = time.perf_counter() - t0
    res = {"train_time_s": fit_s, "predict_time_s": pred_s}
    if TASKS[task]["type"] == "classification":
        proba = model.predict_proba(X_test)[:, 1]
        res.update(
            accuracy=accuracy_score(y_test, pred),
            precision=precision_score(y_test, pred),
            recall=recall_score(y_test, pred),
            f1=f1_score(y_test, pred),
            roc_auc=roc_auc_score(y_test, proba),
        )
        return res, pred, proba
    res.update(
        rmse=float(np.sqrt(mean_squared_error(y_test, pred))),
        mae=mean_absolute_error(y_test, pred),
        r2=r2_score(y_test, pred),
    )
    return res, pred, None


def feature_importance(model, columns):
    """Native importances for tree models (normalised to sum to 1)."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        return pd.Series(imp / imp.sum(), index=columns).sort_values(ascending=False)
    return None


# --------------------------------------------------------------------------
# Time series: ARIMA vs RF vs ARIMA + RF (residual) hybrid
# --------------------------------------------------------------------------
def monthly_rainfall(df, city):
    s = (df[df["city"] == city].set_index("date")["precipitation_sum"]
         .resample("MS").sum())
    s.name = "rainfall_mm"
    return s


def _lag_frame(s, lags=(1, 2, 3, 12)):
    X = pd.DataFrame({f"lag{l}": s.shift(l) for l in lags}, index=s.index)
    X["month_sin"] = np.sin(2 * np.pi * s.index.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * s.index.month / 12)
    return X


def timeseries_hybrid(series, test_months=24):
    """Rolling one-step-ahead forecasts for the last `test_months` months."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    order, sorder = (1, 0, 1), (1, 1, 1, 12)
    train_end = len(series) - test_months

    # --- SARIMA fitted once on train, then updated with each observed month
    sarima = SARIMAX(series.iloc[:train_end], order=order, seasonal_order=sorder,
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    in_sample = sarima.fittedvalues
    resid = series.iloc[:train_end] - in_sample
    sarima_fc = []
    res_obj = sarima
    for i in range(train_end, len(series)):
        sarima_fc.append(float(res_obj.forecast(1).iloc[0]))
        res_obj = res_obj.append(series.iloc[i:i + 1])
    sarima_fc = pd.Series(sarima_fc, index=series.index[train_end:])

    # --- RF on lag features (pure ML)
    X = _lag_frame(series)
    valid = X.notna().all(axis=1)
    tr_idx = series.index[:train_end][valid.iloc[:train_end].values]
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=RANDOM_STATE)
    rf.fit(X.loc[tr_idx], series.loc[tr_idx])
    rf_fc = pd.Series(rf.predict(X.iloc[train_end:]), index=series.index[train_end:])

    # --- Hybrid: SARIMA captures seasonality, RF learns SARIMA's residual pattern
    full_resid = resid.reindex(series.index)
    Xr = _lag_frame(series)
    Xr["sarima_pred"] = in_sample.reindex(series.index)
    Xr["resid_lag1"] = full_resid.shift(1)
    Xr["resid_lag12"] = full_resid.shift(12)
    r_valid = Xr.iloc[:train_end].notna().all(axis=1)
    r_idx = series.index[:train_end][r_valid.values][13:]  # skip SARIMA burn-in
    rf_res = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=RANDOM_STATE)
    rf_res.fit(Xr.loc[r_idx], resid.loc[r_idx])

    # Fill test-period residuals as they become known (one-step-ahead)
    test_resid = series.iloc[train_end:] - sarima_fc
    full_resid.iloc[train_end:] = test_resid.values
    Xr["sarima_pred"].iloc[train_end:] = sarima_fc.values
    Xr["resid_lag1"] = full_resid.shift(1)
    Xr["resid_lag12"] = full_resid.shift(12)
    hybrid_fc = sarima_fc + rf_res.predict(Xr.iloc[train_end:].fillna(0))
    hybrid_fc = hybrid_fc.clip(lower=0)

    actual = series.iloc[train_end:]
    preds = {"SARIMA": sarima_fc.clip(lower=0), "Random Forest (lags)": rf_fc,
             "Hybrid SARIMA + RF": hybrid_fc}
    metrics = []
    for name, p in preds.items():
        metrics.append({
            "model": name,
            "rmse": float(np.sqrt(mean_squared_error(actual, p))),
            "mae": mean_absolute_error(actual, p),
            "r2": r2_score(actual, p),
        })
    return actual, preds, pd.DataFrame(metrics)
