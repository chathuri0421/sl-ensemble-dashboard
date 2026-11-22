"""
Ensemble Learning Dashboard: Sri Lankan weather
Run:  streamlit run app.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve

import ml

st.set_page_config(page_title="Ensemble Dashboard: Sri Lanka", layout="wide")

FAMILY_COLORS = {
    "Baseline": "#9aa0a6",
    "Bagging": "#2a9d8f",
    "Boosting": "#e76f51",
    "Stacking": "#6a4c93",
    "Hybrid": "#e9c46a",
}
SEASON_ORDER = ["NE Monsoon", "1st Inter-monsoon", "SW Monsoon", "2nd Inter-monsoon"]
DATE_FMT = "%m/%d/%y"


def pct(x):
    return f"{x * 100:,.1f}%"


def num(x, d=2):
    return f"{x:,.{d}f}"


@st.cache_data
def load_daily():
    return ml.load_daily()


@st.cache_data
def load_features():
    return ml.build_features(ml.load_daily())


RESULTS_PATH = Path(__file__).parent / "results" / "results.pkl"


@st.cache_resource
def load_results(mtime):
    # mtime in the signature: the cache refreshes whenever results.pkl is rewritten
    with open(RESULTS_PATH, "rb") as fh:
        return pickle.load(fh)


daily = load_daily()
R = load_results(RESULTS_PATH.stat().st_mtime) if RESULTS_PATH.exists() else None

st.sidebar.title("🌦️ Ensemble Dashboard")
st.sidebar.caption("Bagging · Boosting · Stacking · Hybrids\non real Sri Lankan weather data")
page = st.sidebar.radio("Go to", [
    "1. Sri Lankan context & data",
    "2. Ensemble methods explained",
    "3. Model comparison",
    "4. RF vs XGBoost deep dive",
    "5. Model playground",
    "6. Hybrid: SARIMA + RF (time series)",
    "7. Research notes",
    "8. Try your own CSV",
])

if R is None and page not in ("1. Sri Lankan context & data", "2. Ensemble methods explained",
                              "5. Model playground", "8. Try your own CSV"):
    st.error("Results not found. Run `python run_experiments.py` first.")
    st.stop()


# ==========================================================================
# 1. Context
# ==========================================================================
if page.startswith("1"):
    st.title("Sri Lankan weather: why it is a good ensemble problem")
    st.markdown(
        """
Sri Lanka's rainfall is driven by **two monsoons and two inter-monsoon periods**, and the
island splits into **Wet, Intermediate and Dry zones**. The same weather signal (humidity,
pressure, wind direction) means different things in Colombo than in Jaffna or Trincomalee.
That kind of non-linear, interaction-heavy tabular data is exactly where tree ensembles shine.

**Data:** daily weather for 12 cities, 01/01/15 to 12/31/24, from the Open-Meteo historical
archive (ERA5 reanalysis). **Tasks:**
- 🌧️ **Classification:** will it rain tomorrow (> 1 mm)?
- 🌡️ **Regression:** tomorrow's maximum temperature
- 📈 **Time series:** monthly rainfall forecasting (SARIMA vs RF vs hybrid)
"""
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cities", f"{daily['city'].nunique():,}")
    c2.metric("Daily records", f"{len(daily):,}")
    c3.metric("Years of data", f"{daily['date'].dt.year.nunique():,}",
              help=f"{daily['date'].min():%m/%d/%y} – {daily['date'].max():%m/%d/%y}")
    c4.metric("Rainy days (> 1 mm)", pct((daily["precipitation_sum"] > 1).mean()))

    left, right = st.columns([1, 1.3])
    with left:
        coords = pd.DataFrame(
            [(c, *ml_c[:2], ml_c[2]) for c, ml_c in __import__("fetch_data").CITIES.items()],
            columns=["city", "lat", "lon", "zone"])
        annual = (daily.groupby("city")["precipitation_sum"].sum() / 10).rename("avg_annual_mm")
        coords = coords.merge(annual, left_on="city", right_index=True)
        fig = px.scatter_geo(coords, lat="lat", lon="lon", color="zone", size="avg_annual_mm",
                             hover_name="city", text="city",
                             hover_data={"avg_annual_mm": ":,.0f", "lat": False, "lon": False},
                             color_discrete_map={"Wet": "#264653", "Intermediate": "#2a9d8f",
                                                 "Dry": "#e76f51"})
        fig.update_traces(textposition="top center")
        fig.update_geos(projection_type="mercator", resolution=50,
                        lataxis_range=[5.7, 10.0], lonaxis_range=[79.4, 82.2],
                        showcountries=True, showland=True, landcolor="#f1f1ea",
                        showocean=True, oceancolor="#e6f2f8", showcoastlines=True)
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0),
                          title="Cities by climate zone (size = avg annual rainfall)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        m = daily.groupby(["city", "month"])["precipitation_sum"].sum().reset_index()
        m["precipitation_sum"] /= 10  # 10 years -> average month
        piv = m.pivot(index="city", columns="month", values="precipitation_sum")
        piv.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig = px.imshow(piv, color_continuous_scale="Blues", aspect="auto",
                        labels=dict(color="mm / month"), text_auto=".0f")
        fig.update_layout(height=520, title="Average monthly rainfall: the two monsoons are visible")
        st.plotly_chart(fig, use_container_width=True)
    st.info("Wet-zone cities (Colombo, Ratnapura, Galle) peak in May (SW monsoon onset) and "
            "again in Oct–Nov. Dry-zone cities (Jaffna, Trincomalee, Batticaloa) get most of "
            "their rain from Oct to Jan (2nd inter-monsoon and NE monsoon) and stay very dry "
            "from Jun to Aug. A single model must learn both regimes.")
    with st.expander("Peek at the raw data"):
        st.dataframe(daily.head(200).assign(date=lambda d: d["date"].dt.strftime(DATE_FMT)),
                     use_container_width=True)


# ==========================================================================
# 2. Methods
# ==========================================================================
elif page.startswith("2"):
    st.title("Ensemble methods in this project")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🌳 Bagging")
        st.markdown(
            "Train many deep trees on **bootstrap samples** in parallel and average them. "
            "This **reduces variance**.\n\n"
            "**Random Forest** also picks a random subset of features at each split, which "
            "decorrelates the trees further.\n\n"
            "*Here:* Bagging (trees), Random Forest")
    with c2:
        st.subheader("🚀 Boosting")
        st.markdown(
            "Train shallow trees **one after another**, each fixing the errors of the ones "
            "before. This **reduces bias**.\n\n"
            "XGBoost adds regularisation; LightGBM grows leaf-wise with histograms (fast); "
            "CatBoost uses ordered boosting and handles categories natively.\n\n"
            "*Here:* AdaBoost, XGBoost, LightGBM, CatBoost")
    with c3:
        st.subheader("🧱 Stacking")
        st.markdown(
            "Train different **base models** (RF, XGBoost, LightGBM), then a **meta-model** "
            "(logistic / ridge regression) learns how to combine their out-of-fold predictions.\n\n"
            "*Here:* Stacking (RF + XGB + LGBM)")
    st.subheader("🔗 Hybrids")
    st.markdown(
        "- **RF + ANN (tabular):** soft-voting average of a Random Forest and a neural network "
        "(MLP 64→32). Trees capture thresholds and interactions; the ANN captures smooth trends.\n"
        "- **SARIMA + RF (time series):** SARIMA models the seasonal monsoon cycle, then an RF "
        "learns the pattern left in SARIMA's residuals. Final forecast = SARIMA + RF(residual).")
    st.subheader("Experimental design")
    st.markdown(
        "- **Chronological split:** train 2015–2022, test 2023–2024. A random split would "
        "leak future weather into training.\n"
        "- **Walk-forward validation:** three folds (test on 2022, 2023, 2024, each "
        "trained on all earlier years) to get mean ± std.\n"
        "- Same features, split and random seed for every model, so it's a fair comparison.\n"
        "- **Features:** today's weather, rainfall lags (1, 2, 3, 7 and 30 days), pressure and "
        "humidity change, season, climate zone, city, and cyclical day-of-year and wind direction.")


# ==========================================================================
# 3. Comparison
# ==========================================================================
elif page.startswith("3"):
    st.title("Model comparison")
    task = st.radio("Task", list(ml.TASKS), format_func=lambda t: ml.TASKS[t]["label"],
                    horizontal=True)
    T = R["tasks"][task]
    is_clf = ml.TASKS[task]["type"] == "classification"
    main_metric = "roc_auc" if is_clf else "rmse"
    asc = not is_clf
    h = T["holdout"].sort_values(main_metric, ascending=asc).reset_index(drop=True)

    st.caption(f"Train rows: {T['n_train']:,} · Test rows: {T['n_test']:,} (2023–2024)")
    best = h.iloc[0]
    base = h[h["family"] == "Baseline"].sort_values(main_metric, ascending=asc).iloc[0]
    c1, c2, c3 = st.columns(3)
    if is_clf:
        c1.metric("Best model (ROC-AUC)", best["model"], pct(best["roc_auc"]), delta_color="off")
        c2.metric("Best baseline", base["model"], pct(base["roc_auc"]), delta_color="off")
        c3.metric("Gain over baseline", f"+{(best['roc_auc'] - base['roc_auc']) * 100:,.1f} pts")
    else:
        c1.metric("Best model (RMSE)", best["model"], f"{best['rmse']:,.3f} °C", delta_color="off")
        c2.metric("Best baseline", base["model"], f"{base['rmse']:,.3f} °C", delta_color="off")
        c3.metric("Error reduction", pct(1 - best["rmse"] / base["rmse"]))

    # table
    show = h.copy()
    if is_clf:
        cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        for c in cols:
            show[c] = show[c].map(pct)
    else:
        cols = ["rmse", "mae", "r2"]
        show["rmse"] = show["rmse"].map(lambda v: num(v, 3))
        show["mae"] = show["mae"].map(lambda v: num(v, 3))
        show["r2"] = show["r2"].map(pct)
    show["train_time_s"] = show["train_time_s"].map(lambda v: num(v, 2))
    st.dataframe(show[["model", "family"] + cols + ["train_time_s"]],
                 use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        fig = px.bar(h, x=main_metric, y="model", color="family", orientation="h",
                     color_discrete_map=FAMILY_COLORS,
                     title=f"Hold-out {main_metric.upper()} ({'higher' if is_clf else 'lower'} is better)")
        lo = h[main_metric].min()
        hi = h[main_metric].max()
        pad = (hi - lo) * 0.3
        fig.update_xaxes(range=[lo - pad, hi + pad / 3],
                         tickformat=".1%" if is_clf else ",.2f")
        fig.update_yaxes(categoryorder="array", categoryarray=h["model"][::-1])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(h, x="train_time_s", y=main_metric, color="family", text="model",
                         color_discrete_map=FAMILY_COLORS, log_x=True,
                         title="Accuracy vs training cost")
        fig.update_traces(textposition="top center", marker_size=12)
        fig.update_yaxes(tickformat=".1%" if is_clf else ",.2f")
        fig.update_xaxes(title="training time (s, log scale)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Walk-forward validation (3 yearly folds, 2022–2024)")
    cv = T["cv"]
    agg = cv.groupby("model")[main_metric].agg(["mean", "std"]).reset_index()
    agg = agg.sort_values("mean", ascending=asc)
    agg["family"] = agg["model"].map(ml.FAMILY)
    fig = px.bar(agg, x="model", y="mean", error_y="std", color="family",
                 color_discrete_map=FAMILY_COLORS,
                 title=f"Mean {main_metric.upper()} ± std across folds")
    fig.update_yaxes(range=[agg["mean"].min() - 3 * agg["std"].max(),
                            agg["mean"].max() + 2 * agg["std"].max()],
                     tickformat=".1%" if is_clf else ",.2f")
    st.plotly_chart(fig, use_container_width=True)
    fold_piv = cv.pivot(index="model", columns="fold", values=main_metric)
    st.dataframe(fold_piv.style.format(pct if is_clf else (lambda v: num(v, 3))),
                 use_container_width=True)

    if is_clf:
        st.subheader("ROC curves & confusion matrix")
        l2, r2 = st.columns(2)
        with l2:
            fig = go.Figure()
            for name, proba in T["probas"].items():
                fpr, tpr, _ = roc_curve(T["y_test"], proba)
                fig.add_scatter(x=fpr, y=tpr, mode="lines", name=name,
                                line=dict(color=FAMILY_COLORS[ml.FAMILY[name]],
                                          dash="dot" if ml.FAMILY[name] == "Baseline" else None))
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="random",
                            line=dict(color="lightgrey", dash="dash"))
            fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                              height=480)
            st.plotly_chart(fig, use_container_width=True)
        with r2:
            pick = st.selectbox("Model", list(T["preds"]), index=list(T["preds"]).index(best["model"]))
            cm = confusion_matrix(T["y_test"], T["preds"][pick])
            fig = px.imshow(cm, text_auto=",", x=["Pred: dry", "Pred: rain"],
                            y=["Actual: dry", "Actual: rain"], color_continuous_scale="Teal")
            fig.update_layout(height=420, title=pick)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Where does the best model struggle? (by city & season)")
    meta = T["test_meta"].copy()
    if is_clf:
        meta["correct"] = (T["preds"][best["model"]] == T["y_test"]).astype(float)
        g = meta.groupby(["city", "season"])["correct"].mean().reset_index()
        lab = "accuracy"
    else:
        meta["abs_err"] = np.abs(T["preds"][best["model"]] - T["y_test"])
        g = meta.groupby(["city", "season"])["abs_err"].mean().reset_index()
        lab = "MAE °C"
    piv = g.pivot(index="city", columns="season", values=g.columns[-1])[SEASON_ORDER]
    fig = px.imshow(piv, text_auto=".1%" if is_clf else ".2f", aspect="auto",
                    color_continuous_scale="RdYlGn" if is_clf else "RdYlGn_r",
                    labels=dict(color=lab))
    fig.update_layout(height=480, title=f"{best['model']}: {lab} per city per season")
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# 4. RF vs XGBoost
# ==========================================================================
elif page.startswith("4"):
    st.title("Random Forest vs XGBoost")
    task = st.radio("Task", list(ml.TASKS), format_func=lambda t: ml.TASKS[t]["label"],
                    horizontal=True)
    T = R["tasks"][task]
    is_clf = ml.TASKS[task]["type"] == "classification"
    h = T["holdout"].set_index("model")
    metrics = ["accuracy", "f1", "roc_auc"] if is_clf else ["rmse", "mae", "r2"]
    comp = h.loc[["Random Forest", "XGBoost"], metrics + ["train_time_s"]].T
    comp["difference (XGB − RF)"] = comp["XGBoost"] - comp["Random Forest"]
    fmt = {c: (lambda v: num(v, 4)) for c in comp.columns}
    st.dataframe(comp.style.format(fmt), use_container_width=True)

    cv = T["cv"][T["cv"]["model"].isin(["Random Forest", "XGBoost"])]
    key = "roc_auc" if is_clf else "rmse"
    fig = px.line(cv, x="fold", y=key, color="model", markers=True,
                  color_discrete_map={"Random Forest": "#2a9d8f", "XGBoost": "#e76f51"},
                  title=f"{key.upper()} per walk-forward fold")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickformat=".1%" if is_clf else ",.3f")
    st.plotly_chart(fig, use_container_width=True)
    wins = (cv.pivot(index="fold", columns="model", values=key)
              .pipe(lambda p: (p["XGBoost"] > p["Random Forest"]) if is_clf
                    else (p["XGBoost"] < p["Random Forest"]))).sum()
    st.write(f"XGBoost beats Random Forest in **{wins:,} of 3** yearly folds.")

    st.subheader("What each model relies on (feature importance)")
    top = st.slider("Top features", 5, 25, 12)
    c1, c2 = st.columns(2)
    for col, name, color in [(c1, "Random Forest", "#2a9d8f"), (c2, "XGBoost", "#e76f51")]:
        imp = T["importances"][name].head(top)[::-1]
        fig = px.bar(x=imp.values, y=imp.index, orientation="h", title=name,
                     color_discrete_sequence=[color])
        fig.update_xaxes(title="importance (share)", tickformat=".1%")
        fig.update_yaxes(title="")
        col.plotly_chart(fig, use_container_width=True)
    st.caption("RF importance = mean impurity decrease (tends to favour continuous features). "
               "XGBoost importance = gain. Different definitions, so compare rankings, not raw values.")


# ==========================================================================
# 5. Playground
# ==========================================================================
elif page.startswith("5"):
    st.title("Model playground: train live")
    f = load_features()
    task = st.radio("Task", list(ml.TASKS), format_func=lambda t: ml.TASKS[t]["label"],
                    horizontal=True)
    c1, c2, c3, c4 = st.columns(4)
    names = list(ml.get_models(task))
    model_name = c1.selectbox("Model", names, index=names.index("Random Forest"))
    n_est = c2.slider("n_estimators", 20, 600, 200, step=20)
    depth = c3.slider("max_depth", 2, 16, 8)
    lr = c4.select_slider("learning_rate (boosting)", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], 0.05)
    cities = st.multiselect("Cities to include", sorted(f["city"].unique()),
                            default=sorted(f["city"].unique()))
    show_city = st.selectbox("City to plot (regression)", cities) if cities else None
    if st.button("Train & evaluate", type="primary", disabled=not cities):
        sub = f[f["city"].isin(cities)].reset_index(drop=True)
        X, y = ml.make_xy(sub, task)
        X_tr, X_te, y_tr, y_te = ml.time_split(sub, X, y)
        model = ml.get_models(task, n_est, depth, lr)[model_name]
        with st.spinner(f"Training {model_name} on {len(X_tr):,} rows…"):
            res, pred, proba = ml.evaluate(model, X_tr, X_te, y_tr, y_te, task)
        cols = st.columns(len(res))
        for col, (k, v) in zip(cols, res.items()):
            if k.endswith("_s"):
                col.metric(k, f"{v:,.2f} s")
            elif k in ("rmse", "mae"):
                col.metric(k.upper(), f"{v:,.3f}")
            else:
                col.metric(k.upper(), pct(v))
        if ml.TASKS[task]["type"] == "regression":
            d = sub.loc[X_te.index, ["date", "city"]].assign(actual=y_te.values, predicted=pred)
            city = show_city
            d = d[d["city"] == city]
            fig = px.line(d, x="date", y=["actual", "predicted"],
                          title=f"{city}: actual vs predicted max temperature")
            fig.update_xaxes(tickformat=DATE_FMT)
            st.plotly_chart(fig, use_container_width=True)
        imp = ml.feature_importance(model, X.columns)
        if imp is not None:
            fig = px.bar(imp.head(15)[::-1], orientation="h", title="Top 15 features")
            fig.update_layout(showlegend=False)
            fig.update_xaxes(tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pick a model and hyperparameters, then press **Train & evaluate**. "
                "Try increasing max_depth for Random Forest vs XGBoost and watch which one overfits.")


# ==========================================================================
# 6. Time series hybrid
# ==========================================================================
elif page.startswith("6"):
    st.title("Hybrid SARIMA + Random Forest: monthly rainfall")
    st.markdown("Each model forecasts **one month ahead** for 01/01/23 – 12/01/24, trained on "
                "2015–2022 and updated with each new observed month.")
    TS = R["timeseries"]
    summary = pd.concat([v["metrics"].assign(city=k) for k, v in TS.items()])
    city = st.selectbox("City", list(TS), index=list(TS).index("Colombo"))
    d = TS[city]
    c1, c2, c3 = st.columns(3)
    for col, (_, r) in zip([c1, c2, c3], d["metrics"].iterrows()):
        col.metric(r["model"], f"RMSE {r['rmse']:,.1f} mm", f"R² {r['r2'] * 100:,.1f}%",
                   delta_color="off")
    fig = go.Figure()
    s = d["series"]
    fig.add_scatter(x=s.index, y=s.values, name="actual", line=dict(color="#264653"))
    colors = {"SARIMA": "#457b9d", "Random Forest (lags)": "#2a9d8f", "Hybrid SARIMA + RF": "#e76f51"}
    for name, p in d["preds"].items():
        fig.add_scatter(x=p.index, y=p.values, name=name, line=dict(color=colors[name], dash="dot"))
    fig.add_vrect(x0=d["actual"].index[0], x1=d["actual"].index[-1], fillcolor="grey",
                  opacity=0.08, line_width=0, annotation_text="test period")
    fig.update_layout(height=480, yaxis_title="rainfall (mm / month)")
    fig.update_xaxes(tickformat=DATE_FMT)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All 12 cities")
    piv = summary.pivot(index="city", columns="model", values="rmse")
    piv["best"] = piv.idxmin(axis=1)
    st.dataframe(piv.style.format({c: (lambda v: f"{v:,.1f}") for c in piv.columns if c != "best"}),
                 use_container_width=True)
    wins = piv["best"].value_counts()
    st.write("Cities where each model has the lowest RMSE: " +
             ", ".join(f"**{k}** {v:,}" for k, v in wins.items()))
    mean = summary.groupby("model")[["rmse", "mae"]].mean()
    fig = px.bar(mean.reset_index(), x="model", y="rmse", color="model",
                 color_discrete_map=colors, title="Mean RMSE across cities (mm)")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# 7. Research notes
# ==========================================================================
elif page.startswith("7"):
    st.title("Research notes: RF vs gradient boosting")
    st.markdown(
        """
| | Random Forest (bagging) | Gradient boosting (XGBoost / LightGBM / CatBoost) |
|---|---|---|
| How trees are built | Independently, in parallel, deep trees | One after another, shallow trees fixing residuals |
| Reduces mainly | Variance | Bias |
| Tuning effort | Low; works well out of the box | Higher (learning rate, depth, n_estimators, regularisation) |
| Overfitting risk | Low; more trees never hurts | Can overfit without early stopping or regularisation |
| Speed | Parallel training; large memory for deep trees | LightGBM is very fast; XGBoost is fast with `hist` |
| Categorical data | Needs encoding | CatBoost handles categories natively |
| Typical tabular accuracy | Strong baseline | Usually slightly better (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022) |

### Tabular vs image/text data
- **Tabular:** tree ensembles are still state of the art on most medium-sized tabular datasets.
  They handle mixed feature types, missing values, irregular decision boundaries and unscaled
  features without extra work.
- **Image/text:** raw pixels and tokens have spatial or sequential structure. Trees split on
  single features, so they can't learn convolution-style or attention-style patterns.
  CNNs and Transformers win clearly there.
- **Hybrid use:** a common pattern is to use a neural network to extract **embeddings** from
  images or text, then feed them, with tabular features, into XGBoost or LightGBM.

### Limitations of this study
- ERA5 reanalysis is gridded (about 25 km), so it smooths local extremes compared with station gauges.
- 12 cities only; the hill country and Northern Province are under-represented.
- Rain/no-rain uses a fixed 1 mm threshold, and heavy-rain events (> 50 mm) are not modelled separately.
"""
    )


# ==========================================================================
# 8. Upload
# ==========================================================================
elif page.startswith("8"):
    st.title("Try your own dataset")
    st.markdown("Upload any CSV (for example telecom churn or patient data), pick the target "
                "column, and compare the ensembles with a stratified 80.0% / 20.0% split.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(up)
        st.dataframe(df.head(), use_container_width=True)
        target = st.selectbox("Target column", df.columns, index=len(df.columns) - 1)
        drop = st.multiselect("Columns to ignore (IDs, etc.)", [c for c in df.columns if c != target])
        y_raw = df[target]
        is_clf = y_raw.dtype == object or y_raw.nunique() <= 10
        st.write(f"Detected task: **{'classification' if is_clf else 'regression'}**")
        if st.button("Compare models", type="primary"):
            data = df.drop(columns=drop).dropna(subset=[target])
            y = data[target]
            X = data.drop(columns=[target])
            X = pd.get_dummies(X, dtype=float).fillna(X.median(numeric_only=True)).fillna(0)
            X.columns = [str(c).replace("[", "_").replace("]", "_").replace("<", "_") for c in X.columns]
            if is_clf:
                classes = sorted(y.astype(str).unique())
                if len(classes) != 2:
                    st.error("This page supports binary classification or regression only.")
                    st.stop()
                y = (y.astype(str) == classes[1]).astype(int)
                st.caption(f"Positive class = '{classes[1]}'")
            task = "rain" if is_clf else "temp"  # reuse the model zoo for that task type
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if is_clf else None)
            rows = []
            prog = st.progress(0.0)
            models = ml.get_models(task, 200, 8, 0.05)
            for i, (name, model) in enumerate(models.items()):
                res, _, _ = ml.evaluate(model, X_tr, X_te, y_tr, y_te, task)
                rows.append({"model": name, "family": ml.FAMILY[name], **res})
                prog.progress((i + 1) / len(models), text=name)
            out = pd.DataFrame(rows)
            key = "roc_auc" if is_clf else "rmse"
            out = out.sort_values(key, ascending=not is_clf)
            st.dataframe(out, use_container_width=True, hide_index=True)
            fig = px.bar(out, x=key, y="model", color="family", orientation="h",
                         color_discrete_map=FAMILY_COLORS)
            st.plotly_chart(fig, use_container_width=True)
