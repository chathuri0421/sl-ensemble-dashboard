# Ensemble Learning Dashboard: Sri Lankan Weather

ML Assignment 1. This project compares bagging, boosting, stacking and hybrid models on
real daily weather data from 12 Sri Lankan cities (01/01/15 – 12/31/24).

## What's inside

| File | Purpose |
|---|---|
| `fetch_data.py` | Downloads daily weather for 12 cities from the Open-Meteo archive (ERA5, free, no key) |
| `ml.py` | Features, model zoo (DT, Logistic/Ridge, Bagging, RF, AdaBoost, XGBoost, LightGBM, CatBoost, Stacking, RF+ANN), SARIMA+RF hybrid |
| `run_experiments.py` | Runs every experiment once and saves `results/results.pkl` |
| `app.py` | Streamlit dashboard (8 pages) |

**Tasks**
1. Rain tomorrow (> 1 mm): classification
2. Tomorrow's max temperature: regression
3. Monthly rainfall forecast: time series (SARIMA vs RF vs SARIMA + RF hybrid)

**Fair comparison:** chronological split (train 2015–2022, test 2023–2024), walk-forward
validation over 3 yearly folds, the same features and the same seed for every model.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install libomp            # macOS only: XGBoost and LightGBM need it
.venv/bin/python fetch_data.py
.venv/bin/python run_experiments.py
.venv/bin/streamlit run app.py
```

To deploy for free, push the folder (including `data/` and `results/`) to GitHub and
connect it on Streamlit Community Cloud.

## 3-minute video plan

| Time | Page | What to say |
|---|---|---|
| 0:00–0:30 | 1. Sri Lankan context | Problem, 12 cities, monsoons, Wet/Dry zones, data source |
| 0:30–0:50 | 2. Methods | Bagging vs boosting vs stacking in one sentence each; why a chronological split |
| 0:50–1:40 | 3. Comparison | Best model, gain over baseline, ROC curves, the city × season heatmap |
| 1:40–2:10 | 4. RF vs XGBoost | Fold-by-fold winner, feature importance differences |
| 2:10–2:30 | 5. Playground | Change a slider live and retrain |
| 2:30–2:50 | 6. SARIMA + RF | Hybrid vs SARIMA on Colombo, and how many cities it wins |
| 2:50–3:00 | 7. Research notes | Tabular vs image/text, limitations |

Write the explanations of *why* each model won in your own words, using the numbers on the dashboard.
