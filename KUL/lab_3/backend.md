
# Backend Module – Function Documentation

This document describes all public functions found in `backend.py`. It focuses on what each function does, its inputs/outputs, important notes and typical usage.  
> **Context:** The module builds synthetic substation-level datasets by aggregating consumer load profiles, simulating PV generation with `pvlib`, extracting engineered features (including solar-time–aligned ones), training ML models to estimate installed PV capacity, and running several optimization/estimation baselines. It also contains utilities to visualize time series and to load external benchmark datasets (Ausgrid, Pecan Street). 

---

## Table of Contents
- [Data synthesis & feature engineering](#data-synthesis--feature-engineering)
  - [`generate_random_numbers_uniform`](#generate_random_numbers_uniform)
  - [`calculate_poa`](#calculate_poa)
  - [`add_solar_tau`](#add_solar_tau)
  - [`createLoads`](#createloads)
  - [`createGeneration`](#creategeneration)
  - [`extractFeatures`](#extractfeatures)
  - [`extractFeatures_solar`](#extractfeatures_solar)
  - [`createSubstationData`](#createsubstationdata)
  - [`loadDataFluvius`](#loaddatafluvius)
- [Visualization](#visualization)
  - [`visualizeSubstationLoad`](#visualizesubstationload)
  - [`visualizeSubstationLoadSolarTime`](#visualizesubstationloadsuntime)
- [Train/test preparation](#traintest-preparation)
  - [`splitData`](#splitdata)
  - [`viewFeatures`](#viewfeatures)
  - [`viewFeatureCorrelations`](#viewfeaturecorrelations)
- [Model training](#model-training)
  - [`tune_and_train_xgb`](#tune_and_train_xgb)
  - [`tune_and_train_lasso`](#tune_and_train_lasso)
  - [`tune_and_train_elast_net`](#tune_and_train_elast_net)
  - [`tune_and_train_svr`](#tune_and_train_svr)
  - [`create_ffnn`](#create_ffnn)
  - [`tune_and_train_ffnn`](#tune_and_train_ffnn)
  - [`train_all`](#train_all)
  - [`loadTrainedModels`](#loadtrainedmodels)
- [Optimization/baseline estimators](#optimizationbaseline-estimators)
  - [`method_a_optimization`](#method_a_optimization)
  - [`method_b_optimization`](#method_b_optimization)
  - [`method_c_optimization`](#method_c_optimization)
  - [`calculate_methodINM`](#calculate_methodinm)
  - [`calculate_methodINM_POA`](#calculate_methodinm_poa)
  - [`modelBasedPrections`](#modelbasedprections)
- [Evaluation & plotting](#evaluation--plotting)
  - [`predict_and_plot`](#predict_and_plot)
- [External datasets](#external-datasets)
  - [`loadAusGridData`](#loadausgriddata)
  - [`loadTexasData`](#loadtexasdata)

---

## Data synthesis & feature engineering

### `generate_random_numbers_uniform(n, S)`
**Purpose:** Split a positive scalar `S` into `n` positive segments using sorted uniform cut points (Dirichlet‑like).  

**Args**
- `n` *(int)* — number of segments.
- `S` *(float)* — total sum to partition.

**Returns**
- `np.ndarray` of length `n` with non‑negative entries summing to `S`.  

---

### `calculate_poa(weather, solar_position, surface_tilt, surface_azimuth)`
**Purpose:** Compute plane‑of‑array (POA) global irradiance via `pvlib.irradiance.get_total_irradiance` using the isotropic sky model.  

**Args**
- `weather` *(pd.DataFrame)* — columns `dni`, `ghi`, `dhi`; indexed by timestamps.
- `solar_position` *(pd.DataFrame)* — columns `apparent_zenith`, `azimuth` at matching times.
- `surface_tilt` *(float)* — panel tilt (deg).
- `surface_azimuth` *(float)* — panel azimuth (deg).

**Returns**
- `pd.Series` — POA global irradiance (`poa_global`).  

---

### `add_solar_tau(df)`
**Purpose:** Add daylight mask and relative solar‑time variable `tau∈[0,1]` within each day based on location sun elevation.  

**Args**
- `df` *(pd.DataFrame)* — contains column `location` (pvlib `Location`) and a timezone‑aware DateTimeIndex.

**Returns**
- `pd.DataFrame` — input plus `daylight` *(bool)* and `tau` *(float; NaN at night)*.  

---

### `createLoads(nonSolarIDs, lp_df, n, size=50)`
**Purpose:** Create `n` aggregated load profiles by sampling `size` non‑PV EANs and summing their 15‑minute loads.  

**Args**
- `nonSolarIDs` *(pd.Series)* — EAN IDs without PV.
- `lp_df` *(pd.DataFrame)* — Fluvius data with `EAN_ID`, `Datetime`, `Load`.
- `n` *(int)* — number of aggregated profiles.
- `size` *(int)* — customers per aggregate.

**Returns**
- `list[pd.DataFrame]` — per‑aggregate frames (index `Datetime`) with columns `Load`, `Generation`, `Net`, `Date`, `Time`.  

---

### `createGeneration(dfs_in, azimuths, tilts, weather_path, weather_files, size=50, Power=4, Pen=0.5)`
**Purpose:** Simulate PV generation for each aggregated load using `pvlib.ModelChain`, randomizing tilt/azimuth per panel and the weather location.  

**Args**
- `dfs_in` *(list[pd.DataFrame])* — aggregated loads.
- `azimuths`, `tilts` *(iterables of float)* — configuration ranges.
- `weather_path` *(str)* and `weather_files` *(list[str])* — Solcast CSV directory and filenames.
- `size` *(int)* — aggregation size (used in panel count).
- `Power` *(float)* — module `pdc0` rating (kW).
- `Pen` *(float)* — PV penetration fraction; `nPanels = max(int(size*Pen),1)`.

**Returns**
- `list[pd.DataFrame]` — augmented frames with `Generation`, `Net`, `ghi/dhi/dni`, `location`, and `PVkWp`.  

---

### `extractFeatures(dfs_in, D=30, res=4)`
**Purpose:** (Clock‑time) Build weekday (05:00–19:00) daily pivots for Net and GHI; compute per‑time statistics (mean/std/min/max/range/skew/kurt), correlation and covariance with irradiance; compute POA grids over `tilts×azimuths`.  

**Args**
- `dfs_in` *(list[pd.DataFrame])* — per‑substation data.
- `D` *(int)* — reserved (unused in current implementation).
- `res` *(int)* — reserved (unused here).

**Returns**
- `dfs` *(list[pd.DataFrame])* — filtered/augmented series.
- `substation_data_MC` *(list[dict])* — engineered features + labels `maxPV`, `baseline`.
- `Is` *(list[np.ndarray])* — POA matrices of shape `(len(tilts)*len(azimuths), T)`.
- `DailyStats_list` *(list[dict])* — raw stats.  

---

### `extractFeatures_solar(dfs_in, azimuths, tilts, res=4)`
**Purpose:** As above, but bin by **relative solar time** `tau` instead of clock time. Adds `tau`, constructs daily pivots by `tau_bin`, normalizes GHI by 99th percentile, computes stats/correlations, and builds POA matrices.  

**Args**
- `dfs_in` *(list[pd.DataFrame])* — per‑substation data.
- `azimuths`, `tilts` — configuration grids.
- `res` *(int)* — number of `tau` bins.

**Returns**
- `dfs`, `substation_data_MC`, `Is`, `DailyStats_list` — same structure as `extractFeatures`.  

---

### `createSubstationData(size=50, nonSolarIDs=nonSolarIDs, lp_df=lp_df, Power=Power, n=1, weather_files=weather_files, weather_path=weather_path, azimuths=azimuths, tilts=tilts, Pen=0.5)`
**Purpose:** End‑to‑end pipeline: create aggregated loads → simulate PV → extract **solar‑time** features for `n` synthetic substations. Default IDs, data and weather config are defined at module import.  

**Args**
- `size` *(int)* — customers per aggregate.
- `nonSolarIDs`, `lp_df` — Fluvius IDs and dataframe from globals.
- `Power` *(float)* — module power (kW).
- `n` *(int)* — number of substations to synthesize.
- `weather_files`, `weather_path` — Solcast weather sources.
- `azimuths`, `tilts` — configuration grids.
- `Pen` *(float)* — PV penetration used to compute panel count.

**Returns**
- `substation_data_Df_MC` *(pd.DataFrame)* — features + targets (`maxPV`, `baseline`).
- `sizes_MC` *(list[int])* — aggregation sizes.
- `dfs` *(list[pd.DataFrame])* — time series per substation.
- `Is` *(list[np.ndarray])* — POA matrices.  

---

### `loadDataFluvius()`
**Purpose:** Load precomputed Fluvius artifacts from `data/`: `substationDf_Irr_MC.pkl`, `sizes_MC.pkl`, `dfs.pkl`, `Is.pkl`.  

**Returns**
- `(substation_data_Df_MC, sizes_MC, dfs, Is)` in that order.  

---

## Visualization

### `visualizeSubstationLoad(dfs, index=0, date='2022-01-03', vmin=None, vmax=None)`
**Purpose:** Plot, for a selected substation and date, (i) Net vs GHI time series, and (ii) a year‑view heatmap (date × time‑of‑day) of Net.  

**Args**
- `dfs` *(list[pd.DataFrame])*, `index` *(int)*, `date` *(str YYYY-MM-DD)*, optional `vmin`, `vmax` for color scaling.

**Returns**
- `None` — displays figures.  

---

### `visualizeSubstationLoadSolarTime(dfs, index=0, date='2022-01-03', res=10, vmin=None, vmax=None)`
**Purpose:** As above but daily plot uses **relative solar time** `tau`, and the year heatmap is binned by `tau_bin` with `res` bins.  

**Args** / **Returns** — same pattern as `visualizeSubstationLoad`.  

---

## Train/test preparation

### `splitData(substation_data_Df_MC, dfs, Is, test_size=0.1)`
**Purpose:** Split features into train/test by index; standardize features; keep aligned `dfs` and `Is` lists.  

**Args**
- `substation_data_Df_MC` *(pd.DataFrame)*, `dfs` *(list)*, `Is` *(list)*, `test_size` *(float)*.

**Returns**
A dict with keys: `dfs_Train`, `dfs_Test`, `Is_Train`, `Is_Test`, `XIrrTrain_MC`, `XIrrTrain_MC_normalized`, `yIrrTrain_MC`, `XIrrTest_MC`, `XIrrTest_MC_normalized`, `yIrrTest_MC`, `scaler`.  

---

### `viewFeatures(X, y)`
**Purpose:** Plot scatter of each feature vs target (`y`).  
**Args/Returns:** `X` *(pd.DataFrame)*, `y` *(pd.Series)* → displays plots.  

### `viewFeatureCorrelations(X, y)`
**Purpose:** Show styled correlation matrix for `[X  y]`.  
**Args/Returns:** Same as above; renders a styled DataFrame (Jupyter).  

---

## Model training

### `tune_and_train_xgb(XIrrTrain_MC, yIrrTrain_MC)`
**Purpose:** Hyperopt search over `XGBRegressor` hyperparameters with 5‑fold CV (MSE); fit final model on full training data.  

**Returns**
- `(xgb_model, best_hyperparams_dict)`.  

---

### `tune_and_train_lasso(XIrrTrain_MC, yIrrTrain_MC)`
**Purpose:** Hyperopt (log‑uniform) for `alpha`; return fitted `Lasso`.  

---

### `tune_and_train_elast_net(XIrrTrain_MC, yIrrTrain_MC)`
**Purpose:** Hyperopt `alpha` and `l1_ratio`; return fitted `ElasticNet`.  

---

### `tune_and_train_svr(XIrrTrain_MC, yIrrTrain_MC)`
**Purpose:** GridSearchCV over `C`, `epsilon`, `degree`, `kernel`; return best `SVR`.  

---

### `create_ffnn(neurons=32, dropout_rate=0, learning_rate=0.05, kr_l1=1e-4, kr_l2=1e-4, n_features=1)`
**Purpose:** Build a simple Keras FFNN: Dense(ReLU) → Dropout → Dense(1), optimizer Adam (MSE loss).  

---

### `tune_and_train_ffnn(X, y, max_evals=200)`
**Purpose:** Hyperopt search (3‑fold CV) over `neurons`, `learning_rate`, `kr_l1`, `kr_l2` with early stopping; train best model on full data.  
**Returns:** `(best_model, history, best, trials)`.  

---

### `train_all(X, y)`
**Purpose:** Standardize `X`, then train and return: XGBoost, LinearRegression, Lasso, ElasticNet, SVR, FFNN. *(Note: XGBoost is trained on **unscaled** `X`.)*  

---

### `loadTrainedModels(models=["Support Vector Regression"])`
**Purpose:** Load persisted models from `models/` based on names in `models` list. Supported: XGBoost, Linear Regression, Lasso Regression, Elastic Net, Support Vector Regression, Feedforward Neural Network.  
**Returns:** dict `name -> model`.  

---

## Optimization/baseline estimators

All optimization problems use CVXPY; PV capacity vector `x` (per tilt–azimuth) is constrained to be non‑negative; POA matrices `I` are scaled by `/1000` (W/m² → kW/m²).  

### `method_a_optimization(netLoads, Is, tilts, azimuths, verbose=True)`
**Purpose:** Estimate PV by minimizing the **L1 total‑variation** of the first difference of residual `netLoad + (x^T I)`.  

$$\min_{PV^{kWp}} \quad \sum_t^T \left | (P^{net}_t - P^{net} _{t-1}) + PV^{kWp} (I_t - I _{t-1}) \right |$$

**Returns:** `list[float]` — total kWp (`sum(x)`) per substation.  

---

### `method_b_optimization(netLoads, Is, tilts, azimuths, lambda_reg=1, verbose=True)`
**Purpose:** Jointly estimate smooth non‑negative load `y` ($\hat{L}$) and PV vector `x` ($PV^{kWp}$) via

$$\min_{PV^{kWp}, \hat{L}} \quad \sum_t^T \left (P^{net}_{t} + PV^{kWp} \cdot I_t - \hat{L}_t \right )^2 + \lambda \sum_t^T \left |\hat{L}_t - \hat{L} _{t-1}\right |$$

Solver fallback order: CLARABEL → SCS → ECOS.  
**Returns:** `list[float]` — total kWp (`sum(x)`) per substation.  

---

### `method_c_optimization(netLoads, Is, tilts, azimuths, c=5, verbose=True)`
**Purpose:** As Method B but enforce **piecewise‑constant** load by constraining most entries of `Δy` to zero, allowing changes every `c` steps. 

$$\min_{PV^{kWp}, \hat{L}} \quad  \sum_t^T \left (P^{net}_t + PV^{kWp} \cdot I_t - \hat{L}_t \right )^2 $$
$$        \textrm{s.t.} \quad  PV^{kWp} \geq 0 $$
$$                          \hat{L} _{c(i-1)+1} = ... = \hat{L} _{c(i-1)+c} \quad i = 1, ..., T/c$$
$$                           \hat{L} \geq 0$$

**Returns:** `(methodC, Loads)` where `methodC` is a list of `sum(x)` per substation and `Loads` stores the CVXPY variables `y`.  

---

### `calculate_methodINM(dfs, verbose=True)`
**Purpose:** Simple irradiance‑net slope: fit `Net ~ GHI/1000` using `LinearRegression` and return `abs(coef_)` as a PV capacity proxy.  
**Returns:** `list[float]` — absolute value of slope (`np.abs(model.coef_[0][0])`) as total kWp per substation.  

---

### `calculate_methodINM_POA(dfs, Is, alpha=28, verbose=True)`
**Purpose:** Ridge regression of `Net` on POA columns `I^T/1000`; return the sum of absolute coefficients as PV proxy.  
**Returns:** `list[float]` — sum of absolute value of slopes (`np.abs(model.coef_[0].sum())`) total kWp per substation.

---

### `modelBasedPrections(dfs, Is, load_estimation=True, tilts=tilts, azimuths=azimuths, methods_in=['Method INM (POA)', 'Method B'])`
**Purpose:** Either **load** precomputed method outputs from `data/*.pkl` when `load_estimation=True`, or compute them on the fly.  
**Returns:** `dict[str, list[float]]` — method name → predictions.  

---

## Evaluation & plotting

### `predict_and_plot(XTest_scaled, yTest, models, methods)`
**Purpose:** For each ML model and each baseline method, compute predictions, RMSE vs `yTest`, and plot scatter with 1:1 reference line.  
**Returns:** `None` — displays figures.  

---

## External datasets

### `loadAusGridData()`
**Purpose:** Build 30‑minute Ausgrid aggregates for postcodes {2259, 2261, 2290}; attach matching Solcast weather; compute solar‑time features and scale with `models/scaler_30min.pkl`.  
**Returns:**
- `aus_dfs` *(dict[int, list[pd.DataFrame]])*
- `features_postcodes` *(dict[int, np.ndarray])*
- `Is_postcodes` *(dict[int, list[np.ndarray]])*
- `real_values_postcodes` *(dict[int, pd.Series])*  

---

### `loadTexasData(scaler=None)`
**Purpose:** Build an Austin (US/Central) aggregate from Pecan Street (grid + on‑site PV) and Solcast weather; compute solar‑time features and scale with a provided or stored scaler.  
**Returns:**
- `texas_dfs` *(list[pd.DataFrame])*
- `tx_features_scaled` *(np.ndarray)*
- `texas_Is` *(list[np.ndarray])*
- `tx_subDataDf['maxPV']` *(pd.Series)*  

---

## Notes & caveats
- Many functions expect timezone‑aware DateTimeIndex and presence of columns: `Net`, `ghi`, `dhi`, `dni`, `PVkWp`, `location`, `Date`, `Time`.  
- Optimization methods use POA `Is` scaled by `/1000` to convert W/m² → kW/m².  
- Random elements: weather file selection per substation; per‑panel tilt/azimuth; sampled customers — runs are non‑deterministic unless seeds are fixed.  
- Disk dependencies: several loaders expect files under `data/` and `models/`. Import‑time code reads `data/15minuteFluvius.csv` and lists `data/solcast_20cities_flanders/`.  

## Minimal example
```python
# Build one synthetic substation and train a quick model
sub_df, sizes, dfs, Is = createSubstationData(size=50, n=10)
split = splitData(sub_df, dfs, Is, test_size=0.2)

# Train a lasso quickly
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01).fit(split['XIrrTrain_MC_normalized'], split['yIrrTrain_MC'])

# Evaluate simple INM baseline
methods = modelBasedPrections(split['dfs_Test'], split['Is_Test'], load_estimation=False,
                              methods_in=['Method INM (POA)'])
```
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>