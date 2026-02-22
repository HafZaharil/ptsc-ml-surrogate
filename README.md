# PTSC Surrogate Modelling (Eff / EffEX)

Fast surrogate models for parabolic trough solar collector (PTSC) performance, trained on simulation-generated data.  
The goal is to replace slow physics-based runs with near-instant predictions, and to support quick operating-point searches.

This repository trains and compares three regression models and provides an interactive “max finder” that searches for high-efficiency operating points under given ambient conditions.

---

## What this project does

Given a dataset with these inputs:

- `Mhtf` (mass flow rate)
- `Pressurehtf` (HTF pressure)
- `Tin` (inlet temperature)
- `DNI` (direct normal irradiance)
- `Tamb` (ambient temperature)
- `K` (a model factor in the dataset)

…it learns a mapping to either:

- `Eff`  (thermal efficiency)
- `EffEX` (exergetic efficiency)

Then it supports two main uses:

1. **Prediction**: quickly estimate `Eff` / `EffEX` for new inputs.
2. **Search**: for a given `(DNI, Tamb, K)` (and fixed `Pressurehtf`), find:
   - the global best predicted efficiency on a Tin–Mhtf grid
   - the best predicted efficiency for each Tin in `350–850 K` with step `50 K`, including the best `Mhtf` at that Tin

---

## Why surrogate modelling here?

PTSC performance calculations can be slow when they come from detailed physics models and large parameter sweeps.  
A surrogate model turns that into a simple function call:

- input: `(Mhtf, Pressurehtf, Tin, DNI, Tamb, K)`
- output: predicted `Eff` or `EffEX`

Once trained, predictions are fast enough to support:
- dense grid searches
- quick sensitivity checks
- rapid “best operating point” exploration

---

## Methodology (workflow)

### 1) Data
The training and validation datasets are CSV files generated from a physics-based simulation pipeline.

Expected columns:

**Features**
- `Mhtf`, `Pressurehtf`, `Tin`, `DNI`, `Tamb`, `K`

**Targets**
- `Eff` or `EffEX` (depending on the script)

Basic cleaning steps in code:
- strip column names
- replace `inf/-inf` with `NaN`
- drop rows with missing values in features or target

### 2) Train/test split (quick sanity check)
The training CSV is split into train/test (default 80/20) to check if the model is learning correctly and not producing obvious failures.

Metrics reported:
- MAE
- RMSE
- R²

This is mainly a fast internal check. It is not a substitute for external validation.

### 3) External validation 
If you provide a separate validation CSV, the script reports the same metrics on that file.
This is the main indicator of how well the surrogate generalises to unseen combinations.

### 4) Grid-based operating-point search (“max finder”)
For a chosen model and given ambient conditions:
- user enters `DNI`, `Tamb (°C)`, `K`
- `Pressurehtf` is fixed to a constant (default `20000` in the scripts)
- the script searches over `Tin` and `Mhtf`:
  - coarse grid over the full domain
  - fine grid around the best coarse point
- it prints:
  - **the global best predicted operating point**
  - **best predicted value for each Tin (350–850 K, step 50 K)** and the **Mhtf** that achieves it

This is a practical way to turn the surrogate into a decision tool, not just a predictor.

---

## Why these models?

This project uses tree-based regression models because they tend to perform well on structured engineering datasets with:
- non-linear relationships
- interactions between variables
- mixed scaling and non-smooth response regions

### Histogram-based Gradient Boosting (scikit-learn)
Chosen as a strong baseline:
- usually accurate
- fast inference
- handles non-linear behaviour well
- simpler dependency stack than XGBoost

### XGBoost 
Included because it is often a top performer for tabular regression:
- strong accuracy on complex relationships
- robust boosting implementation
- optional dependency (the script runs even if XGBoost is not installed)

### Random Forest (scikit-learn)
Included as a stable reference model:
- strong baseline for non-linear regression
- less sensitive to tuning than boosting in many cases
- easy to interpret as a “bagged trees” benchmark
- The .pkl file are not uploaded due to the huge size of the file.

In short:
- **HGB** = strong baseline, simple stack  
- **XGBoost** = often best accuracy (if available)  
- **RF** = stable benchmark and cross-check  

The “best” model should be decided primarily using the **external validation metrics**, not only train/test split.

---
