# PTSC Surrogate Modelling (Eff / EffEX)

A supervised machine learning project for fast surrogate models of Parabolic Trough Solar Collector (PTSC) performance, trained on validated simulation-generated data from Engineering Equation Solver (EES).

The objective is to replace slow physics-based simulations with near-instant predictions and enable rapid operating-point exploration.

This repository trains and compares tree-based regression models and provides an interactive search tool to identify high-efficiency operating conditions.

---

## Project Overview

The surrogate models learn the relationship between operating conditions and PTSC performance.

### Input Features

- **Mhtf** — Mass flow rate  
- **Pressurehtf** — Heat transfer fluid pressure  
- **Tin** — Inlet temperature  
- **DNI** — Direct normal irradiance  
- **Tamb** — Ambient temperature  
- **K** — Model factor from dataset  

### Target Outputs

- **Eff** — Thermal efficiency  
- **EffEX** — Exergetic efficiency  

After training, the model predicts efficiency directly from operating conditions without requiring repeated thermodynamic simulations.

---

## Why Surrogate Modelling?

Detailed PTSC simulations based on thermodynamic and heat-transfer models can become computationally expensive during:

- Large parametric sweeps  
- Sensitivity analysis  
- Optimisation loops  

A surrogate model replaces repeated physics evaluations with a trained regression function:

```
(Mhtf, Pressurehtf, Tin, DNI, Tamb, K)
                ↓
     Predicted Eff / EffEX
```

Once trained, predictions are extremely fast and suitable for:

- Dense grid searches  
- Sensitivity studies  
- Rapid operating-point exploration  
- Decision-support workflows  

---

## Methodology

### 1. Data Generation

Training and validation datasets are generated from a validated physics-based simulation pipeline implemented in EES.

Expected CSV columns:

#### Features

- Mhtf  
- Pressurehtf  
- Tin  
- DNI  
- Tamb  
- K  

#### Target

- Eff (for Eff model)  
- EffEX (for EffEX model)  

---

### 2. Data Cleaning

The scripts perform basic preprocessing:

- Strip column names  
- Replace `inf` and `-inf` with `NaN`  
- Drop rows with missing values in features or target  

---

### 3. Train/Test Split

The training dataset is split (default 80/20) to provide a quick internal performance check.

Reported metrics:

- **MAE** (Mean Absolute Error)  
- **RMSE** (Root Mean Squared Error)  
- **R²** (Coefficient of Determination)  

This verifies that the model is learning meaningful structure and not exhibiting obvious instability.

---

### 4. External Validation

If a separate validation dataset is provided, the same metrics are computed on unseen combinations.

External validation performance should guide final model selection.

---

## Models Used

Tree-based regression models are selected because PTSC behaviour exhibits:

- Strong non-linearity  
- Interaction between variables  
- Structured tabular data characteristics  

### Histogram-Based Gradient Boosting (scikit-learn)

- Strong baseline model  
- Good accuracy  
- Fast inference  
- Minimal dependency stack  

### XGBoost

- Often high performance on tabular regression  
- Robust gradient boosting implementation  
- Optional dependency  

### Random Forest

- Stable benchmark model  
- Less sensitive to hyperparameters  
- Useful comparison against boosting approaches  

Model choice should be determined primarily by external validation metrics.

---

## Grid-Based Operating-Point Search

The repository includes an interactive search tool.

Given user inputs:

- DNI  
- Tamb (°C)  
- K  

With **Pressurehtf fixed** (default 20000), the script:

1. Performs a coarse grid search over Tin and Mhtf  
2. Refines the search locally around the best coarse result  
3. Reports:
   - Global maximum predicted efficiency  
   - Best predicted efficiency for each Tin from **350 K to 850 K (step 50 K)**  
   - The corresponding mass flow rate for each Tin  

This transforms the surrogate into a practical optimisation tool rather than a simple predictor.

---

## Repository Structure

```
├── models/
│   ├── eff_hgb.pkl
│   ├── eff_xgb.pkl
│   ├── effex_hgb.pkl
│   ├── effex_xgb.pkl
│
├── eff_model.py                # Full version (HGB + XGBoost + Random Forest)
├── eff_model_no_rf.py          # Version without Random Forest
│
├── effex_model.py              # Full version (HGB + XGBoost + Random Forest)
├── effex_model_no_rf.py        # Version without Random Forest
│
├── README.md
├── .gitignore
├── LICENSE
```

## Description

### models/

Contains pre-trained surrogate models for both:

- Thermal efficiency (**Eff**)
- Exergetic efficiency (**EffEX**)

Separate models are provided for:

- Histogram Gradient Boosting (HGB)
- XGBoost

Random Forest models are not included in the repository due to large file size constraints.

---

### eff_model.py

Training, validation, and operating-point search script for **thermal efficiency (Eff)**.

This script:
- Trains Histogram Gradient Boosting and XGBoost models
- Reports train/test metrics
- Supports optional external validation
- Includes an interactive operating-point search tool

---

### effex_model.py

Training, validation, and operating-point search script for **exergetic efficiency (EffEX)**.

Functionality mirrors `eff_model.py`, but uses **EffEX** as the target variable.

---

### README.md

Project documentation describing:

- Objective and methodology  
- Model choices  
- Data structure  
- Operating-point search logic  
- Repository structure  

---

### .gitignore

Specifies files and directories excluded from version control, including:

- Local datasets  
- Temporary files  
- Python cache files  
- Large training artefacts  

---

### LICENSE

MIT License file defining usage permissions.

---

Pre-trained `.pkl` models are included for direct inference.

Users may retrain the models locally using their own generated datasets if required.

## Intended Use

This project demonstrates how machine learning can accelerate thermodynamic modelling workflows.

Suitable applications include:

- Engineering optimisation research  
- Surrogate modelling studies  
- Rapid parametric exploration  
- Machine learning in energy systems  

---

## License

This project is released under the MIT License.

	
