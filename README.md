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
- **K** — Incident angle factor from dataset  

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
- Incidence angle factor (K)  

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

With **Pressurehtf fixed** (default 20,000 kPa), the script:

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

## Results and Discussion

### Thermal efficiency - Train / Test Results

| Model                        | Dataset | MAE    | RMSE   | R²     |
|-----------------------------|--------|--------|--------|--------|
| Histogram Gradient Boosting | Train  | 0.7915 | 1.5877 | 0.9985 |
| Histogram Gradient Boosting | Test   | 0.8110 | 1.7520 | 0.9982 |
| XGBoost                     | Train  | 0.3677 | 0.7566 | 0.9997 |
| XGBoost                     | Test   | 0.4400 | 1.3115 | 0.9990 |
| Random Forest               | Train  | 0.1761 | 0.6690 | 0.9997 |
| Random Forest               | Test   | 0.4779 | 1.8152 | 0.9980 |

### Thermal efficiency - Additional Validation Results (1000 random combinations)
| Model                        | MAE    | RMSE   | R²     |
|-----------------------------|--------|--------|--------|
| Histogram Gradient Boosting | 3.6552 | 6.6540 | 0.9550 |
| XGBoost                     | 3.5139 | 6.3267 | 0.9593 |
| Random Forest               | 3.3152 | 6.3938 | 0.9584 |


### Exergetic efficiency (EffEx) - Train / Test Results

| Model                        | Dataset | MAE    | RMSE   | R²     |
|-----------------------------|--------|--------|--------|--------|
| Histogram Gradient Boosting | Train  | 0.7915 | 1.5877 | 0.9985 |
| Histogram Gradient Boosting | Test   | 0.8110 | 1.7520 | 0.9982 |
| XGBoost                     | Train  | 0.3677 | 0.7566 | 0.9997 |
| XGBoost                     | Test   | 0.4400 | 1.3115 | 0.9990 |
| Random Forest               | Train  | 0.1761 | 0.6690 | 0.9997 |
| Random Forest               | Test   | 0.4779 | 1.8152 | 0.9980 |

### Exergetic efficiency - Additional Validation Results (1000 random combinations)
| Model                        | MAE    | RMSE   | R²     |
|-----------------------------|--------|--------|--------|
| Histogram Gradient Boosting | 3.6552 | 6.6540 | 0.9550 |
| XGBoost                     | 3.5139 | 6.3267 | 0.9593 |
| Random Forest               | 3.3152 | 6.3938 | 0.9584 |


### Results and Discussion

All three models demonstrate extremely high accuracy on the training and test datasets, with R² values exceeding 0.998. This indicates that the surrogate models successfully capture the underlying relationship between operating parameters and PTSC efficiency.

However, the more meaningful assessment comes from the external validation dataset, which consists of independently generated operating conditions. Here, performance decreases to R² values in the range of 0.955–0.959. This drop is expected and confirms that the models are generalising rather than simply memorising the training data.

Across all models, XGBoost shows the most consistent performance. It achieves the highest validation R² and the lowest RMSE, indicating better stability when predicting unseen conditions. Random Forest achieves the lowest MAE, suggesting strong point-wise accuracy, but its significantly lower training error compared to test and validation results indicates mild overfitting. Histogram Gradient Boosting performs reliably, though slightly below the other two models in all evaluation metrics.

A key observation is that while Random Forest fits the training data extremely closely, its generalisation ability is weaker than XGBoost. In contrast, XGBoost maintains a better balance between fitting accuracy and robustness, making it more suitable for predictive applications. In addition, random forest training file is significanly larger than xgboost and the computation time took longer, without tangible imporvemetns on accuracy.


Overall, all models are valid surrogate representations of the PTSC system. However, XGBoost is the most reliable choice for deployment, particularly when predicting performance under new or unseen operating conditions. Hence, XgBoost was chosen for the surrogate model calculator at the end of the  code.

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

	
