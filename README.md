PTSC Surrogate Modelling (Eff / EffEX)

Fast surrogate models for parabolic trough solar collector (PTSC) performance, trained on simulation-generated data.

The goal is to replace slow physics-based simulations with near-instant predictions and to support rapid operating-point searches.

This repository trains and compares tree-based regression models and provides an interactive “max finder” that searches for high-efficiency operating points under given ambient conditions.

⸻

What this project does

Given the following inputs:
	•	Mhtf (mass flow rate)
	•	Pressurehtf (HTF pressure)
	•	Tin (inlet temperature)
	•	DNI (direct normal irradiance)
	•	Tamb (ambient temperature)
	•	K (model factor from the dataset)

The model learns a mapping to either:
	•	Eff (thermal efficiency)
	•	EffEX (exergetic efficiency)

Conceptually:

Input:  (Mhtf, Pressurehtf, Tin, DNI, Tamb, K)
Output: Predicted Eff / EffEX

Once trained, the surrogate can be used for:
	1.	Prediction
Quickly estimate Eff or EffEX for new operating conditions.
	2.	Search
For given ambient conditions (DNI, Tamb, K) and fixed Pressurehtf:
	•	Find the global best predicted efficiency on a Tin–Mhtf grid.
	•	Find the best predicted efficiency for each Tin from 350 K to 850 K (step 50 K), including the Mhtf that achieves it.

⸻

Why surrogate modelling?

Detailed PTSC simulations based on physics models can become slow when running large parameter sweeps.

A surrogate model replaces repeated physics runs with a learned mapping:

(Mhtf, Pressurehtf, Tin, DNI, Tamb, K)
↓
Predicted Eff or EffEX

After training, predictions are extremely fast and suitable for:
	•	Dense grid searches
	•	Sensitivity checks
	•	Rapid operating-point exploration
	•	Decision-support workflows

⸻

Methodology

1) Data

The training and validation datasets are generated from a physics-based simulation pipeline.

Expected CSV columns:

Features:
	•	Mhtf
	•	Pressurehtf
	•	Tin
	•	DNI
	•	Tamb
	•	K

Target:
	•	Eff  (for Eff script)
	•	EffEX (for EffEX script)

Basic cleaning steps in code:
	•	Strip column names
	•	Replace inf and -inf with NaN
	•	Drop rows with missing values in features or target

⸻

2) Train/test split (sanity check)

The training CSV is split into train/test (default 80/20) to check:
	•	Whether the model is learning meaningful structure
	•	Whether obvious overfitting or numerical instability appears

Metrics reported:
	•	MAE
	•	RMSE
	•	R²

This is a quick internal check and not a substitute for external validation.

⸻

3) External validation

If a separate validation CSV is provided, the same metrics are reported on that file.

External validation performance should be used to determine the final model choice.

⸻

4) Grid-based operating-point search (“max finder”)

For a selected model:

User inputs:
	•	DNI
	•	Tamb (°C)
	•	K

The script:
	•	Converts Tamb to Kelvin
	•	Fixes Pressurehtf to a constant (default 20000)
	•	Searches over Tin and Mhtf

Search strategy:
	•	Coarse grid over the full domain
	•	Fine grid around the best coarse point

The script prints:
	1.	The global best predicted operating point
	2.	The best predicted efficiency for each Tin from 350 K to 850 K (step 50 K), including the mass flow rate that achieves it

This turns the surrogate into a practical optimisation tool.

⸻

Models Used

Tree-based regression models are chosen because PTSC behaviour is:
	•	Non-linear
	•	Feature-interactive
	•	Structured tabular data

Histogram-Based Gradient Boosting (scikit-learn)
	•	Strong baseline
	•	Good accuracy
	•	Fast inference
	•	No external dependency

XGBoost
	•	Often achieves high accuracy on tabular data
	•	Robust boosting implementation
	•	Optional dependency

Random Forest (if included)
	•	Stable baseline
	•	Lower sensitivity to tuning
	•	Useful cross-check against boosting methods

Model selection should be based primarily on external validation metrics.

⸻

Notes on model files

Pre-trained .pkl model files are not uploaded due to large file size.

Users can:
	•	Train models locally using their own generated datasets
	•	Or adapt the scripts to their own PTSC datasets
