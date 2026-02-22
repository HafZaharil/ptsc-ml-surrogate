"""
Surrogate modelling for PTSC performance (Eff)

Trains and evaluates three regression models:
- Histogram-based Gradient Boosting (scikit-learn)
- XGBoost (if available)
- Random Forest (scikit-learn)

Reports:
1) Train/test split metrics on the training set
2) Metrics on an external validation file (optional)

Includes an interactive finder:
Given (DNI, Tamb, K), searches a Tin–Mhtf grid (Pressurehtf fixed) and prints:
- the global best predicted Eff
- the best predicted Eff for each Tin = 350..850 (step 50), including the Mhtf that achieves it
"""

import os
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

# ------------------------------------------------------------
# Optional dependency: XGBoost
# ------------------------------------------------------------
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception as e:
    print("XGBoost not available in this environment:", e)
    XGB_OK = False

# ------------------------------------------------------------
# Pandas display settings
# ------------------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------
TRAIN_PATH = "/Users/hafzaharil/Downloads/ML project/Data for training/generated_ml_dataset_surrogate_ready_corrected.csv"
VALIDATION_PATH = "/Users/hafzaharil/Downloads/ML project/Validation data/validation_1000_random_combinations.csv"

MODEL_PATH_HGB = "/Users/hafzaharil/Downloads/eff_model_hist_gradient_boosting_Eff.pkl"
MODEL_PATH_XGB = "/Users/hafzaharil/Downloads/eff_model_xgb_Eff.pkl"
MODEL_PATH_RF  = "/Users/hafzaharil/Downloads/eff_model_rf_Eff.pkl"

# ------------------------------------------------------------
# Search settings
# ------------------------------------------------------------
FIXED_PRESSUREHTF = 20000.0

FEATURE_COLS = ["Mhtf", "Pressurehtf", "Tin", "DNI", "Tamb", "K"]
TARGET_COL = "Eff"

# Global max search (coarse -> fine)
TIN_MIN, TIN_MAX = 350.0, 850.0
MHTF_MIN, MHTF_MAX = 0.50, 5.00

TIN_STEP_COARSE = 5.0
MHTF_STEP_COARSE = 0.10

TIN_WINDOW_FINE = 20.0
MHTF_WINDOW_FINE = 0.50
TIN_STEP_FINE = 0.2
MHTF_STEP_FINE = 0.02

# "Best per Tin" table
TIN_REPORT_MIN = 350.0
TIN_REPORT_MAX = 850.0
TIN_REPORT_STEP = 50.0
MHTF_REPORT_STEP = 0.02


# ============================================================
# Data loading + cleaning
# ============================================================
def load_csv_checked(path: str, require_target: bool = True) -> pd.DataFrame:
    if path is None:
        raise ValueError("CSV path is None.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = FEATURE_COLS + ([TARGET_COL] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    return df


def clean_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET_COL].replace([np.inf, -np.inf], np.nan)

    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask].copy(), y.loc[mask].copy()


# ============================================================
# Metrics
# ============================================================
def print_metrics(label: str, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    print(f"\n[{label}]")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)
    return mae, rmse, r2


def train_test_report(model, X, y, label_prefix: str, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(f"\n===== TRAIN/TEST SPLIT RESULTS: {label_prefix} =====")
    print_metrics(f"{label_prefix} - TRAIN", y_train, y_pred_train)
    print_metrics(f"{label_prefix} - TEST", y_test, y_pred_test)


def validate_metrics_only(model, validation_path: str, label: str):
    df_val = load_csv_checked(validation_path, require_target=True)

    X_val = df_val[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    y_val = df_val[TARGET_COL].replace([np.inf, -np.inf], np.nan)

    mask = X_val.notna().all(axis=1) & y_val.notna()
    X_val = X_val.loc[mask].copy()
    y_val = y_val.loc[mask].copy()

    t0 = time.time()
    y_pred = model.predict(X_val)
    pred_time = time.time() - t0

    print(f"\n===== VALIDATION RESULTS: {label} =====")
    print_metrics(f"{label} - VALIDATION", y_val, y_pred)
    print("Prediction time (s):", pred_time)


# ============================================================
# Model training
# ============================================================
def train_hist_gradient_boosting_model(df_train: pd.DataFrame):
    X, y = clean_xy(df_train)

    reg = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.05,
        max_iter=500,
        random_state=42,
    )

    t0 = time.time()
    reg.fit(X, y)
    print("\nHGB training time (s):", time.time() - t0)

    train_test_report(reg, X, y, "Histogram Gradient Boosting (scikit-learn)")
    return reg


def train_xgb_model(df_train: pd.DataFrame):
    if not XGB_OK:
        raise RuntimeError("XGBoost is not available in this environment.")

    X, y = clean_xy(df_train)

    reg = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.time()
    reg.fit(X, y)
    print("\nXGBoost training time (s):", time.time() - t0)

    train_test_report(reg, X, y, "XGBoost")
    return reg


def train_rf_model(df_train: pd.DataFrame):
    X, y = clean_xy(df_train)

    reg = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.time()
    reg.fit(X, y)
    print("\nRandom Forest training time (s):", time.time() - t0)

    train_test_report(reg, X, y, "Random Forest")
    return reg


# ============================================================
# Interactive search
# ============================================================
def c_to_k(temp_c: float) -> float:
    return float(temp_c) + 273.15


def ask_float(name: str, default=None) -> float:
    if default is None:
        return float(input(f"Enter {name}: ").strip())
    s = input(f"Enter {name} (default {default}): ").strip()
    return float(default) if s == "" else float(s)


def make_grid_predictions(model, dni, tamb_k, k_factor, tin_vals, mhtf_vals):
    Tin_grid, Mhtf_grid = np.meshgrid(tin_vals, mhtf_vals, indexing="xy")
    n = Tin_grid.size

    X = pd.DataFrame(
        {
            "Mhtf": Mhtf_grid.ravel(),
            "Pressurehtf": np.full(n, float(FIXED_PRESSUREHTF)),
            "Tin": Tin_grid.ravel(),
            "DNI": np.full(n, float(dni)),
            "Tamb": np.full(n, float(tamb_k)),
            "K": np.full(n, float(k_factor)),
        }
    )[FEATURE_COLS]

    y_pred = model.predict(X)

    out = X.copy()
    out["Predicted_Eff"] = y_pred
    return out


def find_best_for_ambient(model, dni, tamb_c, k_factor):
    tamb_k = c_to_k(tamb_c)

    tin_coarse = np.arange(TIN_MIN, TIN_MAX + 1e-9, TIN_STEP_COARSE)
    mhtf_coarse = np.arange(MHTF_MIN, MHTF_MAX + 1e-9, MHTF_STEP_COARSE)

    df_coarse = make_grid_predictions(model, dni, tamb_k, k_factor, tin_coarse, mhtf_coarse)
    best_coarse = df_coarse.loc[df_coarse["Predicted_Eff"].idxmax()]

    best_tin = float(best_coarse["Tin"])
    best_mhtf = float(best_coarse["Mhtf"])

    tin_fine_min = max(TIN_MIN, best_tin - TIN_WINDOW_FINE)
    tin_fine_max = min(TIN_MAX, best_tin + TIN_WINDOW_FINE)
    mhtf_fine_min = max(MHTF_MIN, best_mhtf - MHTF_WINDOW_FINE)
    mhtf_fine_max = min(MHTF_MAX, best_mhtf + MHTF_WINDOW_FINE)

    tin_fine = np.arange(tin_fine_min, tin_fine_max + 1e-9, TIN_STEP_FINE)
    mhtf_fine = np.arange(mhtf_fine_min, mhtf_fine_max + 1e-9, MHTF_STEP_FINE)

    df_fine = make_grid_predictions(model, dni, tamb_k, k_factor, tin_fine, mhtf_fine)
    best = df_fine.loc[df_fine["Predicted_Eff"].idxmax()].copy()

    return best, tamb_k


def best_for_each_tin(model, dni, tamb_k, k_factor):
    tin_targets = np.arange(TIN_REPORT_MIN, TIN_REPORT_MAX + 1e-9, TIN_REPORT_STEP)
    mhtf_vals = np.arange(MHTF_MIN, MHTF_MAX + 1e-9, MHTF_REPORT_STEP)

    rows = []
    for tin in tin_targets:
        X = pd.DataFrame(
            {
                "Mhtf": mhtf_vals,
                "Pressurehtf": np.full_like(mhtf_vals, float(FIXED_PRESSUREHTF), dtype=float),
                "Tin": np.full_like(mhtf_vals, float(tin), dtype=float),
                "DNI": np.full_like(mhtf_vals, float(dni), dtype=float),
                "Tamb": np.full_like(mhtf_vals, float(tamb_k), dtype=float),
                "K": np.full_like(mhtf_vals, float(k_factor), dtype=float),
            }
        )[FEATURE_COLS]

        y_pred = model.predict(X)
        best_idx = int(np.argmax(y_pred))

        rows.append(
            {
                "Tin": float(tin),
                "Mhtf": float(mhtf_vals[best_idx]),
                "Pressurehtf": float(FIXED_PRESSUREHTF),
                "DNI": float(dni),
                "Tamb": float(tamb_k),
                "K": float(k_factor),
                "Predicted_Eff": float(y_pred[best_idx]),
            }
        )

    return pd.DataFrame(rows)


def run_surrogate_max_finder(model, model_name: str):
    print("\n============================================================")
    print(f"SURROGATE MAX FINDER — {model_name}")
    print("Inputs: DNI, Tamb (°C), K")
    print(f"Pressurehtf is fixed to {FIXED_PRESSUREHTF} in the search.")
    print("============================================================")

    while True:
        dni = ask_float("DNI")
        tamb_c = ask_float("Tamb (°C)")
        k_factor = ask_float("K (0 to 1)")

        best, tamb_k = find_best_for_ambient(model, dni, tamb_c, k_factor)

        print("\n===== BEST (MAX PREDICTED EFF) =====")
        print(f"Predicted_Eff: {best['Predicted_Eff']:.6f}")
        print(f"Tin (K)      : {best['Tin']:.3f}")
        print(f"Mhtf (kg/s)  : {best['Mhtf']:.3f}")
        print(f"Pressurehtf  : {best['Pressurehtf']:.3f}")
        print(f"DNI          : {best['DNI']:.3f}")
        print(f"Tamb (°C)    : {tamb_c:.3f}")
        print(f"Tamb (K)     : {tamb_k:.3f}")
        print(f"K            : {best['K']:.3f}")

        print(f"\n===== BEST FOR EACH Tin ({int(TIN_REPORT_MIN)}–{int(TIN_REPORT_MAX)}, step {int(TIN_REPORT_STEP)}) =====")
        per_tin = best_for_each_tin(model, dni, tamb_k, k_factor)
        print(per_tin.to_string(index=False))

        ans = input("\nTry another set of ambient conditions? (Y/N): ").strip().upper()
        if ans != "Y":
            break


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    df_train = load_csv_checked(TRAIN_PATH, require_target=True)

    # Histogram Gradient Boosting
    if os.path.exists(MODEL_PATH_HGB):
        reg_hgb = joblib.load(MODEL_PATH_HGB)
        print(f"Loaded HGB model: {MODEL_PATH_HGB}")
        X_all, y_all = clean_xy(df_train)
        train_test_report(reg_hgb, X_all, y_all, "Histogram Gradient Boosting (scikit-learn)")
    else:
        reg_hgb = train_hist_gradient_boosting_model(df_train)
        joblib.dump(reg_hgb, MODEL_PATH_HGB)
        print(f"Saved HGB model: {MODEL_PATH_HGB}")

    # XGBoost
    reg_xgb = None
    if XGB_OK:
        if os.path.exists(MODEL_PATH_XGB):
            reg_xgb = joblib.load(MODEL_PATH_XGB)
            print(f"\nLoaded XGBoost model: {MODEL_PATH_XGB}")
            X_all, y_all = clean_xy(df_train)
            train_test_report(reg_xgb, X_all, y_all, "XGBoost")
        else:
            reg_xgb = train_xgb_model(df_train)
            joblib.dump(reg_xgb, MODEL_PATH_XGB)
            print(f"Saved XGBoost model: {MODEL_PATH_XGB}")
    else:
        print("\nXGBoost not available; skipping XGBoost training.")

    # Random Forest
    if os.path.exists(MODEL_PATH_RF):
        reg_rf = joblib.load(MODEL_PATH_RF)
        print(f"\nLoaded Random Forest model: {MODEL_PATH_RF}")
        X_all, y_all = clean_xy(df_train)
        train_test_report(reg_rf, X_all, y_all, "Random Forest")
    else:
        reg_rf = train_rf_model(df_train)
        joblib.dump(reg_rf, MODEL_PATH_RF)
        print(f"Saved Random Forest model: {MODEL_PATH_RF}")

    # External validation (metrics only)
    if VALIDATION_PATH is not None and os.path.exists(VALIDATION_PATH):
        validate_metrics_only(reg_hgb, VALIDATION_PATH, "Histogram Gradient Boosting (scikit-learn)")
        if reg_xgb is not None:
            validate_metrics_only(reg_xgb, VALIDATION_PATH, "XGBoost")
        validate_metrics_only(reg_rf, VALIDATION_PATH, "Random Forest")
    else:
        print("\nNo validation file found (or VALIDATION_PATH is None). Skipping validation.")

    # Interactive search
    run_surrogate_max_finder(reg_hgb, "Histogram Gradient Boosting (scikit-learn)")
    if reg_xgb is not None:
        run_surrogate_max_finder(reg_xgb, "XGBoost")
    run_surrogate_max_finder(reg_rf, "Random Forest")