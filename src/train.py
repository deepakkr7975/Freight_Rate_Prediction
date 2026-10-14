"""
Freight Rate Prediction — Production Training Pipeline
======================================================
Author: Deepak
Assessment: Spotter AI ML Assessment (Freight Rate Prediction)

Key Innovations & Modeling Strategy:
  1. Rate-per-Mile Transformation:
     Distance accounts for >90% of raw freight rate variance. Handing
     rate-per-mile (posted_rate / distance) to the model allows gradient
     boosted trees to isolate pure market, equipment, load density, and
     geographic routing effects without distance scaling artifacts.
  2. Targeted Winsorization (0.5% / 99.5%):
     EDA demonstrates that ~1.4% of rows suffer from uniformly distributed
     data entry/transmission corruption (outside $1.00-$4.00/mile). Winsorizing
     prevents tree splits from overfitting to severe synthetic noise.
  3. Realistic Rolling-Origin Cross-Validation:
     Respects temporal precedence by training on expanding historical windows
     and predicting subsequent forward periods (July, August, September, October).
  4. Multi-Source December Feature Recovery:
     December inputs lack coordinates, market_index, and quote_signal. Coordinates
     are recovered via historical city centroid lookups, and December market signals
     are joined from concurrent validation.csv records.
  5. High-Performance Gradient Boosted Ensemble:
     Blends scikit-learn's HistGradientBoostingRegressor and LightGBM (MAE objective)
     with regularized shallow trees (31 leaves) to guarantee out-of-sample generalization.
"""

from __future__ import annotations

import os
import sys
import ctypes
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Preload libomp on macOS if present to allow seamless LightGBM execution
if sys.platform == "darwin":
    for libomp_path in [
        "/opt/homebrew/opt/libomp/lib/libomp.dylib",
        "/usr/local/opt/libomp/lib/libomp.dylib",
    ]:
        if os.path.exists(libomp_path):
            try:
                ctypes.CDLL(libomp_path)
                break
            except Exception:
                pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb


# ── Configuration & Aesthetic Constants ────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TEAL = "#064A56"
CORAL = "#E07A5F"
SLATE = "#3D405B"
EARTH_R_MI = 3958.8


# ══════════════════════════════════════════════════════════════════════════
#  1. Helper Functions: Cleaning, Coordinates & Haversine Distance
# ══════════════════════════════════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans dataframe: fixes sign-flipped weights and imputes missing values."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["weight"] = out["weight"].abs()
    out["weight"] = out["weight"].fillna(out["weight"].median())

    if "market_index" in out.columns:
        daily_median = out.groupby(out["date"].dt.normalize())["market_index"].transform("median")
        out["market_index"] = out["market_index"].fillna(daily_median)
        out["market_index"] = out["market_index"].fillna(out["market_index"].median())

    if "quote_signal" in out.columns:
        out["quote_signal"] = out["quote_signal"].fillna(out["quote_signal"].median())

    return out


def make_city_coords(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Builds a deterministic lookup table for city coordinates (lat, lon)."""
    p = df[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
    )
    d = df[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
    )
    combined = pd.concat([p, d]).drop_duplicates(subset=["city"]).dropna()
    return {row.city: (row.lat, row.lon) for row in combined.itertuples()}


def haversine(lat1, lon1, lat2, lon2):
    """Calculates great-circle distance between two coordinate pairs in miles."""
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R_MI * np.arcsin(np.sqrt(a))


# ══════════════════════════════════════════════════════════════════════════
#  2. Feature Engineering
# ══════════════════════════════════════════════════════════════════════════
EQUIPMENT_TYPES = ["Dry Van", "Reefer", "Flatbed"]


def build_features(df: pd.DataFrame, coords: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """
    Constructs feature matrix with distance dynamics, geography, equipment, and market signals.
    """
    X = pd.DataFrame(index=df.index)

    # 1. Distance & Load Dynamics
    X["distance"] = df["distance"].values
    X["log_distance"] = np.log1p(df["distance"].values)
    X["weight"] = df["weight"].values
    X["weight_per_mile"] = df["weight"].values / np.clip(df["distance"].values, 1.0, None)

    # 2. Equipment Encoding
    for eq in EQUIPMENT_TYPES:
        col_name = f"eq_{eq.replace(' ', '_')}"
        X[col_name] = (df["equipment"] == eq).astype(int).values

    # 3. Geography & Routing
    if {"pickup_lat", "pickup_lon"}.issubset(df.columns):
        plat, plon = df["pickup_lat"].values, df["pickup_lon"].values
        dlat, dlon = df["delivery_lat"].values, df["delivery_lon"].values
    else:
        plat = df["pickup"].map(lambda c: coords[c][0]).values
        plon = df["pickup"].map(lambda c: coords[c][1]).values
        dlat = df["delivery"].map(lambda c: coords[c][0]).values
        dlon = df["delivery"].map(lambda c: coords[c][1]).values

    X["pickup_lat"] = plat
    X["pickup_lon"] = plon
    X["delivery_lat"] = dlat
    X["delivery_lon"] = dlon

    hav = haversine(plat, plon, dlat, dlon)
    X["haversine"] = hav
    X["circuity"] = df["distance"].values / np.clip(hav, 1.0, None)
    X["dlat"] = dlat - plat
    X["dlon"] = dlon - plon

    # 4. Market State
    X["market_index"] = df["market_index"].values
    if "quote_signal" in df.columns:
        X["quote_signal"] = df["quote_signal"].values
    else:
        X["quote_signal"] = 0.0

    # 5. Temporal Features
    date_series = pd.to_datetime(df["date"])
    X["dow"] = date_series.dt.dayofweek.values
    X["is_weekend"] = (date_series.dt.dayofweek >= 5).astype(int).values
    X["day_of_month"] = date_series.dt.day.values
    X["month"] = date_series.dt.month.values

    return X


# ══════════════════════════════════════════════════════════════════════════
#  3. Main Execution & Pipeline
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 76)
    print(" SPOTTER AI FREIGHT RATE MODEL TRAINING PIPELINE")
    print("=" * 76)

    # ── Step 1: Load Data ──
    print("\n[Step 1] Loading Datasets...")
    train_path = BASE_DIR / "train-test.csv"
    val_path = BASE_DIR / "validation.csv"
    dec_path = BASE_DIR / "december-chart-inputs.csv"
    tmpl_path = BASE_DIR / "validation-predictions-template.csv"

    train_raw = pd.read_csv(train_path, parse_dates=["date"])
    val_raw = pd.read_csv(val_path, parse_dates=["date"])
    dec_raw = pd.read_csv(dec_path, parse_dates=["date"])
    template = pd.read_csv(tmpl_path)

    print(f"  Training Set:   {train_raw.shape[0]:>6,} rows × {train_raw.shape[1]} columns")
    print(f"  Validation Set: {val_raw.shape[0]:>6,} rows × {val_raw.shape[1]} columns")
    print(f"  December Inputs: {dec_raw.shape[0]:>6} rows × {dec_raw.shape[1]} columns")

    # ── Step 2: Clean & Preprocess ──
    print("\n[Step 2] Cleaning & Preprocessing...")
    train_c = clean_data(train_raw)
    val_c = clean_data(val_raw)

    coords = make_city_coords(train_raw)
    print(f"  Extracted coordinates for {len(coords)} unique operating hubs.")

    # Calculate Rate-Per-Mile (RPM)
    train_c["rpm"] = train_c["posted_rate"] / train_c["distance"]
    rpm_all = train_c["rpm"].values
    dist_all = train_c["distance"].values
    posted_all = train_c["posted_rate"].values

    # Target Winsorization
    lo_rpm, hi_rpm = np.quantile(rpm_all, [0.005, 0.995])
    y_train_winsor = np.clip(rpm_all, lo_rpm, hi_rpm)
    print(f"  Winsorized Target: [{lo_rpm:.3f}, {hi_rpm:.3f}] $/mile")
    corrupt_count = ((rpm_all < lo_rpm) | (rpm_all > hi_rpm)).sum()
    print(f"  Suppressed corrupt load impact: {corrupt_count:,} records ({corrupt_count/len(rpm_all)*100:.2f}%)")

    # ── Step 3: Feature Construction ──
    print("\n[Step 3] Constructing Feature Matrices...")
    X_train = build_features(train_c, coords)
    feature_names = list(X_train.columns)
    print(f"  Generated {len(feature_names)} features: {feature_names}")

    # ── Step 4: Rolling-Origin Cross-Validation ──
    print("\n[Step 4] Evaluating with Rolling-Origin Cross-Validation...")
    print("  Evaluating out-of-sample forward projections across expanding temporal horizons:")

    folds = [
        ("2025-07-01", "2025-08-01", "July"),
        ("2025-08-01", "2025-09-01", "August"),
        ("2025-09-01", "2025-10-01", "September"),
        ("2025-10-01", "2025-11-01", "October"),
    ]

    hist_params = {
        "max_iter": 800,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.12,
        "n_iter_no_change": 30,
        "random_state": RANDOM_STATE,
    }

    cv_results = []
    hist_test_preds = []
    lgb_test_preds = []
    y_test_list = []
    clean_masks = []

    for start_str, end_str, month_name in folds:
        s = pd.Timestamp(start_str)
        e = pd.Timestamp(end_str)

        tr_idx = (train_c["date"] < s).values
        te_idx = ((train_c["date"] >= s) & (train_c["date"] < e)).values

        X_tr, y_tr_raw = X_train.iloc[tr_idx], rpm_all[tr_idx]
        X_te, d_te = X_train.iloc[te_idx], dist_all[te_idx]
        y_te_true = posted_all[te_idx]
        te_rpm = rpm_all[te_idx]

        # Fit Winsorizer strictly on fold training data
        fold_lo, fold_hi = np.quantile(y_tr_raw, [0.005, 0.995])
        y_tr_win = np.clip(y_tr_raw, fold_lo, fold_hi)

        # HistGBM Model
        h_model = HistGradientBoostingRegressor(**hist_params)
        h_model.fit(X_tr, y_tr_win)
        pred_h_rate = np.clip(h_model.predict(X_te) * d_te, 1.0, None)

        # LightGBM Model
        lgb_tr = lgb.Dataset(X_tr, label=y_tr_win)
        lgb_params = {
            "objective": "regression_l1",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 40,
            "reg_lambda": 1.0,
            "verbosity": -1,
            "seed": RANDOM_STATE,
        }
        lgb_m = lgb.train(lgb_params, lgb_tr, num_boost_round=h_model.n_iter_)
        pred_l_rate = np.clip(lgb_m.predict(X_te) * d_te, 1.0, None)

        # Ensemble blend (50/50 initial)
        pred_ens_rate = 0.5 * pred_h_rate + 0.5 * pred_l_rate

        # Mask for clean loads (legitimate economics: $1.00 - $4.00/mile)
        is_clean = (te_rpm > 1.0) & (te_rpm < 4.0)

        mae_all = mean_absolute_error(y_te_true, pred_ens_rate)
        rmse_all = np.sqrt(mean_squared_error(y_te_true, pred_ens_rate))
        mape_all = np.mean(np.abs((y_te_true - pred_ens_rate) / y_te_true)) * 100
        r2_all = r2_score(y_te_true, pred_ens_rate)

        mae_clean = mean_absolute_error(y_te_true[is_clean], pred_ens_rate[is_clean])
        rmse_clean = np.sqrt(mean_squared_error(y_te_true[is_clean], pred_ens_rate[is_clean]))
        mape_clean = np.mean(np.abs((y_te_true[is_clean] - pred_ens_rate[is_clean]) / y_te_true[is_clean])) * 100
        r2_clean = r2_score(y_te_true[is_clean], pred_ens_rate[is_clean])

        cv_results.append({
            "Fold": month_name,
            "Train_N": tr_idx.sum(),
            "Test_N": te_idx.sum(),
            "MAE_All": mae_all,
            "RMSE_All": rmse_all,
            "MAPE_All": mape_all,
            "R2_All": r2_all,
            "MAE_Clean": mae_clean,
            "RMSE_Clean": rmse_clean,
            "MAPE_Clean": mape_clean,
            "R2_Clean": r2_clean,
        })

        hist_test_preds.append(pred_h_rate)
        lgb_test_preds.append(pred_l_rate)
        y_test_list.append(y_te_true)
        clean_masks.append(is_clean)

        print(f"  Fold {month_name:<9} | All Loads: MAE=${mae_all:>6.2f}, MAPE={mape_all:>5.2f}%, R²={r2_all:.4f} | "
              f"Clean: MAE=${mae_clean:>5.2f}, R²={r2_clean:.4f}")

    cv_df = pd.DataFrame(cv_results)
    avg_mae_all = cv_df["MAE_All"].mean()
    avg_mae_clean = cv_df["MAE_Clean"].mean()
    avg_r2_clean = cv_df["R2_Clean"].mean()
    avg_mape_all = cv_df["MAPE_All"].mean()

    print("\n  Summary Across All 4 Rolling Folds:")
    print(f"    • Mean MAE (All Loads):   ${avg_mae_all:.2f}")
    print(f"    • Mean MAE (Clean Loads): ${avg_mae_clean:.2f}")
    print(f"    • Mean R² (Clean Loads):  {avg_r2_clean:.4f}")
    print(f"    • Mean MAPE (All Loads):  {avg_mape_all:.2f}%")

    # ── Step 5: Final Model Retraining on Full Dataset ──
    print("\n[Step 5] Retraining Production Ensemble on 100% of Labeled Data...")
    final_hist = HistGradientBoostingRegressor(**hist_params)
    final_hist.fit(X_train, y_train_winsor)
    print(f"  HistGBM converged after {final_hist.n_iter_} iterations.")

    lgb_full_ds = lgb.Dataset(X_train, label=y_train_winsor)
    lgb_full_params = {
        "objective": "regression_l1",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 40,
        "reg_lambda": 1.0,
        "verbosity": -1,
        "seed": RANDOM_STATE,
    }
    final_lgb = lgb.train(lgb_full_params, lgb_full_ds, num_boost_round=final_hist.n_iter_)
    print(f"  LightGBM trained for {final_hist.n_iter_} rounds.")

    # ── Step 6: Generate Validation Predictions ──
    print("\n[Step 6] Generating Predictions for validation.csv...")
    X_val = build_features(val_c, coords)
    X_val = X_val[feature_names]

    pred_h_val = final_hist.predict(X_val) * val_c["distance"].values
    pred_l_val = final_lgb.predict(X_val) * val_c["distance"].values
    val_pred = 0.5 * pred_h_val + 0.5 * pred_l_val
    val_pred = np.clip(val_pred, 1.0, None)

    sub = template.copy()
    sub["predicted_rate"] = np.round(val_pred, 2)
    val_out_path = BASE_DIR / "validation_predictions.csv"
    sub.to_csv(val_out_path, index=False)
    print(f"  ✓ Saved {len(sub):,} predictions to: {val_out_path.name}")
    print(f"    Summary: Min=${val_pred.min():.2f}, Mean=${val_pred.mean():.2f}, "
          f"Median=${np.median(val_pred):.2f}, Max=${val_pred.max():.2f}")

    # ── Step 7: Recover December Signals & Predict December Inputs ──
    print("\n[Step 7] Recovering December Market Features & Predicting Benchmark...")
    dec_full = dec_raw.copy()
    dec_full["date"] = pd.to_datetime(dec_full["date"])
    dec_full["weight"] = dec_full["weight"].abs()

    # Join daily market_index and quote_signal from validation dataset
    val_daily = (
        val_c.groupby(val_c["date"].dt.normalize())
        .agg(market_index=("market_index", "mean"), quote_signal=("quote_signal", "mean"))
    )
    dec_key = dec_full["date"].dt.normalize()
    dec_full["market_index"] = dec_key.map(val_daily["market_index"])
    dec_full["quote_signal"] = dec_key.map(val_daily["quote_signal"])

    assert dec_full[["market_index", "quote_signal"]].isna().sum().sum() == 0, "Missing December signals"
    print("  ✓ Perfectly recovered December daily market signals from validation timeline.")

    X_dec = build_features(dec_full, coords)
    X_dec = X_dec[feature_names]

    dec_pred_h = final_hist.predict(X_dec) * dec_full["distance"].values
    dec_pred_l = final_lgb.predict(X_dec) * dec_full["distance"].values
    dec_pred = 0.5 * dec_pred_h + 0.5 * dec_pred_l
    dec_pred = np.clip(dec_pred, 1.0, None)

    dec_out = dec_raw.copy()
    dec_out["date"] = pd.to_datetime(dec_out["date"]).dt.strftime("%Y-%m-%d")
    dec_out["predicted_rate"] = np.round(dec_pred, 2)
    # Ensure exact 7 columns
    dec_out = dec_out[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]]
    dec_out_path = BASE_DIR / "december-chart-inputs.csv"
    dec_out.to_csv(dec_out_path, index=False)

    print(f"  ✓ Saved 31 December daily predictions to: {dec_out_path.name}")
    print(f"    Trajectory: Range ${dec_pred.min():.2f} – ${dec_pred.max():.2f} "
          f"(Mean: ${dec_pred.mean():.2f}, ${(dec_pred.mean()/360):.2f}/mile)")

    # ── Step 8: Generate Report Visualizations ──
    print("\n[Step 8] Generating Assessment Figures...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fold comparison bar plot
    fold_names = cv_df["Fold"].tolist()
    x = np.arange(len(fold_names))
    width = 0.35
    axes[0].bar(x - width/2, cv_df["MAE_All"], width, label="All Loads (w/ Noise)", color=CORAL, alpha=0.85)
    axes[0].bar(x + width/2, cv_df["MAE_Clean"], width, label="Clean Loads Only", color=TEAL, alpha=0.85)
    axes[0].set_ylabel("MAE ($)")
    axes[0].set_title("Rolling-Origin Cross-Validation MAE Across Test Months", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(fold_names)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # December predicted rate series
    axes[1].plot(pd.to_datetime(dec_out["date"]), dec_pred, color=TEAL, linewidth=2.5, marker="o", markersize=4)
    axes[1].set_ylabel("Predicted Rate ($)")
    axes[1].set_title("December 2025 Daily Forecast (Lexington → Fort Wayne)", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "07_model_results.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved reports/07_model_results.png")

    # Feature Importance Plot
    if hasattr(final_hist, "_loss"):
        # Permutation or LightGBM split importance
        lgb_imp = final_lgb.feature_importance(importance_type="gain")
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": lgb_imp}).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(imp_df["Feature"], imp_df["Importance"], color=TEAL, alpha=0.85)
        ax.set_xlabel("Feature Importance (Split Gain)")
        ax.set_title("Key Pricing Drivers (Gradient Boosted Feature Importance)", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "08_feature_importance.png", dpi=180, bbox_inches="tight")
        plt.close()
        print("  ✓ Saved reports/08_feature_importance.png")

    print("\n" + "=" * 76)
    print(" TRAINING & INFERENCE COMPLETE! READY FOR SCORER EVALUATION.")
    print("=" * 76)


if __name__ == "__main__":
    main()
