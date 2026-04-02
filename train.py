"""
DeepScaler Local Training — Extracts the pipeline from modelling_pipeline.ipynb
and trains LightGBM + CatBoost locally (CPU-only, no FT-Transformer).

Saves model artifacts + scaler + feature list + sample jobs to models/ directory.

Usage:  python train.py
"""

import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR = Path(__file__).parent / "Data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

ALPHA = 10.0
TARGET_COL = "peak_cpu_utilization"
HIST_COL = "cpu_histogram_seq"

BOOSTING_META_COLS = [
    "requested_cpus", "requested_memory", "priority", "scheduling_class",
    "alloc_collection_id", "collection_type", "parent_collection_id",
]
BOOSTING_USER_COLS = [
    "user_job_count", "user_avg_cpu_util", "user_std_cpu_util",
    "user_max_cpu_util", "user_p90_cpu_util",
    "user_avg_priority", "user_overprovision_rate",
    "user_c0_job_count", "user_c0_avg_cpu", "user_c0_std_cpu",
    "user_c1_job_count", "user_c1_avg_cpu", "user_c1_std_cpu",
    "user_c1_max_cpu", "user_c1_p90_cpu",
    "user_c1_avg_high_mass", "user_c1_avg_max_bkt",
]
SCALAR_TS_COLS = ["ts_avg_cpu_seq", "ts_max_cpu_seq", "ts_avg_mem_seq", "ts_max_mem_seq"]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (extracted from notebook)
# ═══════════════════════════════════════════════════════════════════════════

def asymmetric_rmse(y_true, y_pred, alpha=ALPHA):
    r = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(np.where(r > 0, alpha, 1.0) * r ** 2)))


def parse_json_seq(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            p = json.loads(val)
            return p if isinstance(p, list) else None
        except Exception:
            return None
    return None


def _decode_hist_element(h):
    def _inner(d):
        if isinstance(d, list):
            try:
                return [float(x) for x in d]
            except Exception:
                return None
        if isinstance(d, dict):
            for v in d.values():
                if isinstance(v, list):
                    try:
                        return [float(x) for x in v]
                    except Exception:
                        continue
            try:
                return [float(d[k]) for k in sorted(d.keys())]
            except Exception:
                return None
        return None

    if isinstance(h, (list, dict)):
        return _inner(h)
    if isinstance(h, str):
        try:
            return _inner(json.loads(h))
        except Exception:
            return None
    return None


def pad_hist_seq(seq, n_windows, hist_len):
    zero = [0.0] * hist_len
    if not isinstance(seq, list) or not seq:
        return [zero[:] for _ in range(n_windows)]
    normed = []
    for h in seq[:n_windows]:
        d = _decode_hist_element(h)
        if d:
            if len(d) < hist_len:
                d = d + [0.0] * (hist_len - len(d))
            normed.append(d[:hist_len])
        else:
            normed.append(zero[:])
    while len(normed) < n_windows:
        normed.append(normed[-1][:] if normed else zero[:])
    return normed


def _detect_hist_len(series):
    for val in series:
        seq = parse_json_seq(val) if isinstance(val, str) else val
        if isinstance(seq, list) and seq:
            d = _decode_hist_element(seq[0])
            if d:
                return len(d)
    return None


def parse_histogram_column(df, col):
    if col not in df.columns:
        return df, 0
    print(f"  Parsing '{col}'...")
    raw = df[col].apply(parse_json_seq)
    hist_len = _detect_hist_len(raw[raw.notna()])
    if hist_len is None:
        return df, 0
    print(f"  Histogram buckets: {hist_len}")
    df = df.copy()
    df[col] = raw.apply(lambda s: pad_hist_seq(s, 3, hist_len))
    return df, hist_len


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (from notebook)
# ═══════════════════════════════════════════════════════════════════════════

def load_and_merge_data(data_dir):
    def read_pattern(pattern):
        files = sorted(data_dir.glob(f"*{pattern}*.parquet"))
        if not files:
            print(f"  WARNING: no files matched *{pattern}*")
            return None
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        id_cols = [c for c in ["collection_id", "instance_index"] if c in df.columns]
        if id_cols:
            before = len(df)
            df = df.drop_duplicates(subset=id_cols, keep="last").reset_index(drop=True)
            print(f"  '{pattern}': {len(files)} file(s), {len(df):,} rows (was {before:,})")
        return df

    targets = read_pattern("targets")
    ts_feat = read_pattern("ts_features")
    col_evt = read_pattern("collection_events")

    if targets is None:
        raise FileNotFoundError("No targets parquet files found.")

    if ts_feat is not None:
        keys = [k for k in ["collection_id", "instance_index"]
                if k in targets.columns and k in ts_feat.columns]
        if keys:
            df = targets.merge(ts_feat, on=keys, how="inner", suffixes=("", "_ts"))
            df.drop(columns=[c for c in df.columns if c.endswith("_ts")], inplace=True, errors="ignore")
            print(f"  After join ts_features: {len(df):,} rows")
        else:
            df = targets
    else:
        df = targets

    if col_evt is not None and "collection_id" in df.columns and "collection_id" in col_evt.columns:
        evt = col_evt.drop_duplicates("collection_id", keep="last")
        df = df.merge(evt, on="collection_id", how="left", suffixes=("", "_meta"))
        df.drop(columns=[c for c in df.columns if c.endswith("_meta")], inplace=True, errors="ignore")
        print(f"  After join collection_events: {len(df):,} rows")

    if TARGET_COL not in df.columns:
        avail = [c for c in df.columns if "target" in c.lower() or "cpu" in c.lower()]
        raise ValueError(f"Target column '{TARGET_COL}' not found. Available: {avail[:10]}")

    for col in SCALAR_TS_COLS:
        if col in df.columns:
            df[col] = df[col].apply(parse_json_seq)

    before = len(df)
    valid = df[TARGET_COL].notna() & np.isfinite(df[TARGET_COL].astype(float))
    df = df[valid].copy().reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df):,} invalid target rows")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Train/val/test split (job-level, stratified)
# ═══════════════════════════════════════════════════════════════════════════

def split_by_jobs(df, val_size=0.15, test_size=0.15, random_state=42, strat_col="scheduling_class"):
    rng = np.random.default_rng(random_state)
    train_jobs, val_jobs, test_jobs = set(), set(), set()
    if strat_col in df.columns:
        job_meta = df[["collection_id", strat_col]].drop_duplicates("collection_id").set_index("collection_id")
        classes = job_meta[strat_col].unique()
    else:
        job_meta = None
        classes = ["__all__"]
    for cls in sorted(classes):
        cj = job_meta[job_meta[strat_col] == cls].index.values if job_meta is not None else df["collection_id"].unique()
        s = rng.permutation(cj)
        n_tr = int(len(s) * (1 - val_size - test_size))
        n_va = int(len(s) * val_size)
        train_jobs.update(s[:n_tr])
        val_jobs.update(s[n_tr:n_tr + n_va])
        test_jobs.update(s[n_tr + n_va:])
    df_tr = df[df["collection_id"].isin(train_jobs)].copy()
    df_va = df[df["collection_id"].isin(val_jobs)].copy()
    df_te = df[df["collection_id"].isin(test_jobs)].copy()
    print(f"  Split: train={len(df_tr):,}  val={len(df_va):,}  test={len(df_te):,}")
    return df_tr, df_va, df_te


# ═══════════════════════════════════════════════════════════════════════════
# User features (computed from training labels only — no leakage)
# ═══════════════════════════════════════════════════════════════════════════

def add_user_features(df_train, df_val, df_test):
    user_col = next((c for c in ["user", "user_id"] if c in df_train.columns), None)
    if user_col is None:
        return df_train, df_val, df_test

    agg = {
        "user_job_count": ("collection_id", "nunique"),
        "user_avg_cpu_util": (TARGET_COL, "mean"),
        "user_std_cpu_util": (TARGET_COL, "std"),
        "user_max_cpu_util": (TARGET_COL, "max"),
        "user_p90_cpu_util": (TARGET_COL, lambda x: float(np.percentile(x, 90))),
        "user_overprovision_rate": (TARGET_COL, lambda x: float((x < 0.5).mean())),
    }
    if "priority" in df_train.columns:
        agg["user_avg_priority"] = ("priority", "mean")

    stats = df_train.groupby(user_col).agg(**agg).reset_index()
    stats["user_std_cpu_util"] = stats["user_std_cpu_util"].fillna(0.0)
    feat_cols = [c for c in stats.columns if c != user_col]

    df_train = df_train.merge(stats, on=user_col, how="left")
    df_val = df_val.merge(stats, on=user_col, how="left")
    df_test = df_test.merge(stats, on=user_col, how="left")

    for col in feat_cols:
        for s in [df_train, df_val, df_test]:
            s[col] = s[col].fillna(0.0)

    for s in [df_train, df_val, df_test]:
        sparse = s["user_job_count"] < 20
        s.loc[sparse, "user_max_cpu_util"] = s.loc[sparse, "user_avg_cpu_util"]
        s.loc[sparse, "user_p90_cpu_util"] = s.loc[sparse, "user_avg_cpu_util"]

    if "scheduling_class" in df_train.columns:
        for cls in [0, 1]:
            sub = df_train[df_train["scheduling_class"] == cls]
            if not len(sub):
                continue
            ca = {
                f"user_c{cls}_job_count": ("collection_id", "nunique"),
                f"user_c{cls}_avg_cpu": (TARGET_COL, "mean"),
                f"user_c{cls}_std_cpu": (TARGET_COL, "std"),
            }
            if cls == 1:
                ca["user_c1_max_cpu"] = (TARGET_COL, "max")
                ca["user_c1_p90_cpu"] = (TARGET_COL, lambda x: float(np.percentile(x, 90)))
            cs = sub.groupby(user_col).agg(**ca).reset_index()
            cs[f"user_c{cls}_std_cpu"] = cs[f"user_c{cls}_std_cpu"].fillna(0.0)
            df_train = df_train.merge(cs, on=user_col, how="left")
            df_val = df_val.merge(cs, on=user_col, how="left")
            df_test = df_test.merge(cs, on=user_col, how="left")
            fl = [f"user_c{cls}_job_count", f"user_c{cls}_avg_cpu", f"user_c{cls}_std_cpu"]
            if cls == 1:
                fl += ["user_c1_max_cpu", "user_c1_p90_cpu"]
            for f in fl:
                for s in [df_train, df_val, df_test]:
                    s[f] = s[f].fillna(0.0)
            if cls == 1:
                for s in [df_train, df_val, df_test]:
                    sp = s["user_c1_job_count"] < 10
                    s.loc[sp, "user_c1_max_cpu"] = s.loc[sp, "user_c1_avg_cpu"]
                    s.loc[sp, "user_c1_p90_cpu"] = s.loc[sp, "user_c1_avg_cpu"]

    print(f"  Added user features for {len(stats):,} training users.")
    return df_train, df_val, df_test


# ═══════════════════════════════════════════════════════════════════════════
# Feature engineering
# ═══════════════════════════════════════════════════════════════════════════

def _flatten_scalar_ts(df, col, n_windows=3):
    if col not in df.columns:
        return {}
    s = df[col].apply(lambda x: x if isinstance(x, list) else [0.0] * n_windows)
    return {
        f"{col}_w{i+1}": s.apply(
            lambda x: float(x[i]) if isinstance(x, list) and len(x) > i and x[i] is not None else 0.0
        )
        for i in range(n_windows)
    }


def extract_histogram_stats(df, col=None, n_windows=3):
    col = col or HIST_COL
    if col not in df.columns:
        return {}
    n_buckets = None
    results = {
        f"hist_{s}_w{w+1}": []
        for w in range(n_windows)
        for s in ["max_bucket", "high_mass_ratio", "mean_bucket", "entropy", "peak_ratio"]
    }
    zf = None
    for val in df[col]:
        if isinstance(val, list) and len(val) >= n_windows and n_buckets is None:
            for w in val:
                if isinstance(w, list) and w:
                    n_buckets = len(w)
                    break
        if n_buckets is None:
            n_buckets = 11
        if zf is None:
            zf = [0.0] * n_buckets
        wd = val if isinstance(val, list) and len(val) >= n_windows else [zf[:]] * n_windows
        for w in range(n_windows):
            h = wd[w] if w < len(wd) else zf[:]
            if not isinstance(h, list) or not h:
                h = zf[:]
            if len(h) < n_buckets:
                h = h + [0.0] * (n_buckets - len(h))
            h = h[:n_buckets]
            a = np.array(h, dtype=np.float64)
            t = a.sum()
            results[f"hist_max_bucket_w{w+1}"].append(float(np.argmax(a)))
            results[f"hist_high_mass_ratio_w{w+1}"].append(float(a[max(0, n_buckets - 3):].sum() / (t + 1e-8)))
            results[f"hist_mean_bucket_w{w+1}"].append(float(np.dot(a, np.arange(n_buckets)) / (t + 1e-8)))
            p = a / t if t > 1e-8 else a
            results[f"hist_entropy_w{w+1}"].append(
                float(-np.sum(p * np.log(p + 1e-9)) / np.log(n_buckets + 1e-9)) if t > 1e-8 else 0.0
            )
            sd = np.sort(a)[::-1]
            results[f"hist_peak_ratio_w{w+1}"].append(float(sd[0] / (sd[1] + 1e-8)) if n_buckets > 1 else 0.0)
    return {k: pd.Series(v, index=df.index) for k, v in results.items()}


def prepare_boosting_features(df):
    parts = {}
    for col in BOOSTING_META_COLS + BOOSTING_USER_COLS:
        if col in df.columns:
            s = df[col]
            if s.dtype == bool:
                s = s.astype(int)
            parts[col] = pd.to_numeric(s, errors="coerce").fillna(0.0)
    for col in SCALAR_TS_COLS:
        parts.update(_flatten_scalar_ts(df, col))
    if HIST_COL in df.columns:
        parts.update(extract_histogram_stats(df))
    return pd.DataFrame(parts, index=df.index).fillna(0.0)


eps = 1e-6

def add_interaction_features(X_df):
    X = X_df.copy()
    AVG_CPU = "ts_avg_cpu_seq"
    MAX_CPU = "ts_max_cpu_seq"
    AVG_MEM = "ts_avg_mem_seq"
    MAX_MEM = "ts_max_mem_seq"

    def w(col, n):
        return f"{col}_w{n}"

    for prefix in [AVG_CPU, MAX_CPU, AVG_MEM, MAX_MEM]:
        if w(prefix, 1) in X.columns and w(prefix, 3) in X.columns:
            X[f"{prefix}_trend"] = X[w(prefix, 3)] - X[w(prefix, 1)]
            if w(prefix, 2) in X.columns:
                X[f"{prefix}_accel"] = X[w(prefix, 3)] - 2 * X[w(prefix, 2)] + X[w(prefix, 1)]

    for n in [1, 2, 3]:
        if w(MAX_CPU, n) in X.columns and w(AVG_CPU, n) in X.columns:
            X[f"cpu_burst_ratio_w{n}"] = X[w(MAX_CPU, n)] / (X[w(AVG_CPU, n)] + eps)
        if w(MAX_MEM, n) in X.columns and w(AVG_MEM, n) in X.columns:
            X[f"mem_burst_ratio_w{n}"] = X[w(MAX_MEM, n)] / (X[w(AVG_MEM, n)] + eps)
    if "cpu_burst_ratio_w1" in X.columns and "cpu_burst_ratio_w3" in X.columns:
        X["cpu_burst_trend"] = X["cpu_burst_ratio_w3"] - X["cpu_burst_ratio_w1"]

    if "user_p90_cpu_util" in X.columns and w(MAX_CPU, 3) in X.columns:
        X["job_vs_user_p90"] = X[w(MAX_CPU, 3)] / (X["user_p90_cpu_util"] + eps)
    if "user_avg_cpu_util" in X.columns and w(AVG_CPU, 3) in X.columns:
        X["job_vs_user_avg"] = X[w(AVG_CPU, 3)] / (X["user_avg_cpu_util"] + eps)
    if "user_std_cpu_util" in X.columns and "user_avg_cpu_util" in X.columns:
        X["user_cv"] = X["user_std_cpu_util"] / (X["user_avg_cpu_util"] + eps)

    if "requested_cpus" in X.columns and w(AVG_CPU, 3) in X.columns:
        X["cpu_util_rate"] = X[w(AVG_CPU, 3)] / (X["requested_cpus"] + eps)
    if "requested_memory" in X.columns and w(AVG_MEM, 3) in X.columns:
        X["mem_util_rate"] = X[w(AVG_MEM, 3)] / (X["requested_memory"] + eps)

    if w(MAX_CPU, 3) in X.columns and w(MAX_MEM, 3) in X.columns:
        X["cpu_mem_ratio_w3"] = X[w(MAX_CPU, 3)] / (X[w(MAX_MEM, 3)] + eps)
        X["cpu_mem_product_w3"] = X[w(MAX_CPU, 3)] * X[w(MAX_MEM, 3)]

    return X.fillna(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Main training pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # --- Load data ---
    print("=" * 60)
    print("Loading data...")
    df = load_and_merge_data(DATA_DIR)
    print(f"  {df.shape[0]:,} rows x {df.shape[1]} cols")

    print("\nParsing histograms...")
    df, hist_len = parse_histogram_column(df, HIST_COL)

    print("\nSplitting + user features...")
    df_train, df_val, df_test = split_by_jobs(df, random_state=42)
    df_train, df_val, df_test = add_user_features(df_train, df_val, df_test)

    # User-level class=1 histogram aggregates
    user_col_main = next((c for c in ["user", "user_id"] if c in df_train.columns), None)
    if HIST_COL in df_train.columns and "scheduling_class" in df_train.columns and user_col_main:
        hs = extract_histogram_stats(df_train)
        if hs:
            c1m = (df_train["scheduling_class"] == 1).values
            hm = hs.get("hist_high_mass_ratio_w3", pd.Series(0.0, index=df_train.index))
            mb = hs.get("hist_max_bucket_w3", pd.Series(0.0, index=df_train.index))
            tmp = pd.DataFrame({
                "user": df_train[user_col_main].values,
                "c1": c1m, "hm": hm.values, "mb": mb.values,
            })
            agg = tmp[tmp["c1"]].groupby("user").agg(
                user_c1_avg_high_mass=("hm", "mean"),
                user_c1_avg_max_bkt=("mb", "mean"),
            ).reset_index()
            am = agg.set_index("user")["user_c1_avg_high_mass"]
            ab = agg.set_index("user")["user_c1_avg_max_bkt"]
            for s in [df_train, df_val, df_test]:
                s["user_c1_avg_high_mass"] = s[user_col_main].map(am).fillna(0.0)
                s["user_c1_avg_max_bkt"] = s[user_col_main].map(ab).fillna(0.0)
            print(f"  Added C1 histogram aggregates for {len(agg)} users")

    # --- Targets ---
    y_train_orig = df_train[TARGET_COL].values.astype(np.float32)
    y_val_orig = df_val[TARGET_COL].values.astype(np.float32)
    y_test_orig = df_test[TARGET_COL].values.astype(np.float32)
    y_train = np.log1p(y_train_orig)
    y_val = np.log1p(y_val_orig)

    # --- Features ---
    print("\nBuilding feature matrices...")
    X_train_boost = prepare_boosting_features(df_train)
    X_val_boost = prepare_boosting_features(df_val)
    X_test_boost = prepare_boosting_features(df_test)

    X_train_boost = add_interaction_features(X_train_boost)
    X_val_boost = add_interaction_features(X_val_boost)
    X_test_boost = add_interaction_features(X_test_boost)
    print(f"  Feature matrix: {X_train_boost.shape}")

    feature_names = list(X_train_boost.columns)

    # --- Train LightGBM (quantile α = 0.9091) ---
    print("\n" + "=" * 60)
    print("[LightGBM] Training (quantile=0.9091)...")
    t0 = time.time()

    dt = lgb.Dataset(X_train_boost, label=y_train)
    dv = lgb.Dataset(X_val_boost, label=y_val, reference=dt)
    lgb_params = {
        "objective": "quantile", "alpha": 10 / 11,
        "max_depth": 6, "learning_rate": 0.05,
        "num_leaves": 63, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_child_samples": 20, "metric": "quantile",
        "verbose": -1, "seed": 42, "n_jobs": -1,
    }
    lgb_model = lgb.train(
        lgb_params, dt, num_boost_round=2000,
        valid_sets=[dv], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(40, first_metric_only=True, verbose=False),
            lgb.log_evaluation(200),
        ],
    )
    lgb_val_pred = np.expm1(lgb_model.predict(X_val_boost))
    lgb_test_pred = np.expm1(lgb_model.predict(X_test_boost))
    lgb_val_score = asymmetric_rmse(y_val_orig, lgb_val_pred)
    lgb_test_score = asymmetric_rmse(y_test_orig, lgb_test_pred)
    print(f"  LGB best_iter={lgb_model.best_iteration}  val={lgb_val_score:.4f}  test={lgb_test_score:.4f}  ({time.time()-t0:.1f}s)")

    # --- Train CatBoost (quantile α = 0.90) ---
    print("\n[CatBoost] Training (quantile=0.90)...")
    t0 = time.time()
    cat_model = CatBoostRegressor(
        iterations=2000, depth=6, learning_rate=0.05,
        l2_leaf_reg=10, loss_function="Quantile:alpha=0.90",
        eval_metric="Quantile:alpha=0.90",
        early_stopping_rounds=40,
        random_seed=42, verbose=200,
    )
    cat_model.fit(Pool(X_train_boost, y_train), eval_set=Pool(X_val_boost, y_val))
    cat_val_pred = np.expm1(cat_model.predict(X_val_boost))
    cat_test_pred = np.expm1(cat_model.predict(X_test_boost))
    cat_val_score = asymmetric_rmse(y_val_orig, cat_val_pred)
    cat_test_score = asymmetric_rmse(y_test_orig, cat_test_pred)
    print(f"  CatBoost val={cat_val_score:.4f}  test={cat_test_score:.4f}  ({time.time()-t0:.1f}s)")

    # --- SLSQP Ensemble ---
    print("\n[Ensemble] Optimizing weights via SLSQP...")
    V = np.array([lgb_val_pred, cat_val_pred])
    T = np.array([lgb_test_pred, cat_test_pred])
    names = ["lgb", "cat"]

    def obj(w):
        return asymmetric_rmse(y_val_orig, np.dot(w / w.sum(), V))

    res = minimize(obj, np.ones(2) / 2, bounds=[(0, 1)] * 2, method="SLSQP",
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    w_ensemble = res.x / res.x.sum()
    ensemble_test = np.dot(w_ensemble, T)
    ensemble_score = asymmetric_rmse(y_test_orig, ensemble_test)

    for n, w in zip(names, w_ensemble):
        print(f"  {n}: {w:.4f}")
    print(f"  CPU Ensemble test AsymRMSE: {ensemble_score:.4f}")
    print(f"  Under-prediction rate: {float(np.mean(y_test_orig > ensemble_test)) * 100:.1f}%")

    # --- Save artifacts ---
    print("\n" + "=" * 60)
    print("Saving model artifacts...")

    # Models
    lgb_model.save_model(str(MODEL_DIR / "lightgbm_q9091.txt"))
    cat_model.save_model(str(MODEL_DIR / "catboost_q90.cbm"))

    # Scaler (for future FT-Transformer)
    scaler = StandardScaler()
    scaler.fit(X_train_boost.values.astype(np.float32))
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Feature names and ensemble weights
    metadata = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "ensemble_weights": dict(zip(names, [float(w) for w in w_ensemble])),
        "lgb_test_asymrmse": lgb_test_score,
        "cat_test_asymrmse": cat_test_score,
        "ensemble_test_asymrmse": ensemble_score,
        "under_prediction_rate": float(np.mean(y_test_orig > ensemble_test)),
        "lgb_best_iteration": lgb_model.best_iteration,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # User statistics (for inference on new jobs)
    if user_col_main:
        user_stats = df_train.groupby(user_col_main).agg(
            user_avg_cpu_util=(TARGET_COL, "mean"),
            user_p90_cpu_util=(TARGET_COL, lambda x: float(np.percentile(x, 90))),
            user_max_cpu_util=(TARGET_COL, "max"),
            user_job_count=("collection_id", "nunique"),
        ).reset_index()
        user_stats.to_parquet(MODEL_DIR / "user_stats.parquet", index=False)

    # Save sample of 100 real jobs for the dashboard (stratified by scheduling_class)
    print("\nSaving 100-job sample for dashboard...")
    sample_dfs = []
    for cls in sorted(df_test["scheduling_class"].dropna().unique()):
        cls_df = df_test[df_test["scheduling_class"] == cls]
        n_sample = max(5, int(100 * len(cls_df) / len(df_test)))
        sample_dfs.append(cls_df.sample(n=min(n_sample, len(cls_df)), random_state=42))
    sample_df = pd.concat(sample_dfs).head(100).reset_index(drop=True)

    # Build features for sample jobs and save with metadata
    X_sample = prepare_boosting_features(sample_df)
    X_sample = add_interaction_features(X_sample)
    # Predict with ensemble
    lgb_pred = np.expm1(lgb_model.predict(X_sample))
    cat_pred = np.expm1(cat_model.predict(X_sample))
    sample_pred = w_ensemble[0] * lgb_pred + w_ensemble[1] * cat_pred

    sample_df["ensemble_prediction"] = sample_pred
    sample_df["actual_peak"] = sample_df[TARGET_COL]
    # Keep key columns for display
    keep_cols = [
        "collection_id", "instance_index", "scheduling_class", "priority",
        "requested_cpus", "requested_memory", "job_duration_minutes",
        "actual_peak", "ensemble_prediction",
    ]
    if user_col_main and user_col_main in sample_df.columns:
        keep_cols.append(user_col_main)
    # Also save the full feature matrix for these samples
    X_sample.to_parquet(MODEL_DIR / "sample_features.parquet", index=False)
    sample_df[[c for c in keep_cols if c in sample_df.columns]].to_parquet(
        MODEL_DIR / "sample_jobs.parquet", index=False
    )

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed / 60:.1f} minutes")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"  lightgbm_q9091.txt   ({(MODEL_DIR / 'lightgbm_q9091.txt').stat().st_size / 1e6:.1f} MB)")
    print(f"  catboost_q90.cbm     ({(MODEL_DIR / 'catboost_q90.cbm').stat().st_size / 1e6:.1f} MB)")
    print(f"  metadata.json")
    print(f"  scaler.pkl")
    print(f"  sample_jobs.parquet  ({len(sample_df)} jobs)")
    print(f"  sample_features.parquet")


if __name__ == "__main__":
    main()
