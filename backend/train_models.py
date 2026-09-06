"""
STEP 3 of the ML pipeline: train and evaluate three models on the
synthetic dataset produced by generate_dataset.py.

  1. Anomaly detector — a shallow neural-network autoencoder
     (scikit-learn's MLPRegressor with a bottleneck hidden layer, trained
     to reconstruct its own input) fit ONLY on nominal ("none" fault_type)
     rows. At inference, reconstruction error on a new row is the anomaly
     score: a row that looks like something the network never saw during
     nominal training reconstructs poorly. This is literally "compare live
     behavior with expected behavior" implemented as a model, complementing
     — not just duplicating — the residual-vs-twin comparison already
     computed per-row by the physics layer.

     Honesty note: this is a shallow (3-hidden-layer) neural network via
     scikit-learn, not a deep multi-layer PyTorch/TensorFlow model. A full
     PyTorch install was attempted and is functionally fine but its CUDA-
     bundled wheel is ~550MB and this sandbox only had ~2.8GB free disk —
     installing it risked filling the disk mid-task, so a CPU-only shallow
     NN was used instead. If you're running this outside a disk-constrained
     sandbox, swapping in a PyTorch autoencoder or LSTM is the natural
     upgrade — the feature/label pipeline (features.py, this file's data
     prep) does not need to change for that.

  2. Fault classifier — RandomForestClassifier, multiclass over the 9
     fault_type labels (including "none"). Classical ML, chosen because it
     is fast to train on ~440k rows, robust to the mixed feature scales
     here, and gives interpretable feature importances.

  3. RUL regressor — RandomForestRegressor trained ONLY on rows from runs
     that reached a failure threshold (see dataset_builder.py), predicting
     seconds until that threshold. RUL is capped at RUL_CAP_S — a standard
     technique from the RUL-estimation literature (e.g. NASA C-MAPSS-style
     "piecewise-linear RUL") because a flat, pre-onset sensor reading
     genuinely doesn't carry information about whether failure is 20
     minutes or 3 hours away; asking the regressor to guess an uncapped
     number for those rows would just teach it noise.

Train/test split is done by run_id (not by row) — rows within one run are
highly autocorrelated, so a row-level split would leak information from a
run's future into its own training data and overstate accuracy.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from app.ml.features import add_rolling_features, feature_columns, FEATURE_CHANNELS
from app.ml.fault_injection import FAULT_TYPES

DATA_PATH = "data/synthetic_dataset.csv"
MODEL_DIR = "models"
RUL_CAP_S = 1800.0
ANOMALY_SEVERITY_THRESHOLD = 0.15  # ground truth for evaluating the detector only
AUTOENCODER_TRAIN_SAMPLES = 30_000
# NOTE: this sandbox has a single CPU core (checked via nproc), so
# n_jobs=-1 buys nothing and RandomForest wall-clock time is entirely
# tree_count * depth * row_count. A first attempt (200 trees, depth 18,
# full ~330k-row training set) was still running after several minutes.
# Subsampling the training rows and capping tree depth/count keeps this
# runnable single-core while still training on comfortably more data than
# the number of distinct scenarios (184 runs) requires for these forests
# to converge.
CLASSIFIER_TRAIN_SAMPLES = 60_000
RUL_TRAIN_SAMPLES = 35_000
RANDOM_STATE = 42


def split_run_ids(df: pd.DataFrame, test_frac: float = 0.25, seed: int = RANDOM_STATE):
    """Stratified-by-fault_type split at the run_id level."""
    rng = np.random.RandomState(seed)
    train_ids, test_ids = set(), set()
    run_fault = df.groupby("run_id")["fault_type"].first()
    for fault_type, group in run_fault.groupby(run_fault):
        ids = list(group.index)
        rng.shuffle(ids)
        n_test = max(1, int(round(len(ids) * test_frac)))
        test_ids.update(ids[:n_test])
        train_ids.update(ids[n_test:])
    return train_ids, test_ids


def main():
    t0 = time.time()
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} rows, {df['run_id'].nunique()} runs ({time.time()-t0:.1f}s)")

    print("Computing rolling-window features...")
    df = add_rolling_features(df)
    feat_cols = feature_columns()
    print(f"  {len(feat_cols)} feature columns ({time.time()-t0:.1f}s)")

    train_ids, test_ids = split_run_ids(df)
    train_df = df[df["run_id"].isin(train_ids)]
    test_df = df[df["run_id"].isin(test_ids)]
    print(f"  train runs={len(train_ids)} ({len(train_df):,} rows), "
          f"test runs={len(test_ids)} ({len(test_df):,} rows)")

    X_test = test_df[feat_cols].values

    # =====================================================================
    # 1. Anomaly detector: shallow NN autoencoder, nominal-only training
    # =====================================================================
    print("\n=== Training anomaly detector (autoencoder on nominal data only) ===")
    nominal_train = train_df[train_df["fault_type"] == "none"]
    sample_n = min(AUTOENCODER_TRAIN_SAMPLES, len(nominal_train))
    ae_train = nominal_train.sample(sample_n, random_state=RANDOM_STATE)
    X_ae_train = ae_train[feat_cols].values

    # NOTE: the scaler is fit on a sample of the FULL training set (every
    # fault type), not just the nominal rows the autoencoder itself trains
    # on. This was a real bug the first time through: several residual_*
    # columns (e.g. residual_oil_pressure_Pa) are EXACTLY 0.0 for every
    # nominal row by construction (sensed==expected with no fault
    # applied), so a scaler fit on nominal-only data sees zero variance
    # there. StandardScaler safely avoids a divide-by-zero by setting
    # scale_=1.0 for those columns instead of erroring -- but scale_=1.0
    # means "don't rescale at all", so raw Pascal-scale residuals
    # (~1e5) passed straight through into the network, swamped every
    # properly-normalized feature in the reconstruction error, and blew
    # the anomaly score up to the ~1e8 range (caught by testing the live
    # API against a manually-dialed fault and noticing the score was
    # nowhere near the ~0.03 threshold it should have been). Fitting the
    # scaler on the full distribution gives every column real variance to
    # normalize against.
    scaler_sample_n = min(60_000, len(train_df))
    scaler_fit_sample = train_df.sample(scaler_sample_n, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(scaler_fit_sample[feat_cols].values)
    X_ae_train_s = scaler.transform(X_ae_train)

    autoencoder = MLPRegressor(
        hidden_layer_sizes=(32, 8, 32),
        activation="relu",
        max_iter=120,
        early_stopping=True,
        n_iter_no_change=8,
        random_state=RANDOM_STATE,
    )
    autoencoder.fit(X_ae_train_s, X_ae_train_s)
    print(f"  trained on {sample_n:,} nominal rows, {autoencoder.n_iter_} iterations "
          f"({time.time()-t0:.1f}s)")

    # reconstruction error on the SAME nominal-training distribution sets the
    # detection threshold (99th percentile -> ~1% false-positive rate on
    # nominal data by construction)
    recon_train = autoencoder.predict(X_ae_train_s)
    err_train = np.mean((X_ae_train_s - recon_train) ** 2, axis=1)
    threshold = float(np.percentile(err_train, 99))

    X_test_s = scaler.transform(X_test)
    recon_test = autoencoder.predict(X_test_s)
    err_test = np.mean((X_test_s - recon_test) ** 2, axis=1)
    predicted_anomaly = err_test > threshold
    actual_anomaly = (test_df["fault_type"] != "none") & (test_df["severity"] > ANOMALY_SEVERITY_THRESHOLD)

    p, r, f1, _ = precision_recall_fscore_support(
        actual_anomaly, predicted_anomaly, average="binary", zero_division=0
    )
    print(f"  threshold (99th pct nominal recon. error) = {threshold:.4f}")
    print(f"  test set: precision={p:.3f} recall={r:.3f} f1={f1:.3f}")
    print(f"  ({actual_anomaly.sum():,} true anomalous rows / {len(test_df):,} test rows)")

    # =====================================================================
    # 2. Fault-type classifier
    # =====================================================================
    print("\n=== Training fault-type classifier (RandomForest) ===")
    n_per_class = CLASSIFIER_TRAIN_SAMPLES // train_df["fault_type"].nunique()
    clf_train_parts = [
        g.sample(min(len(g), n_per_class), random_state=RANDOM_STATE)
        for _, g in train_df.groupby("fault_type")
    ]
    clf_train_df = pd.concat(clf_train_parts, ignore_index=True)
    X_train_clf = clf_train_df[feat_cols].values
    y_train_cls = clf_train_df["fault_type"].values
    y_test_cls = test_df["fault_type"].values

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced_subsample"
    )
    clf.fit(X_train_clf, y_train_cls)
    y_pred_cls = clf.predict(X_test)
    print("--- raw metrics (includes every fault run's pre-onset rows, which are\n"
          "    physically identical to nominal and labeled with that run's fault_type\n"
          "    anyway -- this is why 'none' shows low precision / perfect recall: the\n"
          "    model correctly calls pre-onset rows 'none', which counts against it here) ---")
    print(classification_report(y_test_cls, y_pred_cls, zero_division=0))

    post_onset_mask = ((test_df["fault_type"] == "none") | (test_df["severity"] > ANOMALY_SEVERITY_THRESHOLD)).values
    print(f"--- same model, evaluated only where a fault signature actually exists\n"
          f"    (true 'none' rows, or severity>{ANOMALY_SEVERITY_THRESHOLD} -- isolates how well it reads\n"
          f"    an actual fault once one is present, {post_onset_mask.sum():,}/{len(test_df):,} test rows) ---")
    print(classification_report(y_test_cls[post_onset_mask], y_pred_cls[post_onset_mask], zero_division=0))

    importances = sorted(zip(feat_cols, clf.feature_importances_), key=lambda x: -x[1])[:10]
    print("Top 10 features by importance:")
    for name, imp in importances:
        print(f"  {name:35s} {imp:.4f}")

    # =====================================================================
    # 3. RUL regressor — failure-bound runs only
    # =====================================================================
    print("\n=== Training RUL regressor (RandomForest, failure-bound runs only) ===")
    rul_train_df = train_df[train_df["reaches_failure"] == True].copy()  # noqa: E712
    rul_test_df = test_df[test_df["reaches_failure"] == True].copy()  # noqa: E712
    if len(rul_train_df) > RUL_TRAIN_SAMPLES:
        rul_train_df = rul_train_df.sample(RUL_TRAIN_SAMPLES, random_state=RANDOM_STATE)
    rul_train_df["RUL_capped"] = rul_train_df["RUL_s"].clip(upper=RUL_CAP_S)
    rul_test_df["RUL_capped"] = rul_test_df["RUL_s"].clip(upper=RUL_CAP_S)

    reg = RandomForestRegressor(n_estimators=120, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE)
    reg.fit(rul_train_df[feat_cols].values, rul_train_df["RUL_capped"].values)
    y_pred_rul = reg.predict(rul_test_df[feat_cols].values)
    mae = mean_absolute_error(rul_test_df["RUL_capped"].values, y_pred_rul)
    print(f"  train rows={len(rul_train_df):,} test rows={len(rul_test_df):,}")
    print(f"  overall MAE = {mae:.1f}s (cap={RUL_CAP_S:.0f}s)")

    near_failure = rul_test_df["RUL_capped"] < 600
    if near_failure.sum() > 0:
        mae_near = mean_absolute_error(rul_test_df.loc[near_failure, "RUL_capped"], y_pred_rul[near_failure.values])
        print(f"  MAE restricted to RUL<600s (operationally the regime that matters most) = {mae_near:.1f}s "
              f"over {near_failure.sum():,} rows")

    # =====================================================================
    # Save everything needed for inference
    # =====================================================================
    joblib.dump(clf, f"{MODEL_DIR}/fault_classifier.joblib")
    joblib.dump(reg, f"{MODEL_DIR}/rul_regressor.joblib")
    joblib.dump(autoencoder, f"{MODEL_DIR}/anomaly_autoencoder.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/anomaly_scaler.joblib")
    with open(f"{MODEL_DIR}/metadata.json", "w") as f:
        json.dump(
            {
                "feature_columns": feat_cols,
                "fault_types": FAULT_TYPES,
                "anomaly_threshold": threshold,
                "rul_cap_s": RUL_CAP_S,
                "window_samples": 15,
            },
            f,
            indent=2,
        )
    print(f"\nSaved models to {MODEL_DIR}/. Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
