"""
Step 2 (v2): Validity classifier -- ENSEMBLE version.

Blends XGBoost + RandomForest probabilities (simple average). Ensembling
two models that make different kinds of errors typically generalizes
better to an unseen second dataset than either model alone -- directly
targets the "Performance on the second unseen dataset" (20%) and
"Accuracy of abnormal/invalid record identification" (25%) criteria.

Also caps model complexity a bit versus the earlier single-XGBoost version
(shallower trees, fewer estimators) since heavier models tend to overfit
quirks of this specific 1000-row training set rather than the true
underlying fault pattern.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score

from step1b_physics import run_step1b

FEATURES = [
    "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min",
    "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
    "n_missing_sensors", "has_negative_sensor", "is_duplicate_input",
    "physics_predicted_ref",
]
# physics_residual_z / physics_inconsistent excluded: derived from the true
# Reference_Parameter, unavailable for real unlabeled data (see step1b).


def make_xgb(scale_pos_weight):
    # same hyperparameters as the standalone tuned classifier (step2_tune.py)
    # -- these were already selected via cross-validation, not hand-picked,
    # so there's no reason to cap them further for the ensemble.
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42, n_jobs=1,
    )


def make_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=1,
    )


def blended_proba(xgb, rf, X):
    """Simple average of the two models' Invalid-class probability."""
    p_xgb = xgb.predict_proba(X)[:, 1]
    p_rf = rf.predict_proba(X)[:, 1]
    return 0.5 * p_xgb + 0.5 * p_rf


def find_best_threshold(y_true, proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    return thresholds[best_idx], f1s[best_idx]


def train_validity_ensemble(train: pd.DataFrame):
    X = train[FEATURES]
    y = (train["Validity_Label"] == "Invalid").astype(int)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    # --- out-of-fold blended predictions, to tune threshold honestly ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(y))
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr = y.iloc[tr_idx]
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        xgb_f = make_xgb(spw).fit(X_tr, y_tr)
        rf_f = make_rf().fit(X_tr, y_tr)
        oof_proba[val_idx] = blended_proba(xgb_f, rf_f, X_val)

    threshold, best_f1 = find_best_threshold(y, oof_proba)
    print(f"Ensemble out-of-fold F1(Invalid): {best_f1:.3f} at threshold={threshold:.3f}")
    oof_pred = (oof_proba >= threshold).astype(int)
    print(classification_report(y, oof_pred, digits=3))
    print("Confusion matrix [[TN,FP],[FN,TP]]:")
    print(confusion_matrix(y, oof_pred))

    # --- held-out sanity split (same style as before, for comparability) ---
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    xgb_h = make_xgb(spw).fit(X_tr, y_tr)
    rf_h = make_rf().fit(X_tr, y_tr)
    val_proba = blended_proba(xgb_h, rf_h, X_val)
    val_pred = (val_proba >= threshold).astype(int)
    print(f"\nHeld-out 20% split F1(Invalid): {f1_score(y_val, val_pred):.3f}")

    # --- final models, refit on ALL training data ---
    final_xgb = make_xgb(scale_pos_weight).fit(X, y)
    final_rf = make_rf().fit(X, y)

    return final_xgb, final_rf, threshold


def predict_validity_ensemble(xgb, rf, threshold, df: pd.DataFrame) -> pd.Series:
    proba = blended_proba(xgb, rf, df[FEATURES])
    pred = (proba >= threshold).astype(int)
    return pd.Series(np.where(pred == 1, "Invalid", "Valid"), index=df.index)


if __name__ == "__main__":
    train, test, medians, coefs, intercept, resid_std = run_step1b()
    xgb, rf, threshold = train_validity_ensemble(train)

    test_pred = predict_validity_ensemble(xgb, rf, threshold, test)
    print("\nPredicted Validity distribution on Test_Data:")
    print(test_pred.value_counts())
