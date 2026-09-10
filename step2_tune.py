"""
Step 2 tuning: try XGBoost alongside RandomForest, and tune the decision
threshold (instead of the default 0.5) to better trade precision for
recall on the Invalid class, since recall was the weak point (0.56).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_recall_curve, classification_report, confusion_matrix
from xgboost import XGBClassifier

from step1b_physics import run_step1b
from step2_validity_classifier import FEATURES

train, test, medians, coefs, intercept, resid_std = run_step1b()
X = train[FEATURES]
y = (train["Validity_Label"] == "Invalid").astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Model A: RandomForest (as before) ---
rf = RandomForestClassifier(
    n_estimators=400, max_depth=8, min_samples_leaf=3,
    class_weight="balanced", random_state=42, n_jobs=1,
)
rf_proba = cross_val_predict(rf, X, y, cv=skf, method="predict_proba")[:, 1]

# --- Model B: XGBoost ---
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
xgb = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss", random_state=42, n_jobs=1,
)
xgb_proba = cross_val_predict(xgb, X, y, cv=skf, method="predict_proba")[:, 1]

print("=== Default threshold (0.5) comparison, out-of-fold predictions ===")
for name, proba in [("RandomForest", rf_proba), ("XGBoost", xgb_proba)]:
    pred = (proba >= 0.5).astype(int)
    print(f"\n{name} F1(Invalid) @0.5: {f1_score(y, pred):.3f}")
    print(classification_report(y, pred, digits=3))

print("\n=== Threshold tuning (maximize F1 on Invalid class) ===")
for name, proba in [("RandomForest", rf_proba), ("XGBoost", xgb_proba)]:
    precisions, recalls, thresholds = precision_recall_curve(y, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])  # last point has no threshold
    best_thresh = thresholds[best_idx]
    best_pred = (proba >= best_thresh).astype(int)
    print(f"\n{name}: best threshold = {best_thresh:.3f}, F1 = {f1s[best_idx]:.3f}")
    print(classification_report(y, best_pred, digits=3))
    print("Confusion matrix [[TN,FP],[FN,TP]]:")
    print(confusion_matrix(y, best_pred))
