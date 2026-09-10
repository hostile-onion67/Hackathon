"""
Step 2: Validity (Valid/Invalid) classifier.

Uses the 8 raw inputs + the data-quality/physics flags engineered in
step1_clean.py / step1b_physics.py. Reference_Parameter is NOT used as a
feature (it isn't available for the test set / real new data, so using it
would be leakage that can't be replicated at prediction time).

Model: XGBoost (outperformed RandomForest substantially in tuning --
F1(Invalid) 0.91 vs 0.80 -- see step2_tune.py for the comparison).
Handles class imbalance (866 Valid / 134 Invalid) via scale_pos_weight,
and uses a TUNED decision threshold (0.191, chosen by maximizing F1 on
the Invalid class via 5-fold out-of-fold cross-validation) instead of
the default 0.5, since a plain 0.5 cutoff badly under-flags Invalid rows.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from step1b_physics import run_step1b

FEATURES = [
    "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min",
    "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
    "n_missing_sensors", "has_negative_sensor", "is_duplicate_input",
    "physics_predicted_ref",
]
# NOTE: physics_residual_z / physics_inconsistent are deliberately excluded.
# Both were computed FROM the true Reference_Parameter (step1b), which is
# not available for real unlabeled test data -- including them here would
# be leakage the model could never actually use at prediction time.

DECISION_THRESHOLD = 0.191  # tuned in step2_tune.py to maximize F1(Invalid)


def train_validity_classifier(train: pd.DataFrame):
    X = train[FEATURES]
    y = (train["Validity_Label"] == "Invalid").astype(int)  # 1 = Invalid

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42, n_jobs=1,
    )
    clf.fit(X_tr, y_tr)

    val_proba = clf.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= DECISION_THRESHOLD).astype(int)
    print(f"Held-out validation report (1 = Invalid, threshold={DECISION_THRESHOLD}):")
    print(classification_report(y_val, val_pred, digits=3))
    print("Confusion matrix [[TN,FP],[FN,TP]]:")
    print(confusion_matrix(y_val, val_pred))

    # refit on ALL training data for the final model used on test set
    scale_pos_weight_full = (y == 0).sum() / (y == 1).sum()
    final_clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight_full,
        eval_metric="logloss", random_state=42, n_jobs=1,
    )
    final_clf.fit(X, y)

    importances = pd.Series(final_clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)

    return final_clf, importances


def predict_validity(clf, df: pd.DataFrame) -> pd.Series:
    """Returns 'Valid'/'Invalid' string labels using the tuned threshold."""
    proba = clf.predict_proba(df[FEATURES])[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    return pd.Series(np.where(pred == 1, "Invalid", "Valid"), index=df.index)


if __name__ == "__main__":
    train, test, medians, coefs, intercept, resid_std = run_step1b()
    clf, importances = train_validity_classifier(train)

    test_pred = predict_validity(clf, test)
    print("\nPredicted Validity distribution on Test_Data:")
    print(test_pred.value_counts())

