"""
train_models.py
Run this ONE script to train + save both models locally, using the
hyperparameters found via RandomizedSearchCV in step2_hypertune.py and
step3_hypertune.py (5-fold cross-validated, not hand-picked).

Usage:
    python3 train_models.py

Produces (in ./models/):
    validity_classifier.joblib
    reference_regressor.joblib
    physics_equation.joblib   (the Load_Current/Sensor_S2 consistency model)
    imputation_medians.joblib
"""
import os
import joblib
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from step1b_physics import run_step1b, PHYSICS_FEATURES
from step2_hypertune import FEATURES as CLF_FEATURES
from step3_hypertune import REG_FEATURES

MODELS_DIR = "models"

# Best params found by RandomizedSearchCV (5-fold CV, see step2_hypertune.py)
# OOF F1(Invalid) = 0.914 at threshold 0.168
CLF_BEST_PARAMS = {
    "colsample_bytree": 0.7035119926400067,
    "learning_rate": 0.10937834265309729,
    "max_depth": 3,
    "min_child_weight": 2,
    "n_estimators": 409,
    "reg_lambda": 1.054563366576581,
    "subsample": 0.9878338511058234,
}
CLF_THRESHOLD = 0.168

# Best params found by RandomizedSearchCV (5-fold CV, see step3_hypertune.py)
# OOF R2 = 0.9958, MAE = 0.449
REG_BEST_PARAMS = {
    "colsample_bytree": 0.8604308102007778,
    "learning_rate": 0.11979516106525369,
    "max_depth": 2,
    "min_child_weight": 4,
    "n_estimators": 617,
    "reg_lambda": 0.7862303494712339,
    "subsample": 0.7483273008793065,
}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading + cleaning data (step 1 & 1b)...")
    train, test, medians, coefs, intercept, resid_std = run_step1b()

    # ---- physics consistency equation ----
    from step1b_physics import fit_physics_equation
    physics_model, physics_coefs, physics_intercept = fit_physics_equation(train)
    joblib.dump(
        {"model": physics_model, "features": PHYSICS_FEATURES, "resid_std": resid_std},
        f"{MODELS_DIR}/physics_equation.joblib",
    )
    print(f"Saved physics equation: {physics_coefs}, intercept={physics_intercept:.4f}")

    # ---- validity classifier (tuned params) ----
    print("\nTraining validity classifier (XGBoost, tuned)...")
    X_clf = train[CLF_FEATURES]
    y_clf = (train["Validity_Label"] == "Invalid").astype(int)
    scale_pos_weight = (y_clf == 0).sum() / (y_clf == 1).sum()
    clf = XGBClassifier(
        **CLF_BEST_PARAMS,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42, n_jobs=1,
    )
    clf.fit(X_clf, y_clf)
    joblib.dump(
        {"model": clf, "features": CLF_FEATURES, "threshold": CLF_THRESHOLD},
        f"{MODELS_DIR}/validity_classifier.joblib",
    )
    print("Saved validity_classifier.joblib")

    # ---- reference parameter regressor (tuned params, Valid rows only) ----
    print("\nTraining Reference_Parameter regressor (XGBoost, tuned, Valid rows only)...")
    valid_only = train[train["Validity_Label"] == "Valid"]
    X_reg = valid_only[REG_FEATURES]
    y_reg = valid_only["Reference_Parameter"]
    reg = XGBRegressor(**REG_BEST_PARAMS, random_state=42, n_jobs=1)
    reg.fit(X_reg, y_reg)
    joblib.dump(
        {"model": reg, "features": REG_FEATURES},
        f"{MODELS_DIR}/reference_regressor.joblib",
    )
    print("Saved reference_regressor.joblib")

    # ---- imputation medians ----
    joblib.dump(medians, f"{MODELS_DIR}/imputation_medians.joblib")
    print("Saved imputation_medians.joblib")

    print(f"\nAll models saved to ./{MODELS_DIR}/")


if __name__ == "__main__":
    main()

