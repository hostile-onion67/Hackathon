"""
Step 3: Reference_Parameter regressor.

Trained ONLY on rows labeled Valid in the training set (Invalid rows have
corrupted/unreliable sensor-target relationships, so including them would
teach the model noise instead of the real relationship).

We compare RandomForest vs XGBoost (nonlinear models, since EDA + the
step1b physics-consistency check showed a genuine nonlinear regime-shift
in the Valid data that a plain linear equation under-fits).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from step1b_physics import run_step1b

REG_FEATURES = [
    "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min",
    "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
]


def evaluate_model(name, model, X, y, cv):
    pred = cross_val_predict(model, X, y, cv=cv)
    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    print(f"{name}: CV R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
    return r2, mae, rmse


def train_reference_regressor(train: pd.DataFrame):
    valid_only = train[train["Validity_Label"] == "Valid"]
    X = valid_only[REG_FEATURES]
    y = valid_only["Reference_Parameter"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        random_state=42, n_jobs=1,
    )
    xgb = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=1,
    )

    print("=== 5-fold CV comparison (trained on Valid rows only) ===")
    evaluate_model("RandomForest", rf, X, y, kf)
    evaluate_model("XGBoost     ", xgb, X, y, kf)

    # also report a held-out split for a sanity check / final model choice
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    rf.fit(X_tr, y_tr)
    xgb.fit(X_tr, y_tr)
    print("\n=== Held-out 20% split ===")
    for name, m in [("RandomForest", rf), ("XGBoost     ", xgb)]:
        p = m.predict(X_val)
        print(f"{name}: R2={r2_score(y_val, p):.4f}  MAE={mean_absolute_error(y_val, p):.4f}")

    return rf, xgb, X, y


if __name__ == "__main__":
    train, test, medians, coefs, intercept, resid_std = run_step1b()
    rf, xgb, X, y = train_reference_regressor(train)
