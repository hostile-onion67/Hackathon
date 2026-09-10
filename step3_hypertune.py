"""
Step 3 (v2): Reference_Parameter regressor -- hyperparameter tuning via
RandomizedSearchCV, scored by R2 (5-fold CV), on Valid rows only.
"""
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from step1b_physics import run_step1b

REG_FEATURES = [
    "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min",
    "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
]

PARAM_DIST = {
    "n_estimators": randint(200, 700),
    "max_depth": randint(2, 7),
    "learning_rate": uniform(0.01, 0.12),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_lambda": uniform(0.5, 3.0),
    "min_child_weight": randint(1, 6),
}


def tune_regressor(train: pd.DataFrame, n_iter=60):
    valid_only = train[train["Validity_Label"] == "Valid"]
    X = valid_only[REG_FEATURES]
    y = valid_only["Reference_Parameter"]

    base = XGBRegressor(random_state=42, n_jobs=1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        base, PARAM_DIST, n_iter=n_iter, scoring="r2",
        cv=kf, random_state=42, n_jobs=1, verbose=1,
    )
    search.fit(X, y)

    print(f"Best CV R2 from search: {search.best_score_:.4f}")
    print("Best params:", search.best_params_)

    best_reg = search.best_estimator_
    oof_pred = cross_val_predict(best_reg, X, y, cv=kf)
    r2 = r2_score(y, oof_pred)
    mae = mean_absolute_error(y, oof_pred)
    rmse = np.sqrt(mean_squared_error(y, oof_pred))
    print(f"\nOOF R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # refit on ALL Valid rows for the final model
    final_reg = XGBRegressor(**search.best_params_, random_state=42, n_jobs=1)
    final_reg.fit(X, y)

    importances = pd.Series(final_reg.feature_importances_, index=REG_FEATURES).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)

    return final_reg, search.best_params_


if __name__ == "__main__":
    train, test, medians, coefs, intercept, resid_std = run_step1b()
    reg, best_params = tune_regressor(train)
