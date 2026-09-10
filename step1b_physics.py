"""
Step 1b: Physics-consistency check.

We don't know the true governing equation (it's intentionally hidden), but
we CAN learn an approximate one from rows we're fairly confident are clean
(Valid label, no duplicate flag, no negative-sensor flag), then measure how
far every row's actual Reference_Parameter falls from what that learned
relationship predicts. A big deviation means the input/sensor combination
is "impossible to match" with the Reference_Parameter under the discovered
relationship -- i.e. a strong Invalid signal, on top of the duplicate/sensor
flags from step 1.

We use a robust (Huber) linear fit so a handful of remaining bad rows in
the "confident-clean" subset don't distort the equation.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from step1_clean import run_step1, SENSOR_COLS

# Only Load_Current_A + Sensor_S2 are used: S1 and S3 are highly correlated
# with S2 (r=0.79, 0.86) and S2 (r=0.86 with S3), so including them causes
# multicollinearity -- coefficients flip sign and the equation extrapolates
# to physically impossible values on new data. This pair is stable, keeps
# R^2 nearly as high (0.83 vs 0.86), and both coefficients come out with
# the physically expected positive sign.
PHYSICS_FEATURES = ["Load_Current_A", "Sensor_S2"]


def fit_physics_equation(train: pd.DataFrame):
    """Fit an approximate Reference_Parameter = f(inputs) equation on the
    subset of training rows we most trust: Valid label, not a duplicate
    input combo, no negative-sensor fault. HuberRegressor is robust to the
    handful of remaining outliers (e.g. the genuine regime-shift rows)."""
    confident = train[
        (train["Validity_Label"] == "Valid")
        & (train["is_duplicate_input"] == 0)
        & (train["has_negative_sensor"] == 0)
    ]

    X = confident[PHYSICS_FEATURES].values
    y = confident["Reference_Parameter"].values

    model = HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=500)
    model.fit(X, y)

    coefs = dict(zip(PHYSICS_FEATURES, model.coef_))
    return model, coefs, model.intercept_


def add_physics_residual(df: pd.DataFrame, model, resid_std, has_target: bool):
    """Add predicted Reference_Parameter (from the learned equation), the
    residual, its z-score, and a binary 'physics_inconsistent' flag.
    For the Test set (no true Reference_Parameter yet) we still compute the
    predicted value -- it becomes our Task-2 baseline/sanity check."""
    df = df.copy()
    X = df[PHYSICS_FEATURES].values
    df["physics_predicted_ref"] = model.predict(X)

    if has_target:
        df["physics_residual"] = df["Reference_Parameter"] - df["physics_predicted_ref"]
        df["physics_residual_z"] = df["physics_residual"] / resid_std
        df["physics_inconsistent"] = (df["physics_residual_z"].abs() > 3).astype(int)
    return df


def run_step1b():
    train, test, medians = run_step1()

    model, coefs, intercept = fit_physics_equation(train)

    confident = train[
        (train["Validity_Label"] == "Valid")
        & (train["is_duplicate_input"] == 0)
        & (train["has_negative_sensor"] == 0)
    ]
    resid_std = (
        confident["Reference_Parameter"].values
        - model.predict(confident[PHYSICS_FEATURES].values)
    ).std()

    train = add_physics_residual(train, model, resid_std, has_target=True)
    test = add_physics_residual(test, model, resid_std, has_target=False)

    return train, test, medians, coefs, intercept, resid_std


if __name__ == "__main__":
    train, test, medians, coefs, intercept, resid_std = run_step1b()

    print("Learned approximate physics equation:")
    eq = " + ".join(f"{v:.4f} * {k}" for k, v in coefs.items())
    print(f"Reference_Parameter ~= {intercept:.4f} + {eq}")
    print(f"\nResidual std (on confident-clean subset): {resid_std:.4f}")

    print("\nphysics_inconsistent flag vs Validity_Label (train):")
    print(train.groupby("physics_inconsistent")["Validity_Label"].value_counts())

    print("\nOverlap check: physics_inconsistent vs is_duplicate_input (train):")
    print(pd.crosstab(train["physics_inconsistent"], train["is_duplicate_input"]))

    print("\nTest set: predicted Reference_Parameter range (sanity check only):")
    print(test["physics_predicted_ref"].describe())
