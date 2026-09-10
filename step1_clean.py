"""
Step 1: Load raw data, clean it, and engineer data-quality flag features.
These flags (duplicate input combo, sensor fault, missing count) become
INPUT FEATURES for the Validity classifier in step 2 -- we don't just
delete suspicious rows, we let the model learn from the fault pattern.
"""
import pandas as pd
import numpy as np

RAW_PATH = "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
# NOTE: place the dataset xlsx in the SAME folder as this script (or edit
# this path to point wherever it actually sits on your machine).

INPUT_COLS = [
    "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C",
    "Test_Duration_min", "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
]
SENSOR_COLS = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]


def load_raw():
    train = pd.read_excel(RAW_PATH, sheet_name="Training_Data")
    test = pd.read_excel(RAW_PATH, sheet_name="Test_Data")
    return train, test


def engineer_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add data-quality flag columns. Does not touch the raw sensor values yet."""
    df = df.copy()

    # 1. missing-value count per row (before imputation)
    df["n_missing_sensors"] = df[SENSOR_COLS].isna().sum(axis=1)

    # 2. sensor sanity flag: any sensor reading physically impossible (<0)
    df["has_negative_sensor"] = (df[SENSOR_COLS] < 0).any(axis=1).astype(int)

    # 3. duplicate-input-combo flag: same rounded input combo appears >1 time
    #    (repeat test artifact -> strong Invalid signal seen in EDA)
    rounded = df[INPUT_COLS].round(3)
    df["is_duplicate_input"] = rounded.duplicated(keep=False).astype(int)

    return df


def clean_negative_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Negative sensor readings are physically impossible -> treat as missing
    so they get imputed like any other missing value, rather than silently
    keeping an impossible number as a model input."""
    df = df.copy()
    for c in SENSOR_COLS:
        df.loc[df[c] < 0, c] = np.nan
    return df


def impute_sensors(train: pd.DataFrame, test: pd.DataFrame):
    """Median-impute sensor columns, using TRAIN medians only (fit on train,
    apply to both) to avoid leaking test-set information into training."""
    medians = train[SENSOR_COLS].median()
    train = train.copy()
    test = test.copy()
    for c in SENSOR_COLS:
        train[c] = train[c].fillna(medians[c])
        test[c] = test[c].fillna(medians[c])
    return train, test, medians


def run_step1():
    train, test = load_raw()

    train = engineer_quality_flags(train)
    test = engineer_quality_flags(test)

    train = clean_negative_sensors(train)
    test = clean_negative_sensors(test)

    train, test, medians = impute_sensors(train, test)

    return train, test, medians


if __name__ == "__main__":
    train, test, medians = run_step1()
    print("Sensor medians used for imputation (fit on train only):")
    print(medians)
    print("\nTrain shape after step 1:", train.shape)
    print("Test shape after step 1:", test.shape)
    print("\nFlag column summary (train):")
    print(train[["n_missing_sensors", "has_negative_sensor", "is_duplicate_input"]].sum())
    print("\nFlag column summary (test):")
    print(test[["n_missing_sensors", "has_negative_sensor", "is_duplicate_input"]].sum())
    print("\nCross-check: duplicate-input flag vs Validity_Label (train):")
    print(train.groupby("is_duplicate_input")["Validity_Label"].value_counts())
