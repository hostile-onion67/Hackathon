"""
predict_test.py
Loads the models saved by train_models.py, runs the full cleaning +
prediction pipeline on Test_Data, and writes the submission CSV:
    Test_ID, Predicted_Reference_Parameter, Valid/Invalid

Usage:
    python3 predict_test.py --team YourTeamName

If --team is omitted, defaults to "submission".
"""
import argparse
import joblib
import pandas as pd

from step1_clean import load_raw, engineer_quality_flags, clean_negative_sensors, SENSOR_COLS

MODELS_DIR = "models"


def prepare_test_features(test_raw: pd.DataFrame, medians, physics_bundle):
    """Same cleaning steps as training, but medians/physics model come from
    the SAVED training artifacts (no re-fitting on test data)."""
    df = engineer_quality_flags(test_raw)
    df = clean_negative_sensors(df)

    for c in SENSOR_COLS:
        df[c] = df[c].fillna(medians[c])

    physics_model = physics_bundle["model"]
    physics_features = physics_bundle["features"]
    df["physics_predicted_ref"] = physics_model.predict(df[physics_features].values)

    return df


def main(team_name: str):
    print("Loading saved models...")
    clf_bundle = joblib.load(f"{MODELS_DIR}/validity_classifier.joblib")
    reg_bundle = joblib.load(f"{MODELS_DIR}/reference_regressor.joblib")
    physics_bundle = joblib.load(f"{MODELS_DIR}/physics_equation.joblib")
    medians = joblib.load(f"{MODELS_DIR}/imputation_medians.joblib")

    print("Loading + cleaning Test_Data...")
    _, test_raw = load_raw()
    test = prepare_test_features(test_raw, medians, physics_bundle)

    print("Predicting Validity...")
    clf = clf_bundle["model"]
    clf_features = clf_bundle["features"]
    threshold = clf_bundle["threshold"]
    proba_invalid = clf.predict_proba(test[clf_features])[:, 1]
    validity = pd.Series(
        (proba_invalid >= threshold), index=test.index
    ).map({True: "Invalid", False: "Valid"})

    print("Predicting Reference_Parameter...")
    reg = reg_bundle["model"]
    reg_features = reg_bundle["features"]
    predicted_ref = reg.predict(test[reg_features])

    submission = pd.DataFrame({
        "Test_ID": test["Test_ID"],
        "Predicted_Reference_Parameter": predicted_ref,
        "Valid/Invalid": validity.values,
    })

    out_path = f"{team_name}.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}  ({len(submission)} rows)")
    print("\nValidity distribution:")
    print(submission["Valid/Invalid"].value_counts())
    print("\nPredicted_Reference_Parameter summary:")
    print(submission["Predicted_Reference_Parameter"].describe())
    print("\nFirst 5 rows:")
    print(submission.head())

    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="submission", help="Team name (used as output filename)")
    args = parser.parse_args()
    main(args.team)
