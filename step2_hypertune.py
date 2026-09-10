"""
Step 2 (v3): Validity classifier -- hyperparameter tuning via
RandomizedSearchCV (cross-validated, not a single lucky split), then
re-tune the decision threshold for the best found model.
"""
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from xgboost import XGBClassifier
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score, make_scorer

from step1b_physics import run_step1b

FEATURES = [
    "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min",
    "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
    "n_missing_sensors", "has_negative_sensor", "is_duplicate_input",
    "physics_predicted_ref",
]

PARAM_DIST = {
    "n_estimators": randint(150, 500),
    "max_depth": randint(2, 7),
    "learning_rate": uniform(0.01, 0.15),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_lambda": uniform(0.5, 3.0),
    "min_child_weight": randint(1, 6),
}


def find_best_threshold(y_true, proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    return thresholds[best_idx], f1s[best_idx]


def tune_validity_classifier(train: pd.DataFrame, n_iter=60):
    X = train[FEATURES]
    y = (train["Validity_Label"] == "Invalid").astype(int)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    base = XGBClassifier(
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=42, n_jobs=1,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        base, PARAM_DIST, n_iter=n_iter, scoring="f1",
        cv=skf, random_state=42, n_jobs=1, verbose=1,
    )
    search.fit(X, y)

    print(f"Best CV F1(Invalid) from search: {search.best_score_:.4f}")
    print("Best params:", search.best_params_)

    best_clf = search.best_estimator_

    # get honest out-of-fold probabilities from the BEST params to tune threshold
    oof_proba = cross_val_predict(best_clf, X, y, cv=skf, method="predict_proba")[:, 1]
    threshold, best_f1 = find_best_threshold(y, oof_proba)
    print(f"\nTuned threshold: {threshold:.3f}  OOF F1(Invalid) at that threshold: {best_f1:.3f}")
    oof_pred = (oof_proba >= threshold).astype(int)
    print(classification_report(y, oof_pred, digits=3))
    print("Confusion matrix [[TN,FP],[FN,TP]]:")
    print(confusion_matrix(y, oof_pred))

    # refit best params on ALL data for the final model
    final_clf = XGBClassifier(
        **search.best_params_, scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42, n_jobs=1,
    )
    final_clf.fit(X, y)

    importances = pd.Series(final_clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)

    return final_clf, threshold, search.best_params_


def predict_validity(clf, threshold, df: pd.DataFrame) -> pd.Series:
    proba = clf.predict_proba(df[FEATURES])[:, 1]
    pred = (proba >= threshold).astype(int)
    return pd.Series(np.where(pred == 1, "Invalid", "Valid"), index=df.index)


if __name__ == "__main__":
    train, test, medians, coefs, intercept, resid_std = run_step1b()
    clf, threshold, best_params = tune_validity_classifier(train)

    test_pred = predict_validity(clf, threshold, test)
    print("\nPredicted Validity distribution on Test_Data:")
    print(test_pred.value_counts())
