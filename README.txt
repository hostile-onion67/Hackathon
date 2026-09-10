CPRI Hackathon Pipeline - Setup & Run
=======================================

1. (macOS only, if XGBoost hangs) install libomp:
   brew install libomp

2. Install dependencies:
   pip install -r requirements.txt

3. Train everything (cleaning -> physics eq -> classifier -> regressor):
   python3 train_models.py

   This saves 4 files into ./models/:
     - validity_classifier.joblib
     - reference_regressor.joblib
     - physics_equation.joblib
     - imputation_medians.joblib

Files:
  step1_clean.py                  - load, clean, flag data quality issues
  step1b_physics.py               - data-driven consistency equation + flag
  step2_validity_classifier.py    - original single-XGBoost classifier (superseded)
  step2_tune.py                   - RF vs XGBoost model comparison
  step2_ensemble.py               - XGB+RF blend experiment (found NOT to help - kept for reference)
  step2_hypertune.py              - RandomizedSearchCV tuning -> FINAL classifier params (F1=0.914)
  step3_regressor.py              - original single-XGBoost regressor (superseded)
  step3_hypertune.py              - RandomizedSearchCV tuning -> FINAL regressor params (R2=0.996)
  train_models.py                 - MAIN SCRIPT: runs everything end-to-end using
                                     the tuned params, saves models
  CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx  - the dataset

The dataset file is already in this folder, so all scripts run as-is with
no path edits needed. n_jobs is set to 1 everywhere (single-threaded) to
avoid a known macOS OpenMP hang with XGBoost's default n_jobs=-1.

Final model performance (5-fold cross-validated, out-of-fold):
  Validity classifier : F1(Invalid) = 0.914, precision 0.911, recall 0.918
  Reference regressor : R2 = 0.9958, MAE = 0.449

Note: an XGBoost+RandomForest ensemble was tested (step2_ensemble.py) but
performed WORSE than tuned XGBoost alone on honest out-of-fold evaluation,
so it was not used in the final pipeline. Kept in the repo to show the
comparison was actually done, for the "engineering reasoning" criterion.

STILL TO BUILD (next steps):
  - predict_test.py     -> load saved models, run on Test_Data, output <TeamName>.csv
  - generate_summary.py -> summary.json / summary.csv
  - methodology note (write-up, not code)
