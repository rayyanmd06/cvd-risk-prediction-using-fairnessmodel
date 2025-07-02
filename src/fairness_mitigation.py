import os
import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import load_data, clean_impute, harmonize_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def load_and_prepare_for_fairness(data_dir, results_dir, test_size=0.2, random_state=42):
    # Load raw
    clev_raw, fram_raw = load_data(data_dir)

    # Clean & impute separately
    clev = clean_impute(clev_raw)
    fram = clean_impute(fram_raw)

    # Harmonize
    clev, fram = harmonize_features(clev, fram)

    # Merge then impute leftover NaNs
    df = pd.concat([clev, fram], ignore_index=True)
    df = clean_impute(df)

    # Extract X, y, sensitive
    X = df.drop(columns=['target']).values
    y = (df['target'] > 0).astype(int).values
    A = df['sex'].astype(int).values

    # Split
    X_tr, X_te, y_tr, y_te, A_tr, A_te = train_test_split(
        X, y, A,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Scale
    scaler = joblib.load(os.path.join(results_dir, 'scaler.pkl'))
    X_tr_scaled = scaler.transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    return X_tr_scaled, X_te_scaled, y_tr, y_te, A_tr, A_te

def mitigate_demographic_parity(data_dir='data', results_dir='results'):
    print("Loading and preparing data for fairness (no SMOTE)…")
    X_tr, X_te, y_tr, y_te, A_tr, A_te = load_and_prepare_for_fairness(data_dir, results_dir)

    # Base learner
    base = LogisticRegression(solver='liblinear')

    # ExponentiatedGradient for demographic parity
    mitigator = ExponentiatedGradient(
        estimator=base,
        constraints=DemographicParity(),
        eps=0.02
    )

    print("Training fairness‑constrained model…")
    mitigator.fit(
        X_tr, y_tr,
        sensitive_features=A_tr
    )

    # Predict discrete labels on the test set
    y_pred = mitigator.predict(X_te)

    # Evaluate performance
    print("\n=== Fair Model Performance ===")
    print(f"Accuracy : {accuracy_score(y_te, y_pred):.3f}")
    print(f"Precision: {precision_score(y_te, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_te, y_pred):.3f}")
    print(f"F1 score : {f1_score(y_te, y_pred):.3f}")
    # Compute AUC on discrete predictions
    print(f"ROC AUC  : {roc_auc_score(y_te, y_pred):.3f}")

    # Fairness metrics
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference
    )
    dp = demographic_parity_difference(y_te, y_pred, sensitive_features=A_te)
    eo = equalized_odds_difference(y_te, y_pred, sensitive_features=A_te)

    print("\n=== Fairness Metrics ===")
    print(f"Demographic parity difference: {dp:.3f}")
    print(f"Equalized odds difference   : {eo:.3f}")

    # Save the fairness‑constrained model
    fair_model_path = os.path.join(results_dir, 'fair_dp_model.pkl')
    joblib.dump(mitigator, fair_model_path)
    print(f"\nSaved fair model → {fair_model_path}")

    # Overwrite the “final_model.pkl” with this fair model
    final_model_path = os.path.join(results_dir, 'final_model.pkl')
    joblib.dump(mitigator, final_model_path)
    print(f"✅ Fair model saved as final_model.pkl → {final_model_path}")

    # Re‐save the scaler (unchanged, but ensures inference picks it up)
    scaler_src = joblib.load(os.path.join(results_dir, 'scaler.pkl'))
    joblib.dump(scaler_src, os.path.join(results_dir, 'scaler.pkl'))
    print("✅ Scaler re‑saved as scaler.pkl")

if __name__ == '__main__':
    mitigate_demographic_parity()
