import os
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)
from sklearn.metrics import accuracy_score

def load_and_preprocess_cleveland(data_dir):
    path = os.path.join(data_dir, 'heart_cleveland.csv')
    df = pd.read_csv(path, header=0, na_values='?')

    # Impute missing with median
    imp = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imp.fit_transform(df), columns=df.columns)

    # Binarize target
    df_imputed['target'] = df_imputed['target'].apply(lambda v: 1 if v > 0 else 0)
    return df_imputed

def compute_fairness(data_dir='data', results_dir='results', test_size=0.2, random_state=42):
    # Load & preprocess
    df = load_and_preprocess_cleveland(data_dir)

    # Split features, target, protected attr
    X = df.drop(columns=['target'])
    y = df['target'].values
    A = df['sex'].astype(int).values

    # Train/test split
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A, test_size=test_size, stratify=y, random_state=random_state
    )

    # Load scaler & final model
    scaler = joblib.load(os.path.join(results_dir, 'scaler.pkl'))
    model  = joblib.load(os.path.join(results_dir, 'final_model.pkl'))

    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Overall accuracy
    overall_acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall accuracy (Cleveland test set): {overall_acc:.3f}\n")

    # Per‑group accuracy
    for group in np.unique(A_test):
        mask = (A_test == group)
        acc = accuracy_score(y_test[mask], y_pred[mask])
        print(f"Accuracy for sex={group}: {acc:.3f}")

    # Fairness metrics
    dp = demographic_parity_difference(y_test, y_pred, sensitive_features=A_test)
    eo = equalized_odds_difference(y_test, y_pred, sensitive_features=A_test)

    print("\nFairness metrics:")
    print(f"  Demographic parity difference: {dp:.3f}")
    print(f"  Equalized odds difference:      {eo:.3f}\n")

if __name__ == '__main__':
    compute_fairness()
