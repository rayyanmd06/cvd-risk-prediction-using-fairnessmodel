import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(data_dir):
    clev_path = os.path.join(data_dir, 'heart_cleveland.csv')
    fram_path = os.path.join(data_dir, 'framingham.csv')
    clev = pd.read_csv(clev_path)
    fram = pd.read_csv(fram_path)
    return clev, fram

def clean_impute(df):
    df = df.replace('?', np.nan).apply(pd.to_numeric, errors='coerce')
    imp = SimpleImputer(strategy='median')
    return pd.DataFrame(imp.fit_transform(df), columns=df.columns)

def harmonize_features(cleveland, framingham):

    framingham = framingham.rename(columns={
        'male': 'sex',
        'totChol': 'chol',
        'sysBP': 'trestbps',
        'diaBP': 'thalach',    # proxy mapping
        'CVD': 'target'
    })
    cols = [
        'age','sex','cp','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal','target'
    ]

    for c in cols:
        if c not in framingham.columns:
            framingham[c] = np.nan
    return cleveland[cols], framingham[cols]

def preprocess(data_dir, test_size=0.2, random_state=42):
    # load
    cleveland, framingham = load_data(data_dir)

    # clean & per‑dataset impute (existing columns)
    cleveland = clean_impute(cleveland)
    framingham = clean_impute(framingham)

    # harmonize schemas, add missing cols as NaN
    clev, fram = harmonize_features(cleveland, framingham)

    # combine
    df = pd.concat([clev, fram], ignore_index=True)

    # IMPUTE AGAIN TO FILL ALL NEW NaNs
    df = clean_impute(df)

    # split features/target & binarize
    X = df.drop(columns=['target'])
    y = df['target'].apply(lambda v: 1 if v > 0 else 0).astype(int)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # SMOTE 
    sm = SMOTE(random_state=random_state)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler
