import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

def get_models():
    return {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(random_state=42),  
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'mlp': MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500, random_state=42)
    }

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from sklearn.ensemble import VotingClassifier

def train_and_evaluate(X_train, y_train, X_test, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    trained_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in get_models().items():
        print(f"\n🔍 Training {name}...")

        if name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8]
            }
            grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=cv, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"✅ Best params for random_forest: {grid_search.best_params_}")
        else:
            model.fit(X_train, y_train)

        trained_models[name] = model  # Keep model for ensemble

        # Cross-validation scores
        cv_scores = {
            'cv_accuracy': cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv).mean(),
            'cv_precision': cross_val_score(model, X_train, y_train, scoring='precision', cv=cv).mean(),
            'cv_recall': cross_val_score(model, X_train, y_train, scoring='recall', cv=cv).mean(),
            'cv_f1_score': cross_val_score(model, X_train, y_train, scoring='f1', cv=cv).mean(),
            'cv_roc_auc': cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv).mean()
        }

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        test_scores = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred),
            'test_f1_score': f1_score(y_test, y_pred),
            'test_roc_auc': roc_auc_score(y_test, y_prob)
        }

        results[name] = {**cv_scores, **test_scores}

        # Save individual model
        joblib.dump(model, os.path.join(output_dir, f'{name}_model.pkl'))

    # Ensemble Model
    print("\n🤖 Training VotingClassifier Ensemble...")

    ensemble = VotingClassifier(
        estimators=[
            ('lr', trained_models['logistic_regression']),
            ('rf', trained_models['random_forest']),
            ('xgb', trained_models['xgboost'])
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]

    ensemble_metrics = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_f1_score': f1_score(y_test, y_pred),
        'test_roc_auc': roc_auc_score(y_test, y_prob)
    }

    results['ensemble_voting'] = ensemble_metrics
    joblib.dump(ensemble, os.path.join(output_dir, 'ensemble_voting_model.pkl'))
    print("✅ Ensemble model trained and saved.")

    return results
