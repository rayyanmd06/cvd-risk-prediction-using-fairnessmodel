import os
import joblib
import pandas as pd

from src.data_preprocessing import preprocess
from src.model_training import train_and_evaluate
from src.utils import ensure_dir, save_results
from src.explainability import shap_summary_plot, shap_individual_plot

if __name__ == '__main__':
    BASE_DIR    = os.path.dirname(__file__)
    DATA_DIR    = os.path.join(BASE_DIR, 'data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    ensure_dir(RESULTS_DIR)

    try:
        # Preprocess
        print("1) Preprocessing data...")
        X_tr, y_tr, X_te, y_te, scaler = preprocess(DATA_DIR)
        print("   ✔ Preprocessing done.")

        # Save the scaler
        scaler_path = os.path.join(RESULTS_DIR, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"   ✔ Scaler saved → {scaler_path}")

        # Train & evaluate
        print("2) Training & evaluating models...")
        metrics = train_and_evaluate(X_tr, y_tr, X_te, y_te, RESULTS_DIR)
        print("   ✔ Training & evaluation done.")

        # Save metrics
        metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
        save_results(metrics, metrics_path)
        print(f"3) Saved metrics → {metrics_path}")

        # Save final model
        final_model_name = 'ensemble_voting' if 'ensemble_voting' in metrics else max(
            metrics, key=lambda k: metrics[k]['test_roc_auc']
        )
        final_model_src = os.path.join(RESULTS_DIR, f"{final_model_name}_model.pkl")
        final_model = joblib.load(final_model_src)
        final_model_dst = os.path.join(RESULTS_DIR, 'final_model.pkl')
        joblib.dump(final_model, final_model_dst)
        print(f"4) Final model ({final_model_name}) saved → {final_model_dst}")

        # SHAP Explainability using XGBoost
        print("5) Generating SHAP explainability plots using XGBoost...")

        # Get feature names (exclude target)
        raw = pd.read_csv(os.path.join(DATA_DIR, 'heart_cleveland.csv'), nrows=0)
        feature_names = list(raw.columns[:-1])

        # Load XGBoost model explicitly
        shap_model_name = 'xgboost'
        shap_model_path = os.path.join(RESULTS_DIR, f'{shap_model_name}_model.pkl')
        shap_model = joblib.load(shap_model_path)
        print(f"   ↪ Explaining model: {shap_model_name}")

        # Summary plot
        shap_summary_plot(
            model=shap_model,
            X=X_te,
            feature_names=feature_names,
            output_path=os.path.join(RESULTS_DIR, 'shap_summary.png')
        )
        print("   ✔ SHAP summary plot saved → shap_summary.png")

        # Individual waterfall plot for the first test instance
        shap_individual_plot(
            model=shap_model,
            X=X_te,
            index=0,
            feature_names=feature_names,
            output_path=os.path.join(RESULTS_DIR, 'shap_waterfall_0.png')
        )
        print("   ✔ SHAP individual plot saved → shap_waterfall_0.png")

        # All done
        print("Pipeline completed successfully with explainability.")
        print("Summary metrics:", metrics)

    except Exception as e:
        print("Pipeline failed with error:")
        print(e)
