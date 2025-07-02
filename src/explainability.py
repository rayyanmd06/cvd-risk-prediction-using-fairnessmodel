import shap
import matplotlib.pyplot as plt


def shap_summary_plot(model, X, feature_names, output_path):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.savefig(output_path)
    plt.close()


def shap_individual_plot(model, X, index, feature_names, output_path):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    plt.figure()
    shap.plots.waterfall(shap_values[index], show=False)
    plt.savefig(output_path)
    plt.close()

import shap

def explain_model(model, X_train, output_path):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(output_path)
