import os
import json
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
METRICS_PATH = os.path.join(RESULTS_DIR, 'metrics.json')

def plot_metric(metric_name, metrics, save_dir):
    models = list(metrics.keys())
    values = [metrics[model][metric_name] for model in models]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, values, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} Comparison')
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    output_path = os.path.join(save_dir, f'{metric_name}_comparison.png')
    plt.savefig(output_path)
    print(f"✔ Saved {metric_name} chart → {output_path}")
    plt.close()

def main():
    print("📊 Visualizing metrics...")

    if not os.path.exists(METRICS_PATH):
        print(f"❌ Error: metrics.json not found at {METRICS_PATH}")
        return

    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)

    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        plot_metric(metric, metrics, RESULTS_DIR)

    print("✅ All visualizations saved.")

if __name__ == '__main__':
    main()
