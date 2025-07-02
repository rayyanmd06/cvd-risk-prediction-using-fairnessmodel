import json
import joblib
import os
import matplotlib.pyplot as plt

# Load original model metrics
with open('results/metrics.json') as f:
    metrics = json.load(f)
orig_accuracy = metrics['ensemble_voting']['test_accuracy']

# Fairness metrics recorded manually after mitigation:
# You can store them in results/fairness_metrics.json if you prefer; for now:
fairness = {
    'fair': dict(
        accuracy=0.855,
        dp=0.016,
        eo=0.055
    ),
    'orig': dict(
        accuracy=orig_accuracy,
        dp=0.211,
        eo=0.100
    )
}

labels = ['Original', 'Fair']

# Accuracy plot
plt.figure()
plt.bar(labels, [fairness['orig']['accuracy'], fairness['fair']['accuracy']])
plt.ylabel('Accuracy')
plt.title('Accuracy: Original vs Fair Model')
plt.savefig('results/accuracy_tradeoff.png')
plt.close()

# Demographic Parity plot
plt.figure()
plt.bar(labels, [fairness['orig']['dp'], fairness['fair']['dp']])
plt.ylabel('Demographic Parity Difference')
plt.title('Demographic Parity: Original vs Fair Model')
plt.savefig('results/dp_tradeoff.png')
plt.close()

# Equalized Odds plot
plt.figure()
plt.bar(labels, [fairness['orig']['eo'], fairness['fair']['eo']])
plt.ylabel('Equalized Odds Difference')
plt.title('Equalized Odds: Original vs Fair Model')
plt.savefig('results/eo_tradeoff.png')
plt.close()

print("Saved plots to results/: accuracy_tradeoff.png, dp_tradeoff.png, eo_tradeoff.png")
