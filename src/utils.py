import json
import os

def save_results(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
