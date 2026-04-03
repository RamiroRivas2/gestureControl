"""
Train a Random Forest classifier on collected hand landmark data.

Reads .npy files from pi/data/ directory, trains a classifier,
and saves the model to pi/model/gesture_classifier.pkl
"""

import numpy as np
import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    config = load_config()
    gesture_names = list(config["gestures"].keys())
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)

    X = []
    y = []
    label_map = {}

    print("=== Loading Training Data ===\n")

    for idx, gesture in enumerate(gesture_names):
        filepath = os.path.join(data_dir, f"{gesture}.npy")
        if not os.path.exists(filepath):
            print(f"  WARNING: No data for '{gesture}' - skipping")
            continue

        samples = np.load(filepath)
        print(f"  {gesture}: {len(samples)} samples")

        X.extend(samples)
        y.extend([idx] * len(samples))
        label_map[idx] = gesture

    if not X:
        print("\nERROR: No training data found. Run collect_data.py first.")
        return

    X = np.array(X)
    y = np.array(y)

    print(f"\nTotal samples: {len(X)}")
    print(f"Gesture classes: {len(label_map)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Train Random Forest
    print("\n=== Training Random Forest ===\n")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    accuracy = clf.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.1%}")

    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"Cross-Val Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std() * 2:.1%})")

    # Detailed report
    y_pred = clf.predict(X_test)
    target_names = [label_map[i] for i in sorted(label_map.keys())]
    print(f"\n=== Classification Report ===\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("=== Confusion Matrix ===\n")
    cm = confusion_matrix(y_test, y_pred)
    # Print header
    header = "          " + " ".join(f"{name[:8]:>8}" for name in target_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{val:>8}" for val in row)
        print(f"{target_names[i][:10]:>10} {row_str}")

    # Save model and label map
    model_path = os.path.join(model_dir, "gesture_classifier.pkl")
    label_map_path = os.path.join(model_dir, "label_map.json")

    joblib.dump(clf, model_path)
    with open(label_map_path, "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)

    print(f"\nModel saved to: {model_path}")
    print(f"Label map saved to: {label_map_path}")
    print("\nDone! You can now run gesture_control.py")


if __name__ == "__main__":
    main()
