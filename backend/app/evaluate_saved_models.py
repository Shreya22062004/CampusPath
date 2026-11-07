"""Evaluate existing app artifacts on the research pipeline's held-out test split.

Run from the project root with the project environment active:
    python -m backend.app.evaluate_saved_models

This loads the saved 25 pipelines and does not fit or overwrite any model.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split

from experiments.run_experiments import SEED, TARGET, load_data

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "campus_placement_data.csv"
ARTIFACTS = ROOT / "backend" / "artifacts"
RESULTS = ROOT / "backend" / "results"


def main():
    meta = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    df = load_data(DATA_PATH)
    X, y = df.drop(columns=[TARGET]), df[TARGET].astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    y_test = np.asarray(y_test)
    rows, matrices, roc_curves = [], {}, {}
    for feature_set, columns in meta["feature_sets"].items():
        for model_name in meta["models"]:
            path = ARTIFACTS / f"{model_name.lower()}__{feature_set.lower().replace('-', '_')}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Missing trained model artifact: {path}")
            model = joblib.load(path)
            proba = model.predict_proba(X_test[columns])[:, list(model.classes_).index(meta["positive_class"])]
            pred = (proba >= 0.5).astype(int)
            cm = confusion_matrix(y_test, pred, labels=[0, 1])
            row = {
                "model": model_name,
                "feature_set": feature_set,
                "n_features": len(columns),
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, proba)),
            }
            key = f"{model_name}|{feature_set}"
            rows.append(row)
            matrices[key] = cm.tolist()
            fpr, tpr, _ = roc_curve(y_test, proba)
            # Keep ROC API payload small while retaining the complete curve shape.
            ix = np.unique(np.linspace(0, len(fpr) - 1, min(201, len(fpr))).round().astype(int))
            roc_curves[key] = {"fpr": fpr[ix].tolist(), "tpr": tpr[ix].tolist(), "auc": row["roc_auc"]}

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "saved_model_metrics.csv", index=False)
    report = {
        "dataset_records": int(len(df)),
        "test_records": int(len(y_test)),
        "class_counts": {"not_placed": int((y == 0).sum()), "placed": int((y == 1).sum())},
        "evaluation": "Existing saved pipelines evaluated on the stratified 20% holdout (random_state=42); threshold=0.5.",
        "metrics": rows,
        "confusion_matrices": matrices,
        "roc_curves": roc_curves,
    }
    (RESULTS / "saved_model_analytics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluated {len(rows)} saved model configurations on {len(y_test):,} held-out records.")
    print(f"Best ROC-AUC: {max(rows, key=lambda item: item['roc_auc'])}")
    print(f"Saved analytics to {RESULTS}")


if __name__ == "__main__":
    main()
