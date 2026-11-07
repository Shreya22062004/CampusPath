"""Train and save models for the web application.

Run from the project root:
    python -m backend.app.train_models --sample 20000 --folds 3
Omit --sample to train on the complete dataset.
"""
import argparse
import json
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.run_experiments import (
    TARGET, SEED, SUBSETS, load_data, shap_ranking, get_models,
    build_pipeline
)
import experiments.run_experiments as research

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "campus_placement_data.csv"
ARTIFACTS = ROOT / "backend" / "artifacts"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    work_results = ROOT / "backend" / "results"
    work_results.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data, args.sample)
    X, y = df.drop(columns=[TARGET]), df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    research.POS_WEIGHT = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    ranking = shap_ranking(X_train, y_train, str(work_results))
    pd.DataFrame({"feature": ranking, "rank": range(1, len(ranking)+1)}).to_csv(
        ARTIFACTS / "feature_ranking.csv", index=False
    )

    # Save all model/feature-set pipelines for the interactive prediction page.
    model_names = list(get_models().keys())
    feature_sets = {f"Top-{k}": ranking[:k] for k in [3, 5, 7, 10]}
    feature_sets["All"] = list(X_train.columns)
    metadata = {"target": TARGET, "models": model_names, "feature_sets": {},
                "input_columns": list(X_train.columns), "positive_class": 1}
    for set_name, columns in feature_sets.items():
        metadata["feature_sets"][set_name] = columns
        for model_name in model_names:
            print(f"Training app model: {model_name} / {set_name}")
            pipeline = build_pipeline(model_name, columns, X_train)
            pipeline.fit(X_train[columns], y_train)
            safe_model = model_name.lower()
            safe_set = set_name.lower().replace("-", "_")
            joblib.dump(pipeline, ARTIFACTS / f"{safe_model}__{safe_set}.joblib")
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved model artifacts to: {ARTIFACTS}")
    # Build the saved held-out evaluation payload consumed by the dashboard API.
    # The evaluator reloads these pipelines and uses the same data/sample split.
    from backend.app.evaluate_saved_models import main as evaluate_saved_models
    evaluate_saved_models(data_path=args.data, sample=args.sample)
    print("Next: start the API with: uvicorn backend.app.main:app --reload")

if __name__ == "__main__":
    main()
