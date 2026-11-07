"""
SHAP-Guided Feature Ranking for Student Placement Prediction
------------------------------------------------------------
Pipeline (matches the proposal workflow):
  1. Load + clean data (drop leakage columns)
  2. Stratified train/test split (test set untouched until the end)
  3. Initial model (XGBoost) on TRAIN ONLY -> SHAP feature ranking
  4. Build Top-3 / 5 / 7 / 10 / All feature subsets
  5. Train 5 models x 5 feature sets (25 combos): CV on train + final test
  6. Save tables, plots, confusion matrices, significance tests

Usage:
    python run_experiments.py --data campus_placement_data.csv --out results
    python run_experiments.py --data campus_placement_data.csv --out results_quick --sample 20000 --folds 3
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

TARGET = "placed"
# salary_lpa exists only for placed students -> target leakage. student_id is just an identifier.
LEAK_OR_ID = ["salary_lpa", "student_id"]
SUBSETS = [3, 5, 7, 10, "All"]
POS_WEIGHT = 1.0  # set in main() = n_negative / n_positive (matches class_weight="balanced")


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
def load_data(path, sample=None):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")

    # 'specialization' is NaN when the student has no MBA -> make it an explicit category
    df["specialization"] = df["specialization"].fillna("No MBA")

    df = df.drop(columns=LEAK_OR_ID, errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)

    if sample and sample < len(df):
        df, _ = train_test_split(df, train_size=sample, stratify=df[TARGET], random_state=SEED)
        df = df.reset_index(drop=True)
        print(f"Using stratified sample of {len(df):,} rows")

    print("Class balance:\n", df[TARGET].value_counts(normalize=True).round(3).to_string())
    return df


def eda_plots(df, out):
    num = df.select_dtypes("number")
    plt.figure(figsize=(12, 9))
    sns.heatmap(num.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation matrix (numeric features)")
    plt.tight_layout()
    plt.savefig(f"{out}/eda_correlation.png", dpi=150)
    plt.close()

    corr_target = num.corr()[TARGET].drop(TARGET).sort_values()
    plt.figure(figsize=(7, 7))
    corr_target.plot(kind="barh")
    plt.title("Correlation with placement")
    plt.tight_layout()
    plt.savefig(f"{out}/eda_corr_with_target.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# 2. Preprocessing helpers
# --------------------------------------------------------------------------- #
def make_preprocessor(cols, X, scale):
    cat = [c for c in cols if X[c].dtype == "object" or str(X[c].dtype).startswith(("str", "category"))]
    num = [c for c in cols if c not in cat]
    num_tf = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("num", num_tf, num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)]
    )


def get_models():
    return {
        "LogisticRegression": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED), True),
        "RandomForest": (RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                n_jobs=-1, random_state=SEED), False),
        "XGBoost": (XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
                                  colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
                                  scale_pos_weight=POS_WEIGHT, random_state=SEED, verbosity=0), False),
        "LightGBM": (LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.08, subsample=0.8,
                                    colsample_bytree=0.8, class_weight="balanced", n_jobs=-1,
                                    random_state=SEED, verbose=-1), False),
        "CatBoost": (CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08, auto_class_weights="Balanced",
                                        random_seed=SEED, verbose=0, thread_count=-1), False),
    }


def build_pipeline(model_name, cols, X):
    model, scale = get_models()[model_name]
    return Pipeline([("prep", make_preprocessor(cols, X, scale)), ("clf", model)])


# --------------------------------------------------------------------------- #
# 3. SHAP ranking (TRAIN ONLY -> no leakage into the test set)
# --------------------------------------------------------------------------- #
def shap_ranking(X_train, y_train, out, shap_sample=5000):
    cols = list(X_train.columns)
    pipe = Pipeline([
        ("prep", make_preprocessor(cols, X_train, scale=False)),
        ("clf", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
                              colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
                              random_state=SEED, verbosity=0)),
    ]).fit(X_train, y_train)

    Xs = X_train.sample(min(shap_sample, len(X_train)), random_state=SEED)
    Xt = pipe.named_steps["prep"].transform(Xs)
    names = pipe.named_steps["prep"].get_feature_names_out()
    sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xt)

    # Aggregate one-hot columns back to their parent feature so we rank ORIGINAL features
    abs_mean = np.abs(sv).mean(axis=0)
    parent = [n.split("__", 1)[1] for n in names]
    parent = [next((c for c in cols if p == c or p.startswith(c + "_")), p) for p in parent]
    rank = (pd.DataFrame({"feature": parent, "mean_abs_shap": abs_mean})
            .groupby("feature", as_index=False).sum()
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    rank["rank"] = rank.index + 1
    rank.to_csv(f"{out}/shap_feature_ranking.csv", index=False)

    plt.figure(figsize=(8, 8))
    plt.barh(rank["feature"][::-1], rank["mean_abs_shap"][::-1])
    plt.xlabel("Mean |SHAP value| (summed over one-hot columns)")
    plt.title("SHAP feature importance (train data)")
    plt.tight_layout()
    plt.savefig(f"{out}/shap_bar.png", dpi=150)
    plt.close()

    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"{out}/shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSHAP ranking (top 10):\n", rank.head(10).to_string(index=False))
    return rank["feature"].tolist()


# --------------------------------------------------------------------------- #
# 4. Evaluation
# --------------------------------------------------------------------------- #
def metrics(y, pred, proba):
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, proba),
    }


def run_experiments(X_train, X_test, y_train, y_test, ranking, folds, out):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows, fold_rows, cms = [], [], {}
    model_names = list(get_models().keys())

    for k in SUBSETS:
        cols = ranking if k == "All" else ranking[:k]
        label = "All" if k == "All" else f"Top-{k}"
        for mname in model_names:
            print(f"  [{label:6s}] {mname}")
            # ---- cross-validation on TRAIN ----
            cv_scores = []
            for f, (tr, va) in enumerate(skf.split(X_train, y_train)):
                pipe = build_pipeline(mname, cols, X_train)
                pipe.fit(X_train.iloc[tr][cols], y_train.iloc[tr])
                proba = pipe.predict_proba(X_train.iloc[va][cols])[:, 1]
                m = metrics(y_train.iloc[va], (proba >= 0.5).astype(int), proba)
                cv_scores.append(m)
                fold_rows.append({"feature_set": label, "model": mname, "fold": f, **m})
            cv = pd.DataFrame(cv_scores)

            # ---- final fit on all TRAIN, evaluate once on TEST ----
            pipe = build_pipeline(mname, cols, X_train)
            pipe.fit(X_train[cols], y_train)
            proba = pipe.predict_proba(X_test[cols])[:, 1]
            pred = (proba >= 0.5).astype(int)
            t = metrics(y_test, pred, proba)
            cms[(label, mname)] = confusion_matrix(y_test, pred)

            row = {"feature_set": label, "n_features": len(cols), "model": mname}
            for met in t:
                row[f"CV_{met}_mean"] = cv[met].mean()
                row[f"CV_{met}_std"] = cv[met].std()
                row[f"Test_{met}"] = t[met]
            rows.append(row)

    res = pd.DataFrame(rows)
    folds_df = pd.DataFrame(fold_rows)
    res.to_csv(f"{out}/results_all_experiments.csv", index=False)
    folds_df.to_csv(f"{out}/results_cv_folds.csv", index=False)
    return res, folds_df, cms


# --------------------------------------------------------------------------- #
# 5. Reporting
# --------------------------------------------------------------------------- #
def make_reports(res, folds_df, cms, out):
    order = [f"Top-{k}" if k != "All" else "All" for k in SUBSETS]

    # Test-set pivot tables per metric
    for met in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
        piv = res.pivot(index="model", columns="feature_set", values=f"Test_{met}")[order]
        piv.round(4).to_csv(f"{out}/table_test_{met}.csv")

    # Performance vs number of features
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, met in zip(axes, ["F1", "ROC_AUC"]):
        for m, g in res.groupby("model"):
            g = g.set_index("feature_set").loc[order]
            ax.errorbar(order, g[f"CV_{met}_mean"], yerr=g[f"CV_{met}_std"], marker="o", capsize=3, label=m)
        ax.set_title(f"CV {met} vs feature subset")
        ax.set_xlabel("Feature set")
        ax.set_ylabel(met)
        ax.grid(alpha=.3)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(f"{out}/performance_vs_features.png", dpi=150)
    plt.close()

    # Heatmap of test ROC-AUC
    piv = res.pivot(index="model", columns="feature_set", values="Test_ROC_AUC")[order]
    plt.figure(figsize=(8, 4))
    sns.heatmap(piv, annot=True, fmt=".4f", cmap="viridis")
    plt.title("Test ROC-AUC: model x feature set")
    plt.tight_layout()
    plt.savefig(f"{out}/heatmap_test_roc_auc.png", dpi=150)
    plt.close()

    # Confusion matrices (grid)
    models = res["model"].unique()
    fig, axes = plt.subplots(len(models), len(order), figsize=(3 * len(order), 2.8 * len(models)))
    for i, m in enumerate(models):
        for j, s in enumerate(order):
            sns.heatmap(cms[(s, m)], annot=True, fmt="d", cbar=False, cmap="Blues", ax=axes[i, j],
                        xticklabels=["Not placed", "Placed"], yticklabels=["Not placed", "Placed"])
            axes[i, j].set_title(f"{m} | {s}", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out}/confusion_matrices.png", dpi=130)
    plt.close()

    # Paired Wilcoxon: each subset vs All, per model, on CV-fold ROC-AUC
    sig = []
    for m in models:
        base = folds_df[(folds_df.model == m) & (folds_df.feature_set == "All")].sort_values("fold")["ROC_AUC"].values
        for s in order[:-1]:
            alt = folds_df[(folds_df.model == m) & (folds_df.feature_set == s)].sort_values("fold")["ROC_AUC"].values
            try:
                p = wilcoxon(alt, base).pvalue
            except ValueError:
                p = np.nan
            sig.append({"model": m, "subset": s, "mean_AUC_subset": alt.mean(), "mean_AUC_all": base.mean(),
                        "diff": alt.mean() - base.mean(), "wilcoxon_p": p})
    pd.DataFrame(sig).to_csv(f"{out}/significance_vs_all.csv", index=False)

    print("\n=== Test ROC-AUC ===")
    print(piv.round(4).to_string())
    print("\n=== Test F1 ===")
    print(res.pivot(index="model", columns="feature_set", values="Test_F1")[order].round(4).to_string())


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--sample", type=int, default=None, help="stratified subsample size for quick runs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test_size", type=float, default=0.2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load_data(a.data, a.sample)
    eda_plots(df, a.out)

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=a.test_size, stratify=y, random_state=SEED)
    X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
    y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

    global POS_WEIGHT
    POS_WEIGHT = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"XGBoost scale_pos_weight = {POS_WEIGHT:.3f}")

    ranking = shap_ranking(X_train, y_train, a.out)
    pd.Series(ranking, name="feature").to_csv(f"{a.out}/ranking_order.csv", index_label="rank")

    res, folds_df, cms = run_experiments(X_train, X_test, y_train, y_test, ranking, a.folds, a.out)
    make_reports(res, folds_df, cms, a.out)
    print(f"\nDone. Outputs in ./{a.out}/")

if __name__ == "__main__":
    main()
