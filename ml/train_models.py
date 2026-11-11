"""
Train and compare churn prediction models, then export everything the
React app needs to results.json.

Pipeline
  1. Load data, split 80/20 (stratified so both sets keep the same churn rate)
  2. Preprocess: scale numeric features, one-hot encode categorical features
  3. Train Logistic Regression, Random Forest and Gradient Boosting
  4. Evaluate with 5-fold cross-validation (ROC-AUC) and on the held-out test set
  5. Explain the models: permutation feature importance + logistic regression coefficients
  6. Export metrics, ROC curves, confusion matrices, EDA summaries and the
     logistic regression parameters (so the browser can make live predictions)

Run:  python ml/train_models.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customers.csv"
OUT = ROOT / "web" / "src" / "data" / "results.json"
SEED = 42

NUMERIC = ["tenure_months", "monthly_charges", "support_calls", "senior_citizen"]
CATEGORICAL = ["contract", "internet_service", "tech_support", "payment_method", "paperless_billing"]
FEATURES = NUMERIC + CATEGORICAL
LABELS = {
    "tenure_months": "Tenure (months)", "monthly_charges": "Monthly charges",
    "support_calls": "Support calls", "senior_citizen": "Senior citizen",
    "contract": "Contract type", "internet_service": "Internet service",
    "tech_support": "Tech support", "payment_method": "Payment method",
    "paperless_billing": "Paperless billing",
}

df = pd.read_csv(DATA)
X, y = df[FEATURES], df["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                            class_weight="balanced", random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                    learning_rate=0.05, random_state=SEED),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results, fitted = [], {}

for name, clf in MODELS.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("model", clf)])
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    pipe.fit(X_train, y_train)
    fitted[name] = pipe

    proba = pipe.predict_proba(X_test)[:, 1]
    # Balanced models use 0.5; boosting is unweighted so pick the threshold matching the base rate
    threshold = 0.5 if name != "Gradient Boosting" else float(y_train.mean())
    pred = (proba >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_test, proba)
    keep = np.linspace(0, len(fpr) - 1, 60).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    results.append({
        "model": name,
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_auc_std": round(float(cv_auc.std()), 4),
        "test_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "threshold": round(threshold, 3),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc": [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in keep],
    })

best = max(results, key=lambda r: r["test_auc"])

# Permutation importance on the best model (works for any model type)
perm = permutation_importance(fitted[best["model"]], X_test, y_test, scoring="roc_auc",
                              n_repeats=10, random_state=SEED, n_jobs=-1)
importance = sorted(
    [{"feature": LABELS[f], "importance": round(float(m), 4)}
     for f, m in zip(FEATURES, perm.importances_mean)],
    key=lambda d: d["importance"], reverse=True)

# Logistic regression parameters -> live predictor in the browser
lr = fitted["Logistic Regression"]
prep, model = lr.named_steps["prep"], lr.named_steps["model"]
scaler, encoder = prep.named_transformers_["num"], prep.named_transformers_["cat"]
coefs = model.coef_[0]
predictor = {
    "intercept": float(model.intercept_[0]),
    "numeric": [{"name": n, "mean": float(m), "scale": float(s), "coef": float(c)}
                for n, m, s, c in zip(NUMERIC, scaler.mean_, scaler.scale_, coefs[:len(NUMERIC)])],
    "categorical": [],
}
i = len(NUMERIC)
for col, cats in zip(CATEGORICAL, encoder.categories_):
    levels = {str(cats[0]): 0.0}  # dropped baseline category
    for cat in cats[1:]:
        levels[str(cat)] = float(coefs[i]); i += 1
    predictor["categorical"].append({"name": col, "levels": levels})

# Simple EDA summaries for the "Insights" tab
def churn_by(col):
    g = df.groupby(col)["churn"].agg(["mean", "count"]).reset_index()
    return [{"group": str(r[col]), "churn_rate": round(float(r["mean"]), 4), "customers": int(r["count"])}
            for _, r in g.iterrows()]

df["tenure_band"] = pd.cut(df["tenure_months"], [0, 6, 12, 24, 48, 72],
                           labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
df["support_calls_band"] = pd.cut(df["support_calls"], [-1, 0, 1, 2, 3, 4, 100],
                                  labels=["0", "1", "2", "3", "4", "5+"])

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "dataset": {"customers": len(df), "churn_rate": round(float(df["churn"].mean()), 4),
                "train_size": len(X_train), "test_size": len(X_test)},
    "models": results,
    "best_model": best["model"],
    "feature_importance": importance,
    "eda": {
        "contract": churn_by("contract"),
        "tenure_band": churn_by("tenure_band"),
        "internet_service": churn_by("internet_service"),
        "payment_method": churn_by("payment_method"),
        "support_calls": churn_by("support_calls_band"),
    },
    "predictor": predictor,
}, indent=1))

print(f"Customers: {len(df):,}  churn rate: {df.churn.mean():.1%}\n")
print(f"{'Model':22s} {'CV AUC':>12s} {'Test AUC':>9s} {'Acc':>6s} {'Prec':>6s} {'Recall':>7s} {'F1':>6s}")
for r in results:
    print(f"{r['model']:22s} {r['cv_auc_mean']:.3f}±{r['cv_auc_std']:.3f} {r['test_auc']:9.3f} "
          f"{r['accuracy']:6.3f} {r['precision']:6.3f} {r['recall']:7.3f} {r['f1']:6.3f}")
print(f"\nBest model: {best['model']}")
print("Top drivers:", ", ".join(d["feature"] for d in importance[:4]))
print(f"Wrote {OUT}")
