import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "dating_dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Convert the multi-class product outcome into a binary "good match" label.
    We treat outcomes that indicate positive progression as class 1.
    """
    good_outcomes = {
        "Mutual Match",
        "Date Happened",
        "Relationship Formed",
        "Instant Match",
    }
    return df["match_outcome"].isin(good_outcomes).astype(int)


def parse_interest_count(series: pd.Series) -> pd.Series:
    return series.fillna("").apply(lambda x: len([item.strip() for item in str(x).split(",") if item.strip()]))


def load_and_prepare() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)

    # Basic feature engineering from raw fields
    df["interest_count"] = parse_interest_count(df["interest_tags"])
    df["engagement_score"] = (
        (0.30 * df["likes_received"])
        + (0.45 * df["mutual_matches"])
        + (0.25 * df["message_sent_count"])
    )
    df["activity_score"] = df["app_usage_time_min"] * df["swipe_right_ratio"]
    df["profile_completeness"] = (
        (df["profile_pics_count"] / 6.0) * 0.4 + (df["bio_length"] / 500.0) * 0.6
    ).clip(0, 1)

    y = build_target(df)
    X = df.drop(columns=["match_outcome", "interest_tags", "swipe_time_of_day"])
    return X, y


def train() -> None:
    X, y = load_and_prepare()

    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1_score": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate_test": float(np.mean(y_test)),
    }

    sample_idx = X_test.index[:2]
    sample_rows = X_test.loc[sample_idx].copy()
    sample_prob = pipeline.predict_proba(sample_rows)[:, 1]
    sample_pred = pipeline.predict(sample_rows)
    sample_output = []
    for i, row_idx in enumerate(sample_idx):
        sample_output.append(
            {
                "row_id": int(row_idx),
                "predicted_label": int(sample_pred[i]),
                "confidence": float(sample_prob[i]),
                "input_profile": sample_rows.iloc[i].to_dict(),
            }
        )

    joblib.dump(pipeline, ARTIFACT_DIR / "match_pipeline.pkl")
    joblib.dump(X.columns.tolist(), ARTIFACT_DIR / "feature_columns.pkl")

    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(ARTIFACT_DIR / "sample_predictions.json", "w", encoding="utf-8") as f:
        json.dump(sample_output, f, indent=2, default=str)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
