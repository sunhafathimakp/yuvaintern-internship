from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / "week4_figures"
FIGURE_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "Telco-Customer-Churn.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    if "Churn" not in data.columns:
        raise ValueError("The dataset must contain a 'Churn' target column.")
    return data


def prepare_data(data: pd.DataFrame):
    data = data.drop_duplicates().copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    zero_tenure = data["tenure"].eq(0)
    data.loc[zero_tenure & data["TotalCharges"].isna(), "TotalCharges"] = 0

    y = data["Churn"].astype(str).str.strip().map({"No": 0, "Yes": 1})
    valid = y.notna()
    y = y.loc[valid].astype(int)
    X = data.loc[valid].drop(columns=["Churn", "customerID"], errors="ignore")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return X, y, preprocessor


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, random_state: int = 42):
    """Create a repeatable holdout split with class proportions preserved."""
    rng = np.random.default_rng(random_state)
    train_parts, test_parts = [], []
    for label in sorted(y.unique()):
        positions = np.flatnonzero(y.to_numpy() == label)
        rng.shuffle(positions)
        n_test = int(np.ceil(test_size * len(positions)))
        test_parts.extend(positions[:n_test])
        train_parts.extend(positions[n_test:])
    train_positions = np.asarray(train_parts)
    test_positions = np.asarray(test_parts)
    rng.shuffle(train_positions)
    rng.shuffle(test_positions)
    return (
        X.iloc[train_positions], X.iloc[test_positions],
        y.iloc[train_positions], y.iloc[test_positions],
    )


def main() -> None:
    data = load_data()
    X, y, preprocessor = prepare_data(data)
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_score)
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"Rows used: {len(X)}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Churn prevalence: {y.mean():.3%}")
    print("\nTest-set metrics")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nClassification report")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"], zero_division=0))

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ConfusionMatrixDisplay(matrix, display_labels=["Stayed", "Churned"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("Logistic Regression: Test-Set Confusion Matrix")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(fpr, tpr, color="#1769aa", linewidth=2.5, label=f"Logistic Regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", label="Chance (AUC = 0.500)")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve on the Held-Out Test Set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_roc_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metrics = pd.DataFrame(
        [{"metric": "accuracy", "value": accuracy}, {"metric": "precision", "value": precision},
         {"metric": "recall", "value": recall}, {"metric": "f1", "value": f1},
         {"metric": "roc_auc", "value": auc}]
    )
    metrics.to_csv(BASE_DIR / "week4_metrics.csv", index=False)
    print(f"\nCharts saved in: {FIGURE_DIR}")
    print(f"Metrics saved to: {BASE_DIR / 'week4_metrics.csv'}")


if __name__ == "__main__":
    main()
