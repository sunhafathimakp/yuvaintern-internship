"""Reproduce the Week 5 churn analyses from the supplied Telco CSV.

Install: pip install -r requirements.txt
Run:     python week5_comprehensive_analysis.py
Optional: python week5_comprehensive_analysis.py path/to/Telco-Customer-Churn.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
FIGURES = BASE / "week5_figures"
FIGURES.mkdir(exist_ok=True)


def load_and_clean() -> pd.DataFrame:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "Telco-Customer-Churn.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    zero_tenure = df["tenure"].eq(0)
    df.loc[zero_tenure & df["TotalCharges"].isna(), "TotalCharges"] = 0
    return df.drop_duplicates().copy()


def stratified_split(X, y, test_size: float = 0.20, random_state: int = 42):
    """Reproducible class-preserving holdout used in the Week 4 model."""
    rng = np.random.default_rng(random_state)
    train_positions, test_positions = [], []
    for label in sorted(y.unique()):
        positions = np.flatnonzero(y.to_numpy() == label)
        rng.shuffle(positions)
        n_test = int(np.ceil(test_size * len(positions)))
        test_positions.extend(positions[:n_test])
        train_positions.extend(positions[n_test:])
    train_positions = np.asarray(train_positions)
    test_positions = np.asarray(test_positions)
    rng.shuffle(train_positions)
    rng.shuffle(test_positions)
    return (
        X.iloc[train_positions], X.iloc[test_positions],
        y.iloc[train_positions], y.iloc[test_positions],
    )


def statistical_analysis(df: pd.DataFrame) -> dict:
    churn = df["Churn"].astype(str).str.strip().eq("Yes")
    contract_table = pd.crosstab(df["Contract"], churn)
    chi2, p_chi, dof, _ = stats.chi2_contingency(contract_table, correction=False)
    n = contract_table.to_numpy().sum()
    cramer_v = np.sqrt(chi2 / (n * min(contract_table.shape[0]-1, contract_table.shape[1]-1)))

    yes = df.loc[churn, "MonthlyCharges"].astype(float)
    no = df.loc[~churn, "MonthlyCharges"].astype(float)
    test = stats.ttest_ind(yes, no, equal_var=False)
    difference = yes.mean() - no.mean()
    se = np.sqrt(yes.var(ddof=1)/len(yes) + no.var(ddof=1)/len(no))
    df_welch = (yes.var(ddof=1)/len(yes) + no.var(ddof=1)/len(no))**2 / (
        (yes.var(ddof=1)/len(yes))**2/(len(yes)-1) +
        (no.var(ddof=1)/len(no))**2/(len(no)-1)
    )
    ci = (difference-stats.t.ppf(.975, df_welch)*se,
          difference+stats.t.ppf(.975, df_welch)*se)
    pooled_sd = np.sqrt(((len(yes)-1)*yes.var(ddof=1)+(len(no)-1)*no.var(ddof=1))/(len(yes)+len(no)-2))
    return {
        "contract_table": contract_table,
        "chi2": chi2, "chi_p": p_chi, "chi_dof": dof, "cramer_v": cramer_v,
        "monthly_churn_mean": yes.mean(), "monthly_stay_mean": no.mean(),
        "monthly_difference": difference, "monthly_ci": ci,
        "welch_t": test.statistic, "welch_p": test.pvalue,
        "cohens_d": difference/pooled_sd,
    }


def train_and_evaluate(df: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    y = df["Churn"].astype(str).str.strip().map({"No": 0, "Yes": 1})
    valid = y.notna()
    y = y.loc[valid].astype(int)
    X = df.loc[valid].drop(columns=["Churn", "customerID"], errors="ignore")
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("encode", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric_pipe, numeric),
                              ("categorical", categorical_pipe, categorical)])
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    model = Pipeline([("preprocessing", prep),
                      ("classifier", LogisticRegression(max_iter=2000,
                                                         class_weight="balanced",
                                                         solver="lbfgs",
                                                         random_state=42))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    results = {
        "rows": len(X), "train_rows": len(X_train), "test_rows": len(X_test),
        "churn_prevalence": y.mean(),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, score),
        "matrix": confusion_matrix(y_test, pred, labels=[0, 1]),
    }
    return results, y_test.to_numpy(), pred, score


def make_figures(df: pd.DataFrame, y_test, pred, score) -> None:
    churn = df["Churn"].astype(str).str.strip().eq("Yes")
    rates = df.assign(_churn=churn).groupby("Contract", observed=True)["_churn"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.8))
    rates.reindex(["Month-to-month", "One year", "Two year"]).plot.bar(ax=ax, color="#1769aa")
    ax.set(title="Observed Churn Rate by Contract Type", ylabel="Share churned", xlabel="Contract type")
    ax.set_ylim(0, .5); ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(FIGURES/"week5_01_contract_churn_rates.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    groups=[df.loc[churn,"MonthlyCharges"],df.loc[~churn,"MonthlyCharges"]]
    ax.boxplot(groups, showfliers=False, patch_artist=True,
               boxprops={"facecolor":"#b9d8ee"}, medianprops={"color":"#173f67","linewidth":2})
    ax.set_xticks([1, 2], ["Churned", "Stayed"])
    ax.set(title="Monthly Charges by Churn Outcome", ylabel="Monthly charges")
    ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(FIGURES/"week5_02_monthly_charges_by_churn.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    matrix=confusion_matrix(y_test,pred,labels=[0,1])
    fig,ax=plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay(matrix,display_labels=["Stayed","Churned"]).plot(ax=ax,cmap="Blues",colorbar=False,values_format="d")
    ax.set_title("Logistic Regression Confusion Matrix")
    fig.tight_layout(); fig.savefig(FIGURES/"week5_03_model_confusion_matrix.png",dpi=200,bbox_inches="tight"); plt.close(fig)

    fpr,tpr,_=roc_curve(y_test,score); auc=roc_auc_score(y_test,score)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot(fpr,tpr,color="#1769aa",lw=2.4,label=f"Logistic regression (AUC = {auc:.3f})")
    ax.plot([0,1],[0,1],"--",color="#777777",label="Chance (AUC = 0.500)")
    ax.set(title="ROC Curve on the Held-Out Test Set",xlabel="False positive rate",ylabel="True positive rate")
    ax.legend(loc="lower right"); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(FIGURES/"week5_04_model_roc_curve.png",dpi=200,bbox_inches="tight"); plt.close(fig)


def main() -> None:
    df = load_and_clean()
    stat = statistical_analysis(df)
    model, y_test, pred, score = train_and_evaluate(df)
    make_figures(df, y_test, pred, score)
    print(f"Rows: {model['rows']} | train: {model['train_rows']} | test: {model['test_rows']}")
    print(f"Churn prevalence: {model['churn_prevalence']:.3%}")
    print(f"Contract chi-square({stat['chi_dof']}): {stat['chi2']:.4f}; p={stat['chi_p']:.4g}; Cramer's V={stat['cramer_v']:.4f}")
    print(f"Monthly charges: churned=${stat['monthly_churn_mean']:.2f}, stayed=${stat['monthly_stay_mean']:.2f}; difference=${stat['monthly_difference']:.2f}; 95% CI=({stat['monthly_ci'][0]:.2f}, {stat['monthly_ci'][1]:.2f}); Welch p={stat['welch_p']:.4g}")
    print("Model metrics:", {k: round(model[k], 4) for k in ["accuracy","precision","recall","f1","roc_auc"]})
    metrics = pd.DataFrame([{"metric":k,"value":model[k]} for k in ["accuracy","precision","recall","f1","roc_auc"]])
    metrics.to_csv(BASE/"week5_model_metrics.csv",index=False)
    print(f"Figures saved to: {FIGURES}")


if __name__ == "__main__":
    main()
