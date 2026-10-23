from pathlib import Path
import sys
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# Dataset and hypotheses

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)

ALPHA = 0.05
NUMBER_OF_TESTS = 2
ADJUSTED_ALPHA = ALPHA / NUMBER_OF_TESTS  # Bonferroni threshold: 0.025

# Primary hypothesis: contract type and churn
H0_PRIMARY = (
    "Contract type and churn status are independent."
)
H1_PRIMARY = (
    "Contract type and churn status are associated."
)

# Secondary hypothesis: monthly charges and churn
H0_SECONDARY = (
    "Mean monthly charges are equal for churned and stayed customers."
)
H1_SECONDARY = (
    "Mean monthly charges differ between churned and stayed customers."
)


def load_data():
    """Read a supplied CSV path or download the public IBM CSV."""
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
        return pd.read_csv(csv_path)

    csv_path = Path("Telco-Customer-Churn.csv")

    if not csv_path.exists():
        print("Downloading the IBM Telco Customer Churn dataset...")
        urllib.request.urlretrieve(DATA_URL, csv_path)

    return pd.read_csv(csv_path)


def wilson_interval(successes, total, z=1.959963984540054):
    """Return a two-sided 95% Wilson confidence interval."""
    proportion = successes / total
    denominator = 1 + (z ** 2 / total)

    center = (
        proportion + z ** 2 / (2 * total)
    ) / denominator

    margin = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / total
            + z ** 2 / (4 * total ** 2)
        )
        / denominator
    )

    return center - margin, center + margin


def main():
    sns.set_theme(style="whitegrid", palette="deep")

    print("PRIMARY HYPOTHESIS")
    print("H0:", H0_PRIMARY)
    print("H1:", H1_PRIMARY)

    print("\nSECONDARY HYPOTHESIS")
    print("H0:", H0_SECONDARY)
    print("H1:", H1_SECONDARY)

    print(f"\nSignificance level: {ALPHA}")
    print(
        "Bonferroni-adjusted significance level: "
        f"{ADJUSTED_ALPHA}"
    )

    # Load and clean the dataset

    df = load_data()

    # Convert TotalCharges to numeric.
    # Blank values become NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # In this dataset, blank TotalCharges rows have tenure = 0.
    missing_total_charges = df["TotalCharges"].isna()

    if (
        missing_total_charges.any()
        and (df.loc[missing_total_charges, "tenure"] == 0).all()
    ):
        df.loc[missing_total_charges, "TotalCharges"] = 0

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    df["Churn"] = df["Churn"].astype(str).str.strip()

    # Primary test: chi-square test of independence
    # Contract type vs churn status

    contingency_table = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )[["No", "Yes"]]

    chi_square, chi_p_value, degrees_freedom, expected_counts = (
        stats.chi2_contingency(
            contingency_table,
            correction=False
        )
    )

    sample_size = int(contingency_table.to_numpy().sum())

    cramers_v = np.sqrt(
        chi_square
        / (
            sample_size
            * min(
                contingency_table.shape[0] - 1,
                contingency_table.shape[1] - 1
            )
        )
    )

    contract_order = [
        "Month-to-month",
        "One year",
        "Two year"
    ]

    churn_counts = (
        df.groupby("Contract", observed=True)["Churn"]
        .apply(lambda values: int((values == "Yes").sum()))
    )

    contract_totals = df.groupby(
        "Contract",
        observed=True
    ).size()

    churn_rates = churn_counts / contract_totals

    contract_confidence_intervals = {}

    for contract in contract_order:
        contract_confidence_intervals[contract] = wilson_interval(
            int(churn_counts[contract]),
            int(contract_totals[contract])
        )

    # Planned comparison:
    # Month-to-month vs pooled fixed-term customers

    month_to_month = df["Contract"] == "Month-to-month"
    churned = df["Churn"] == "Yes"

    churned_month_to_month = int(
        (month_to_month & churned).sum()
    )
    stayed_month_to_month = int(
        (month_to_month & ~churned).sum()
    )
    churned_fixed_term = int(
        (~month_to_month & churned).sum()
    )
    stayed_fixed_term = int(
        (~month_to_month & ~churned).sum()
    )

    risk_month_to_month = (
        churned_month_to_month
        / (churned_month_to_month + stayed_month_to_month)
    )

    risk_fixed_term = (
        churned_fixed_term
        / (churned_fixed_term + stayed_fixed_term)
    )

    risk_difference = risk_month_to_month - risk_fixed_term

    risk_difference_se = np.sqrt(
        risk_month_to_month
        * (1 - risk_month_to_month)
        / (churned_month_to_month + stayed_month_to_month)
        +
        risk_fixed_term
        * (1 - risk_fixed_term)
        / (churned_fixed_term + stayed_fixed_term)
    )

    risk_difference_ci = (
        risk_difference - 1.96 * risk_difference_se,
        risk_difference + 1.96 * risk_difference_se
    )

    risk_ratio = risk_month_to_month / risk_fixed_term

    log_risk_ratio_se = np.sqrt(
        1 / churned_month_to_month
        - 1 / (churned_month_to_month + stayed_month_to_month)
        +
        1 / churned_fixed_term
        - 1 / (churned_fixed_term + stayed_fixed_term)
    )

    risk_ratio_ci = (
        np.exp(np.log(risk_ratio) - 1.96 * log_risk_ratio_se),
        np.exp(np.log(risk_ratio) + 1.96 * log_risk_ratio_se)
    )

    # Secondary test: Welch's independent-samples t-test
    # Monthly charges by churn status


    monthly_charges_churned = df.loc[
        churned,
        "MonthlyCharges"
    ].astype(float)

    monthly_charges_stayed = df.loc[
        ~churned,
        "MonthlyCharges"
    ].astype(float)

    t_statistic, t_p_value = stats.ttest_ind(
        monthly_charges_churned,
        monthly_charges_stayed,
        equal_var=False
    )

    mean_difference = (
        monthly_charges_churned.mean()
        - monthly_charges_stayed.mean()
    )

    standard_error = np.sqrt(
        monthly_charges_churned.var(ddof=1)
        / len(monthly_charges_churned)
        +
        monthly_charges_stayed.var(ddof=1)
        / len(monthly_charges_stayed)
    )

    variance_term_churned = (
        monthly_charges_churned.var(ddof=1)
        / len(monthly_charges_churned)
    )

    variance_term_stayed = (
        monthly_charges_stayed.var(ddof=1)
        / len(monthly_charges_stayed)
    )

    welch_degrees_freedom = (
        (variance_term_churned + variance_term_stayed) ** 2
        /
        (
            variance_term_churned ** 2
            / (len(monthly_charges_churned) - 1)
            +
            variance_term_stayed ** 2
            / (len(monthly_charges_stayed) - 1)
        )
    )

    critical_t = stats.t.ppf(
        0.975,
        welch_degrees_freedom
    )

    mean_difference_ci = (
        mean_difference - critical_t * standard_error,
        mean_difference + critical_t * standard_error
    )

    pooled_standard_deviation = np.sqrt(
        (
            (len(monthly_charges_churned) - 1)
            * monthly_charges_churned.var(ddof=1)
            +
            (len(monthly_charges_stayed) - 1)
            * monthly_charges_stayed.var(ddof=1)
        )
        /
        (
            len(monthly_charges_churned)
            + len(monthly_charges_stayed)
            - 2
        )
    )

    cohens_d = mean_difference / pooled_standard_deviation

    # Create an output folder for figures

    figure_folder = Path("week3_figures")
    figure_folder.mkdir(exist_ok=True)

    # Figure 1: Churn rate by contract type
    # Error bars show Wilson 95% confidence intervals.
    rates = [
        churn_rates[contract]
        for contract in contract_order
    ]

    lower_errors = [
        churn_rates[contract]
        - contract_confidence_intervals[contract][0]
        for contract in contract_order
    ]

    upper_errors = [
        contract_confidence_intervals[contract][1]
        - churn_rates[contract]
        for contract in contract_order
    ]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    ax.bar(
        contract_order,
        rates,
        color=["#d97757", "#4c78a8", "#59a14f"],
        width=0.62
    )

    ax.errorbar(
        range(len(contract_order)),
        rates,
        yerr=[lower_errors, upper_errors],
        fmt="none",
        ecolor="#273746",
        capsize=5,
        linewidth=1.6
    )

    ax.set_title("Churn rate by contract type")
    ax.set_xlabel("Contract type")
    ax.set_ylabel("Customers who churned")
    ax.set_ylim(0, max(rates) + 0.16)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )

    for index, rate in enumerate(rates):
        ax.text(
            index,
            rate + 0.025,
            f"{rate:.1%}",
            ha="center",
            fontsize=10
        )

    fig.tight_layout()
    fig.savefig(
        figure_folder / "01_contract_churn_rates.png",
        dpi=190
    )
    plt.close(fig)

    # Figure 2: Monthly charges by churn status

    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges",
        order=["No", "Yes"],
        hue="Churn",
        hue_order=["No", "Yes"],
        legend=False,
        showfliers=False,
        palette={
            "No": "#4c78a8",
            "Yes": "#d97757"
        },
        ax=ax
    )

    plot_sample = df.sample(
        min(len(df), 900),
        random_state=17
    )

    sns.stripplot(
        data=plot_sample,
        x="Churn",
        y="MonthlyCharges",
        order=["No", "Yes"],
        color="#222222",
        alpha=0.16,
        size=2.2,
        ax=ax
    )

    ax.set_title("Monthly charges by churn outcome")
    ax.set_xlabel("Churned during the recorded period")
    ax.set_ylabel("Monthly charges")

    fig.tight_layout()
    fig.savefig(
        figure_folder / "02_monthly_charges_by_churn.png",
        dpi=190
    )
    plt.close(fig)

    # Figure 3: Monthly charge distributions and group means

    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    sns.histplot(
        data=df,
        x="MonthlyCharges",
        hue="Churn",
        hue_order=["No", "Yes"],
        stat="density",
        common_norm=False,
        bins=30,
        element="step",
        fill=False,
        ax=ax
    )

    ax.axvline(
        monthly_charges_stayed.mean(),
        color="#4c78a8",
        linestyle="--",
        linewidth=1.5,
        label=f"Stayed mean: ${monthly_charges_stayed.mean():.2f}"
    )

    ax.axvline(
        monthly_charges_churned.mean(),
        color="#d97757",
        linestyle="--",
        linewidth=1.5,
        label=f"Churned mean: ${monthly_charges_churned.mean():.2f}"
    )

    ax.set_title("Monthly charge distributions by churn outcome")
    ax.set_xlabel("Monthly charges")
    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        figure_folder / "03_monthly_charge_distributions.png",
        dpi=190
    )
    plt.close(fig)

    # Print results and hypothesis decisions

    total_churned = int(churned.sum())

    print("\nDATA SUMMARY")
    print(f"Customer records analyzed: {len(df)}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print(
        f"Churned customers: {total_churned} "
        f"({churned.mean():.2%})"
    )

    print("\nCHURN RATE BY CONTRACT TYPE")
    for contract in contract_order:
        ci_low, ci_high = contract_confidence_intervals[contract]

        print(
            f"{contract}: "
            f"{churn_counts[contract]}/{contract_totals[contract]} "
            f"= {churn_rates[contract]:.2%}; "
            f"Wilson 95% CI [{ci_low:.2%}, {ci_high:.2%}]"
        )

    print("\nPRIMARY TEST: PEARSON CHI-SQUARE TEST")
    print(f"Chi-square statistic: {chi_square:.4f}")
    print(f"Degrees of freedom: {degrees_freedom}")
    print(f"p-value: {chi_p_value:.6g}")
    print(f"Cramer's V: {cramers_v:.4f}")

    if chi_p_value < ADJUSTED_ALPHA:
        print(
            "Decision: Reject H0_PRIMARY. "
            "The results support an association between "
            "contract type and churn status."
        )
    else:
        print(
            "Decision: Fail to reject H0_PRIMARY. "
            "There is not enough evidence of an association "
            "between contract type and churn status."
        )

    print("\nMONTH-TO-MONTH VS FIXED-TERM CONTRACTS")
    print(
        f"Risk difference: {risk_difference:.2%}; "
        f"95% CI [{risk_difference_ci[0]:.2%}, "
        f"{risk_difference_ci[1]:.2%}]"
    )
    print(
        f"Risk ratio: {risk_ratio:.4f}; "
        f"95% CI [{risk_ratio_ci[0]:.4f}, "
        f"{risk_ratio_ci[1]:.4f}]"
    )

    print("\nMONTHLY CHARGES BY CHURN STATUS")
    print(
        f"Churned: n={len(monthly_charges_churned)}, "
        f"mean=${monthly_charges_churned.mean():.2f}, "
        f"SD=${monthly_charges_churned.std(ddof=1):.2f}"
    )
    print(
        f"Stayed: n={len(monthly_charges_stayed)}, "
        f"mean=${monthly_charges_stayed.mean():.2f}, "
        f"SD=${monthly_charges_stayed.std(ddof=1):.2f}"
    )

    print("\nSECONDARY TEST: WELCH TWO-SAMPLE T-TEST")
    print(f"t-statistic: {t_statistic:.4f}")
    print(f"Degrees of freedom: {welch_degrees_freedom:.1f}")
    print(f"p-value: {t_p_value:.6g}")
    print(
        f"Mean difference (churned - stayed): "
        f"${mean_difference:.2f}"
    )
    print(
        f"95% CI for mean difference: "
        f"[${mean_difference_ci[0]:.2f}, "
        f"${mean_difference_ci[1]:.2f}]"
    )
    print(f"Cohen's d: {cohens_d:.4f}")

    if t_p_value < ADJUSTED_ALPHA:
        print(
            "Decision: Reject H0_SECONDARY. "
            "The results support a difference in mean "
            "monthly charges between churn groups."
        )
    else:
        print(
            "Decision: Fail to reject H0_SECONDARY. "
            "There is not enough evidence of a difference "
            "in mean monthly charges between churn groups."
        )

    print(
        "\nFigures saved in: "
        f"{figure_folder.resolve()}"
    )

    print(
        "\nInterpretation note: These are observational "
        "associations. They do not prove that contract type "
        "or monthly charges caused churn."
    )


if __name__ == "__main__":
    main()