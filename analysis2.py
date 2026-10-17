import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset from Week 1
df = pd.read_csv("cleaned_telco_churn.csv")

# Create folder to save graphs
os.makedirs("week2_graphs", exist_ok=True)

# Set visualization style
sns.set_theme(style="whitegrid")

# Create numeric churn column
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

print("=" * 50)
print("WEEK 2: DATA VISUALIZATION AND STORYTELLING")
print("=" * 50)

print("\nDataset shape:", df.shape)
print("\nChurn distribution:")
print(df["Churn"].value_counts())

# GRAPH 1: CHURN RATE BY CONTRACT TYPE

contract_churn = (
    df.groupby("Contract")["ChurnFlag"].mean() * 100
).reset_index()

contract_churn.columns = ["Contract", "ChurnRate"]

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    data=contract_churn,
    x="Contract",
    y="ChurnRate"
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=3)

plt.title("Customer Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()

plt.savefig(
    "week2_graphs/01_contract_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# GRAPH 2: CUSTOMER TENURE BY CHURN

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Customer Tenure by Churn Status")
plt.xlabel("Churn Status")
plt.ylabel("Tenure (Months)")
plt.tight_layout()

plt.savefig(
    "week2_graphs/02_tenure_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# GRAPH 3: MONTHLY CHARGES BY CHURN

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn Status")
plt.ylabel("Monthly Charges")
plt.tight_layout()

plt.savefig(
    "week2_graphs/03_monthly_charges_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# GRAPH 4: CHURN RATE BY PAYMENT METHOD
payment_churn = (
    df.groupby("PaymentMethod")["ChurnFlag"].mean() * 100
).reset_index()

payment_churn.columns = ["PaymentMethod", "ChurnRate"]

payment_churn = payment_churn.sort_values(
    "ChurnRate",
    ascending=False
)

plt.figure(figsize=(11, 6))

ax = sns.barplot(
    data=payment_churn,
    x="PaymentMethod",
    y="ChurnRate"
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=3)

plt.title("Customer Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(
    "week2_graphs/04_payment_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# GRAPH 5: CHURN RATE BY INTERNET SERVICE

internet_churn = (
    df.groupby("InternetService")["ChurnFlag"].mean() * 100
).reset_index()

internet_churn.columns = ["InternetService", "ChurnRate"]

internet_churn = internet_churn.sort_values(
    "ChurnRate",
    ascending=False
)

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    data=internet_churn,
    x="InternetService",
    y="ChurnRate"
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=3)

plt.title("Customer Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()

plt.savefig(
    "week2_graphs/05_internet_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# PRINT RESULTS FOR THE REPORT

print("\n" + "=" * 50)
print("CHURN RATE BY CONTRACT TYPE")
print("=" * 50)
print(contract_churn.round(2).to_string(index=False))

print("\n" + "=" * 50)
print("CHURN RATE BY PAYMENT METHOD")
print("=" * 50)
print(payment_churn.round(2).to_string(index=False))

print("\n" + "=" * 50)
print("CHURN RATE BY INTERNET SERVICE")
print("=" * 50)
print(internet_churn.round(2).to_string(index=False))

print("\n" + "=" * 50)
print("WEEK 2 ANALYSIS COMPLETED SUCCESSFULLY!")
print("Five graphs saved in the week2_graphs folder.")
print("=" * 50)