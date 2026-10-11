
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("Original dataset shape:", df.shape)

# Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Inspect rows with missing TotalCharges
missing_rows = df["TotalCharges"].isnull()

print("\nRows with missing TotalCharges:")
print(df.loc[missing_rows, ["tenure", "MonthlyCharges"]])

# Handle missing TotalCharges
if missing_rows.any():
    if (df.loc[missing_rows, "tenure"] == 0).all():
        # Zero tenure indicates no accumulated charges yet
        df.loc[missing_rows, "TotalCharges"] = 0
    else:
        # Use median if missing rows are not all zero-tenure
        median_value = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(
            median_value
        )

# Remove duplicate rows if any
df = df.drop_duplicates()

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nCleaned dataset shape:", df.shape)

# Save cleaned dataset
df.to_csv("cleaned_telco_churn.csv", index=False)

# Create folder for graphs
os.makedirs("graphs", exist_ok=True)

# Graph 1: Monthly Charges Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="MonthlyCharges", bins=30, kde=True)
plt.title("Distribution of Monthly Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("graphs/01_monthly_charges.png", dpi=150)
plt.show()
plt.close()

# Graph 2: Customer Churn Count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("graphs/02_churn_distribution.png", dpi=150)
plt.show()
plt.close()

# Graph 3: Correlation Heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(8, 6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("graphs/03_correlation_heatmap.png", dpi=150)
plt.show()
plt.close()

print("\nCleaning completed!")
print("Cleaned CSV saved.")
print("Three graphs saved in the graphs folder.")