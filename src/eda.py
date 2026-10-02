import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data


def basic_overview(df):
    """Display basic information about the dataset."""

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isnull().sum()
    missing_percentage = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_percentage": missing_percentage
    })

    print(missing_df[missing_df["missing_count"] > 0])


def check_duplicates(df):
    """Check for duplicate rows."""

    duplicate_count = df.duplicated().sum()

    print("\n" + "=" * 60)
    print("DUPLICATES")
    print("=" * 60)

    print(f"Duplicate rows: {duplicate_count:,}")


def analyze_target(df):
    """Analyze the transaction value target variable."""

    print("\n" + "=" * 60)
    print("TARGET VARIABLE: TRANS_VALUE")
    print("=" * 60)

    print(df["TRANS_VALUE"].describe())

    print(f"\nMean:   AED {df['TRANS_VALUE'].mean():,.2f}")
    print(f"Median: AED {df['TRANS_VALUE'].median():,.2f}")
    print(f"Min:    AED {df['TRANS_VALUE'].min():,.2f}")
    print(f"Max:    AED {df['TRANS_VALUE'].max():,.2f}")


def plot_target_distribution(df):
    """Plot the distribution of transaction values."""

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["TRANS_VALUE"],
        bins=100,
        kde=True
    )

    plt.title("Dubai Property Transaction Values")
    plt.xlabel("Transaction Value (AED)")
    plt.ylabel("Number of Transactions")

    plt.tight_layout()

    plt.savefig(
        "reports/transaction_value_distribution.png",
        dpi=300
    )

    plt.show()


def plot_log_target_distribution(df):
    """Plot log-transformed transaction values."""

    plt.figure(figsize=(10, 6))

    sns.histplot(
        np.log1p(df["TRANS_VALUE"]),
        bins=100,
        kde=True
    )

    plt.title("Log Distribution of Dubai Property Transaction Values")
    plt.xlabel("log(Transaction Value)")
    plt.ylabel("Number of Transactions")

    plt.tight_layout()

    plt.savefig(
        "reports/log_transaction_value_distribution.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":

    df = load_data()

    basic_overview(df)
    check_duplicates(df)
    analyze_target(df)

    plot_target_distribution(df)
    plot_log_target_distribution(df)