import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data


def inspect_numeric_columns(df):
    """Inspect numerical columns for unusual values."""

    print("\n" + "=" * 60)
    print("NUMERICAL COLUMN SUMMARY")
    print("=" * 60)

    numeric_columns = [
        "TRANS_VALUE",
        "ACTUAL_AREA",
        "price_per_sqm"
    ]

    print(df[numeric_columns].describe().T)


def inspect_target_extremes(df):
    """Inspect the highest and lowest transaction values."""

    print("\n" + "=" * 60)
    print("TARGET EXTREMES")
    print("=" * 60)

    print("\nLowest 10 transactions:")
    print(
        df[
            [
                "TRANS_VALUE",
                "ACTUAL_AREA",
                "AREA_EN",
                "PROP_SB_TYPE_EN",
                "ROOMS_EN"
            ]
        ]
        .sort_values("TRANS_VALUE")
        .head(10)
        .to_string(index=False)
    )

    print("\nHighest 10 transactions:")
    print(
        df[
            [
                "TRANS_VALUE",
                "ACTUAL_AREA",
                "AREA_EN",
                "PROP_SB_TYPE_EN",
                "ROOMS_EN"
            ]
        ]
        .sort_values("TRANS_VALUE", ascending=False)
        .head(10)
        .to_string(index=False)
    )


def inspect_area(df):
    """Inspect property area values."""

    print("\n" + "=" * 60)
    print("PROPERTY AREA")
    print("=" * 60)

    print(df["ACTUAL_AREA"].describe())

    print(f"\nZero/negative area values:")
    print((df["ACTUAL_AREA"] <= 0).sum())

    print("\nSmallest 10 properties:")
    print(
        df[
            [
                "ACTUAL_AREA",
                "TRANS_VALUE",
                "AREA_EN",
                "PROP_SB_TYPE_EN"
            ]
        ]
        .sort_values("ACTUAL_AREA")
        .head(10)
        .to_string(index=False)
    )

    print("\nLargest 10 properties:")
    print(
        df[
            [
                "ACTUAL_AREA",
                "TRANS_VALUE",
                "AREA_EN",
                "PROP_SB_TYPE_EN"
            ]
        ]
        .sort_values("ACTUAL_AREA", ascending=False)
        .head(10)
        .to_string(index=False)
    )


def inspect_categories(df):
    """Inspect important categorical columns."""

    print("\n" + "=" * 60)
    print("CATEGORICAL VARIABLES")
    print("=" * 60)

    categorical_columns = [
        "PROCEDURE_EN",
        "IS_FREE_HOLD_EN",
        "AREA_EN",
        "PROP_SB_TYPE_EN",
        "ROOMS_EN"
    ]

    for column in categorical_columns:

        print("\n" + "-" * 60)
        print(column)

        print(
            df[column]
            .value_counts(dropna=False)
            .head(15)
            .to_string()
        )


def inspect_dates(df):
    """Inspect transaction dates."""

    print("\n" + "=" * 60)
    print("TRANSACTION DATES")
    print("=" * 60)

    dates = pd.to_datetime(
        df["INSTANCE_DATE"],
        errors="coerce"
    )

    print(f"\nInvalid dates: {dates.isna().sum()}")

    print(f"Earliest transaction: {dates.min()}")
    print(f"Latest transaction:   {dates.max()}")

    print("\nTransactions by year:")

    yearly_counts = (
        dates
        .dt.year
        .value_counts()
        .sort_index()
    )

    print(yearly_counts.to_string())


def inspect_target_relationships(df):
    """Investigate the relationship between area and transaction value."""

    print("\n" + "=" * 60)
    print("TARGET RELATIONSHIPS")
    print("=" * 60)

    correlation = df[
        [
            "TRANS_VALUE",
            "ACTUAL_AREA",
            "price_per_sqm"
        ]
    ].corr()

    print("\nCorrelation matrix:")
    print(correlation)


def plot_area_vs_price(df):
    """Plot property area against transaction value."""

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="ACTUAL_AREA",
        y="TRANS_VALUE",
        alpha=0.4
    )

    plt.title("Property Area vs Transaction Value")
    plt.xlabel("Property Area")
    plt.ylabel("Transaction Value (AED)")

    plt.tight_layout()

    plt.savefig(
        "reports/area_vs_transaction_value.png",
        dpi=300
    )

    plt.show()


def plot_price_per_sqm(df):
    """Plot price per square metre."""

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["price_per_sqm"].dropna(),
        bins=100,
        kde=True
    )

    plt.title("Distribution of Price per Square Metre")
    plt.xlabel("Price per Square Metre")
    plt.ylabel("Number of Transactions")

    plt.tight_layout()

    plt.savefig(
        "reports/price_per_sqm_distribution.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":

    df = load_data()

    inspect_numeric_columns(df)

    inspect_target_extremes(df)

    inspect_area(df)

    inspect_categories(df)

    inspect_dates(df)

    inspect_target_relationships(df)

    plot_area_vs_price(df)

    plot_price_per_sqm(df)