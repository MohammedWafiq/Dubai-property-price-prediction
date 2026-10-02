import pandas as pd
import numpy as np
from pathlib import Path

from data_loader import load_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(exist_ok=True)


def analyze_high_value_transactions(df):

    data = df.copy()

    data["INSTANCE_DATE"] = pd.to_datetime(
        data["INSTANCE_DATE"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Calculate price per square metre for investigation only
    # --------------------------------------------------------

    data["calculated_ppsqm"] = (
        data["TRANS_VALUE"]
        / data["ACTUAL_AREA"]
    )

    data["calculated_ppsqm"] = data[
        "calculated_ppsqm"
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Define high-value transactions
    # --------------------------------------------------------

    high_value = data[
        data["TRANS_VALUE"] >= 10_000_000
    ].copy()

    high_value = high_value.sort_values(
        "TRANS_VALUE",
        ascending=False
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "HIGH-VALUE TRANSACTION ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTransactions >= AED 10M: "
        f"{len(high_value):,}"
    )

    print(
        f"Percentage of dataset: "
        f"{len(high_value) / len(data) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\nHigh-value transaction summary:"
    )

    print(
        high_value[
            "TRANS_VALUE"
        ].describe()
    )

    # --------------------------------------------------------
    # Top 30 transactions
    # --------------------------------------------------------

    columns = [
        "INSTANCE_DATE",
        "TRANS_VALUE",
        "ACTUAL_AREA",
        "calculated_ppsqm",
        "AREA_EN",
        "PROP_SB_TYPE_EN",
        "ROOMS_EN",
        "PROJECT_EN",
        "PROCEDURE_EN",
        "IS_FREE_HOLD_EN"
    ]

    top_transactions = high_value[
        columns
    ].head(30)

    print(
        "\n" + "=" * 80
    )

    print(
        "TOP 30 HIGHEST-VALUE TRANSACTIONS"
    )

    print(
        "=" * 80
    )

    print(
        top_transactions.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    top_path = (
        REPORTS_DIR
        / "high_value_transactions.csv"
    )

    high_value[
        columns
    ].to_csv(
        top_path,
        index=False
    )

    print(
        f"\nSaved:"
    )

    print(
        top_path
    )

    # --------------------------------------------------------
    # High-value transactions by area
    # --------------------------------------------------------

    area_summary = (
        high_value
        .groupby("AREA_EN")
        .agg(
            transaction_count=(
                "TRANS_VALUE",
                "count"
            ),
            total_value=(
                "TRANS_VALUE",
                "sum"
            ),
            median_value=(
                "TRANS_VALUE",
                "median"
            ),
            median_ppsqm=(
                "calculated_ppsqm",
                "median"
            )
        )
        .sort_values(
            "total_value",
            ascending=False
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "HIGH-VALUE TRANSACTIONS BY AREA"
    )

    print(
        "=" * 80
    )

    print(
        area_summary.head(20).to_string()
    )

    area_path = (
        REPORTS_DIR
        / "high_value_by_area.csv"
    )

    area_summary.to_csv(
        area_path
    )

    # --------------------------------------------------------
    # High-value transactions by property type
    # --------------------------------------------------------

    property_summary = (
        high_value
        .groupby("PROP_SB_TYPE_EN")
        .agg(
            transaction_count=(
                "TRANS_VALUE",
                "count"
            ),
            total_value=(
                "TRANS_VALUE",
                "sum"
            ),
            median_value=(
                "TRANS_VALUE",
                "median"
            ),
            median_ppsqm=(
                "calculated_ppsqm",
                "median"
            )
        )
        .sort_values(
            "total_value",
            ascending=False
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "HIGH-VALUE TRANSACTIONS BY PROPERTY TYPE"
    )

    print(
        "=" * 80
    )

    print(
        property_summary.to_string()
    )

    property_path = (
        REPORTS_DIR
        / "high_value_by_property_type.csv"
    )

    property_summary.to_csv(
        property_path
    )

    # --------------------------------------------------------
    # Extreme price per square metre
    # --------------------------------------------------------

    extreme_ppsqm = data[
        data["calculated_ppsqm"] >= 100_000
    ].copy()

    extreme_ppsqm = extreme_ppsqm.sort_values(
        "calculated_ppsqm",
        ascending=False
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "EXTREME PRICE PER SQUARE METRE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTransactions >= AED 100,000/m²: "
        f"{len(extreme_ppsqm):,}"
    )

    print(
        extreme_ppsqm[
            columns
        ].head(30).to_string(
            index=False
        )
    )

    extreme_path = (
        REPORTS_DIR
        / "extreme_ppsqm_transactions.csv"
    )

    extreme_ppsqm[
        columns
    ].to_csv(
        extreme_path,
        index=False
    )

    print(
        f"\nSaved:"
    )

    print(
        extreme_path
    )


if __name__ == "__main__":

    df = load_data()

    analyze_high_value_transactions(
        df
    )