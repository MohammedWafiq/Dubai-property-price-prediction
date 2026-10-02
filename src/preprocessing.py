import pandas as pd
import numpy as np

from data_loader import load_data


TARGET_COLUMN = "TRANS_VALUE"


LEAKAGE_COLUMNS = [
    "price_per_sqm",
    "value_band",
    "size_category"
]

def prepare_data(df):
    """
    Prepare the raw Dubai property dataset for modelling.

    The raw dataframe is not modified.
    """

    data = df.copy()

    # ---------------------------------------------------------
    # 1. Remove columns that contain information derived
    #    directly from the target variable.
    # ---------------------------------------------------------

    data = data.drop(
        columns=LEAKAGE_COLUMNS,
        errors="ignore"
    )

    # ---------------------------------------------------------
    # 2. Convert transaction date
    # ---------------------------------------------------------

    data["INSTANCE_DATE"] = pd.to_datetime(
        data["INSTANCE_DATE"],
        errors="coerce"
    )

    data["transaction_month"] = (
        data["INSTANCE_DATE"].dt.month
    )

    data["transaction_day"] = (
        data["INSTANCE_DATE"].dt.day
    )

    data["transaction_day_of_week"] = (
        data["INSTANCE_DATE"].dt.dayofweek
    )

    data["transaction_day_of_year"] = (
        data["INSTANCE_DATE"].dt.dayofyear
    )

    # The original date column is no longer needed
    # after extracting its useful components.
    data = data.drop(
        columns=["INSTANCE_DATE"]
    )

    # ---------------------------------------------------------
    # 3. Handle missing categorical values
    # ---------------------------------------------------------

    categorical_columns = [
        "PROCEDURE_EN",
        "IS_FREE_HOLD_EN",
        "AREA_EN",
        "PROP_SB_TYPE_EN",
        "ROOMS_EN",
        "NEAREST_METRO_EN",
        "NEAREST_MALL_EN",
        "NEAREST_LANDMARK_EN",
        "PROJECT_EN"
    ]

    for column in categorical_columns:

        if column in data.columns:

            data[column] = (
                data[column]
                .fillna("Unknown")
                .astype(str)
            )

    # ---------------------------------------------------------
    # 4. Ensure numerical columns are numeric
    # ---------------------------------------------------------

    numerical_columns = [
        "ACTUAL_AREA"
    ]

    for column in numerical_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data


def show_prepared_data(data):
    """Display information about the prepared dataset."""

    print("\n" + "=" * 60)
    print("PREPARED DATASET")
    print("=" * 60)

    print(f"\nShape: {data.shape}")

    print("\nColumns:")
    for column in data.columns:
        print(f"  - {column}")

    print("\nMissing values:")

    missing = data.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values.")
    else:
        print(missing)


if __name__ == "__main__":

    df = load_data()

    prepared_df = prepare_data(df)

    show_prepared_data(prepared_df)