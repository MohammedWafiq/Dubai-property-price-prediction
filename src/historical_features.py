import pandas as pd
import numpy as np


def add_historical_price_features(df):
    """
    Add leakage-safe historical market features.

    For every transaction, historical information comes only
    from transactions occurring on earlier dates.

    The current transaction's target value is never used.
    """

    data = df.copy()

    # --------------------------------------------------------
    # Prepare dates
    # --------------------------------------------------------

    data["INSTANCE_DATE"] = pd.to_datetime(
        data["INSTANCE_DATE"],
        errors="coerce"
    )

    data = data.sort_values(
        "INSTANCE_DATE"
    ).reset_index(drop=True)

    data["_date"] = data[
        "INSTANCE_DATE"
    ].dt.date

    # --------------------------------------------------------
    # Calculate price per square metre
    #
    # This is temporary and is ONLY used to calculate
    # historical information.
    # --------------------------------------------------------

    data["_ppsqm"] = (
        data["TRANS_VALUE"]
        / data["ACTUAL_AREA"]
    )

    data["_ppsqm"] = data[
        "_ppsqm"
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ========================================================
    # Helper function
    # ========================================================

    def create_group_history(group_column):

        daily = (
            data
            .groupby(
                [group_column, "_date"],
                dropna=False
            )
            .agg(
                daily_median_ppsqm=(
                    "_ppsqm",
                    "median"
                ),
                daily_transaction_count=(
                    "_ppsqm",
                    "count"
                )
            )
            .reset_index()
        )

        daily = daily.sort_values(
            [group_column, "_date"]
        )

        # Previous-date historical median
        daily[
            "historical_ppsqm"
        ] = (
            daily
            .groupby(
                group_column,
                dropna=False
            )[
                "daily_median_ppsqm"
            ]
            .transform(
                lambda x:
                    x.shift(1)
                    .expanding()
                    .median()
            )
        )

        # Previous-date cumulative transaction count
        daily[
            "historical_count"
        ] = (
            daily
            .groupby(
                group_column,
                dropna=False
            )[
                "daily_transaction_count"
            ]
            .transform(
                lambda x:
                    x.shift(1)
                    .expanding()
                    .sum()
            )
        )

        result = data[
            [group_column, "_date"]
        ].merge(
            daily[
                [
                    group_column,
                    "_date",
                    "historical_ppsqm",
                    "historical_count"
                ]
            ],
            on=[
                group_column,
                "_date"
            ],
            how="left"
        )

        return (
            result["historical_ppsqm"].values,
            result["historical_count"].values
        )

    # ========================================================
    # Project history
    # ========================================================

    print(
        "Creating historical project features..."
    )

    (
        data["historical_project_ppsqm"],
        data["historical_project_count"]
    ) = create_group_history(
        "PROJECT_EN"
    )

    # ========================================================
    # Area history
    # ========================================================

    print(
        "Creating historical area features..."
    )

    (
        data["historical_area_ppsqm"],
        data["historical_area_count"]
    ) = create_group_history(
        "AREA_EN"
    )

    # ========================================================
    # Property-type history
    # ========================================================

    print(
        "Creating historical property-type features..."
    )

    (
        data[
            "historical_property_type_ppsqm"
        ],
        data[
            "historical_property_type_count"
        ]
    ) = create_group_history(
        "PROP_SB_TYPE_EN"
    )

    # ========================================================
    # Overall market history
    # ========================================================

    print(
        "Creating historical overall market features..."
    )

    daily_market = (
        data
        .groupby("_date")["_ppsqm"]
        .agg(
            daily_median="median",
            daily_count="count"
        )
        .sort_index()
    )

    data[
        "historical_market_ppsqm"
    ] = (
        daily_market[
            "daily_median"
        ]
        .shift(1)
        .expanding()
        .median()
    ).reindex(
        data["_date"]
    ).values

    data[
        "historical_market_count"
    ] = (
        daily_market[
            "daily_count"
        ]
        .shift(1)
        .expanding()
        .sum()
    ).reindex(
        data["_date"]
    ).values

    # ========================================================
    # Hierarchical historical price
    # ========================================================

    data[
        "historical_best_ppsqm"
    ] = (
        data["historical_project_ppsqm"]
        .fillna(
            data["historical_area_ppsqm"]
        )
        .fillna(
            data[
                "historical_property_type_ppsqm"
            ]
        )
        .fillna(
            data["historical_market_ppsqm"]
        )
    )

    # ========================================================
    # Identify historical price source
    # ========================================================

    data[
        "historical_price_source"
    ] = np.select(
        [
            data[
                "historical_project_ppsqm"
            ].notna(),

            data[
                "historical_area_ppsqm"
            ].notna(),

            data[
                "historical_property_type_ppsqm"
            ].notna(),

            data[
                "historical_market_ppsqm"
            ].notna()
        ],
        [
            "project",
            "area",
            "property_type",
            "overall_market"
        ],
        default="none"
    )

    # ========================================================
    # Remove temporary columns
    # ========================================================

    data = data.drop(
        columns=[
            "_date",
            "_ppsqm"
        ]
    )

    return data


# ============================================================
# TEST SCRIPT
# ============================================================

if __name__ == "__main__":

    from data_loader import load_data

    df = load_data()

    result = add_historical_price_features(
        df
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "HISTORICAL FEATURE SUMMARY"
    )

    print(
        "=" * 60
    )

    feature_columns = [
        "historical_project_ppsqm",
        "historical_area_ppsqm",
        "historical_property_type_ppsqm",
        "historical_market_ppsqm",
        "historical_project_count",
        "historical_area_count",
        "historical_property_type_count",
        "historical_market_count",
        "historical_best_ppsqm"
    ]

    print(
        result[
            feature_columns
        ].describe()
    )

    print(
        "\nMissing values:"
    )

    print(
        result[
            feature_columns
        ].isnull().sum()
    )

    print(
        "\nHistorical price source:"
    )

    print(
        result[
            "historical_price_source"
        ].value_counts()
    )