from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "property_price_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        print("\nModel file not found.")

        print(
            "\nPlease run:"
        )

        print(
            "python src\\train_final_model.py"
        )

        raise SystemExit

    print(
        "\nLoading prediction model..."
    )

    artifact = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    return artifact


# ============================================================
# INPUT HELPERS
# ============================================================

def get_text_input(
    prompt,
    default=None
):

    if default is not None:

        value = input(
            f"{prompt} "
            f"[{default}]: "
        ).strip()

        if value == "":
            return default

        return value

    while True:

        value = input(
            f"{prompt}: "
        ).strip()

        if value:
            return value

        print(
            "Please enter a value."
        )


def get_float_input(
    prompt,
    minimum=None
):

    while True:

        value = input(
            f"{prompt}: "
        ).strip()

        try:

            number = float(value)

            if minimum is not None and number <= minimum:

                print(
                    f"Please enter a value "
                    f"greater than {minimum}."
                )

                continue

            return number

        except ValueError:

            print(
                "Please enter a valid number."
            )


# ============================================================
# GET PROPERTY DETAILS
# ============================================================

def get_property_details():

    print("\n" + "=" * 70)

    print(
        "PROPERTY DETAILS"
    )

    print("=" * 70)

    print(
        "\nEnter the details of the property "
        "you want to estimate."
    )

    print(
        "\nYou can type 'Unknown' when you "
        "do not know a value."
    )

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    area = get_text_input(
        "\nArea"
    )

    project = get_text_input(
        "Project"
    )

    # --------------------------------------------------------
    # Property information
    # --------------------------------------------------------

    property_type = get_text_input(
        "Property type"
    )

    rooms = get_text_input(
        "Rooms"
    )

    property_area = get_float_input(
        "Property area (m²)",
        minimum=0
    )

    # --------------------------------------------------------
    # Transaction information
    # --------------------------------------------------------

    procedure = get_text_input(
        "Procedure",
        default="Sale"
    )

    freehold = get_text_input(
        "Freehold status",
        default="Free Hold"
    )

    # --------------------------------------------------------
    # Nearby locations
    # --------------------------------------------------------

    nearest_metro = get_text_input(
        "Nearest metro"
    )

    nearest_mall = get_text_input(
        "Nearest mall"
    )

    nearest_landmark = get_text_input(
        "Nearest landmark"
    )

    return {
        "AREA_EN": area,
        "PROP_SB_TYPE_EN": property_type,
        "ROOMS_EN": rooms,
        "PROJECT_EN": project,
        "PROCEDURE_EN": procedure,
        "IS_FREE_HOLD_EN": freehold,
        "ACTUAL_AREA": property_area,
        "NEAREST_METRO_EN": nearest_metro,
        "NEAREST_MALL_EN": nearest_mall,
        "NEAREST_LANDMARK_EN": nearest_landmark
    }


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def prepare_prediction_input(
    property_data,
    artifact
):

    lookups = artifact["lookups"]

    data = pd.DataFrame(
        [property_data]
    )

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

    # Use the latest date available in the
    # training dataset as the reference date.

    reference_date = pd.Timestamp(
        artifact["metadata"]["dataset_end"]
    )

    data["transaction_month"] = (
        reference_date.month
    )

    data["transaction_day"] = (
        reference_date.day
    )

    data["transaction_day_of_week"] = (
        reference_date.dayofweek
    )

    data["transaction_day_of_year"] = (
        reference_date.dayofyear
    )

    # --------------------------------------------------------
    # Historical project price
    # --------------------------------------------------------

    data[
        "historical_project_ppsqm"
    ] = data[
        "PROJECT_EN"
    ].map(
        lookups["project_ppsqm"]
    )

    # --------------------------------------------------------
    # Historical area price
    # --------------------------------------------------------

    data[
        "historical_area_ppsqm"
    ] = data[
        "AREA_EN"
    ].map(
        lookups["area_ppsqm"]
    )

    # --------------------------------------------------------
    # Historical property type price
    # --------------------------------------------------------

    data[
        "historical_property_type_ppsqm"
    ] = data[
        "PROP_SB_TYPE_EN"
    ].map(
        lookups["property_type_ppsqm"]
    )

    # --------------------------------------------------------
    # Overall market price
    # --------------------------------------------------------

    data[
        "historical_market_ppsqm"
    ] = lookups[
        "market_ppsqm"
    ]

    # --------------------------------------------------------
    # Historical transaction counts
    # --------------------------------------------------------

    data[
        "historical_project_count"
    ] = data[
        "PROJECT_EN"
    ].map(
        lookups["project_count"]
    )

    data[
        "historical_area_count"
    ] = data[
        "AREA_EN"
    ].map(
        lookups["area_count"]
    )

    data[
        "historical_property_type_count"
    ] = data[
        "PROP_SB_TYPE_EN"
    ].map(
        lookups["property_type_count"]
    )

    data[
        "historical_market_count"
    ] = lookups[
        "market_count"
    ]

    # --------------------------------------------------------
    # Best available historical price
    # --------------------------------------------------------

    data[
        "historical_best_ppsqm"
    ] = (
        data[
            "historical_project_ppsqm"
        ]
        .fillna(
            data[
                "historical_area_ppsqm"
            ]
        )
        .fillna(
            data[
                "historical_property_type_ppsqm"
            ]
        )
        .fillna(
            data[
                "historical_market_ppsqm"
            ]
        )
    )

    # --------------------------------------------------------
    # Historical price source
    # --------------------------------------------------------

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
            ].notna()
        ],
        [
            "project",
            "area",
            "property_type"
        ],
        default="overall_market"
    )

    # --------------------------------------------------------
    # Select model features
    # --------------------------------------------------------

    feature_columns = (
        artifact["categorical_columns"]
        +
        artifact["numerical_columns"]
    )

    data = data[
        feature_columns
    ]

    return data


# ============================================================
# PREDICT PRICE
# ============================================================

def predict_price(
    property_data,
    artifact
):

    data = prepare_prediction_input(
        property_data,
        artifact
    )

    preprocessor = artifact[
        "preprocessor"
    ]

    model = artifact[
        "model"
    ]

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed_data = (
        preprocessor.transform(
            data
        )
    )

    # --------------------------------------------------------
    # Predict log price
    # --------------------------------------------------------

    log_prediction = model.predict(
        processed_data
    )

    # --------------------------------------------------------
    # Convert back to AED
    # --------------------------------------------------------

    prediction = np.expm1(
        log_prediction[0]
    )

    prediction = max(
        prediction,
        0
    )

    return prediction, data


# ============================================================
# FORMAT AED
# ============================================================

def format_aed(value):

    if value >= 1_000_000:

        return (
            f"AED "
            f"{value / 1_000_000:.2f}M"
        )

    if value >= 1_000:

        return (
            f"AED "
            f"{value / 1_000:.0f}K"
        )

    return (
        f"AED "
        f"{value:,.0f}"
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_prediction(
    property_data,
    prediction,
    prepared_data,
    artifact
):

    lookups = artifact[
        "lookups"
    ]

    print("\n" + "=" * 70)

    print(
        "PROPERTY PRICE ESTIMATE"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Property summary
    # --------------------------------------------------------

    print("\nProperty")

    print(
        f"  Area:           "
        f"{property_data['AREA_EN']}"
    )

    print(
        f"  Property type:  "
        f"{property_data['PROP_SB_TYPE_EN']}"
    )

    print(
        f"  Rooms:          "
        f"{property_data['ROOMS_EN']}"
    )

    print(
        f"  Size:           "
        f"{property_data['ACTUAL_AREA']:,.2f} m²"
    )

    print(
        f"  Project:        "
        f"{property_data['PROJECT_EN']}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "ESTIMATED TRANSACTION VALUE"
    )

    print(
        "-" * 70
    )

    print(
        f"\n  {format_aed(prediction)}"
    )

    print(
        f"\n  AED {prediction:,.0f}"
    )

    # --------------------------------------------------------
    # Market information
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "MARKET INFORMATION"
    )

    print(
        "-" * 70
    )

    historical_area_price = (
        prepared_data[
            "historical_area_ppsqm"
        ].iloc[0]
    )

    historical_project_price = (
        prepared_data[
            "historical_project_ppsqm"
        ].iloc[0]
    )

    historical_best_price = (
        prepared_data[
            "historical_best_ppsqm"
        ].iloc[0]
    )

    # Historical area price

    if pd.notna(
        historical_area_price
    ):

        print(
            f"\n  Area median price/m²: "
            f"AED {historical_area_price:,.0f}"
        )

    # Historical project price

    if pd.notna(
        historical_project_price
    ):

        print(
            f"  Project median price/m²: "
            f"AED {historical_project_price:,.0f}"
        )

    # Best historical price

    print(
        f"  Reference price/m²: "
        f"AED {historical_best_price:,.0f}"
    )

    # Estimated market value based purely
    # on historical price/m².

    market_value = (
        property_data[
            "ACTUAL_AREA"
        ]
        *
        historical_best_price
    )

    print(
        f"\n  Area × reference price/m²: "
        f"{format_aed(market_value)}"
    )

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "MODEL INFORMATION"
    )

    print(
        "-" * 70
    )

    print(
        f"\n  Model: "
        f"{artifact['metadata']['model']}"
    )

    print(
        f"  Training transactions: "
        f"{artifact['metadata']['dataset_rows']:,}"
    )

    print(
        f"  Data period: "
        f"{artifact['metadata']['dataset_start']}"
        f" to "
        f"{artifact['metadata']['dataset_end']}"
    )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "       DUBAI PROPERTY PRICE PREDICTION"
    )

    print("=" * 70)

    print(
        "\nEstimate the transaction value "
        "of a Dubai property."
    )

    artifact = load_model()

    property_data = (
        get_property_details()
    )

    print(
        "\nCalculating estimate..."
    )

    prediction, prepared_data = (
        predict_price(
            property_data,
            artifact
        )
    )

    display_prediction(
        property_data,
        prediction,
        prepared_data,
        artifact
    )


if __name__ == "__main__":
    main()