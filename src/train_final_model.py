from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from data_loader import load_data
from preprocessing import prepare_data


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "property_price_model.pkl"


# ============================================================
# SETTINGS
# ============================================================

TARGET_COLUMN = "TRANS_VALUE"

RANDOM_STATE = 42


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

CATEGORICAL_COLUMNS = [
    "PROCEDURE_EN",
    "IS_FREE_HOLD_EN",
    "AREA_EN",
    "PROP_SB_TYPE_EN",
    "ROOMS_EN",
    "NEAREST_METRO_EN",
    "NEAREST_MALL_EN",
    "NEAREST_LANDMARK_EN",
    "PROJECT_EN",
    "historical_price_source"
]

NUMERICAL_COLUMNS = [
    "ACTUAL_AREA",

    "transaction_month",
    "transaction_day",
    "transaction_day_of_week",
    "transaction_day_of_year",

    "historical_project_ppsqm",
    "historical_area_ppsqm",
    "historical_property_type_ppsqm",
    "historical_market_ppsqm",
    "historical_best_ppsqm",

    "historical_project_count",
    "historical_area_count",
    "historical_property_type_count",
    "historical_market_count"
]


# ============================================================
# CREATE MARKET LOOKUPS
# ============================================================

def create_market_lookups(df):
    """
    Create market statistics from the complete dataset.

    These statistics will be used by the application when
    predicting the price of a new property.
    """

    data = df.copy()

    data["INSTANCE_DATE"] = pd.to_datetime(
        data["INSTANCE_DATE"],
        errors="coerce"
    )

    # Calculate price per square metre.
    #
    # This is NOT used as an input feature directly.
    # It is only used to create historical market statistics.
    data["calculated_ppsqm"] = (
        data["TRANS_VALUE"] /
        data["ACTUAL_AREA"]
    )

    data["calculated_ppsqm"] = (
        data["calculated_ppsqm"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    data = data[
        data["ACTUAL_AREA"] > 0
    ].copy()

    # --------------------------------------------------------
    # Project-level median price/m²
    # --------------------------------------------------------

    project_ppsqm = (
        data
        .dropna(subset=["PROJECT_EN"])
        .groupby("PROJECT_EN")["calculated_ppsqm"]
        .median()
        .to_dict()
    )

    # --------------------------------------------------------
    # Area-level median price/m²
    # --------------------------------------------------------

    area_ppsqm = (
        data
        .groupby("AREA_EN")["calculated_ppsqm"]
        .median()
        .to_dict()
    )

    # --------------------------------------------------------
    # Property-type median price/m²
    # --------------------------------------------------------

    property_type_ppsqm = (
        data
        .groupby("PROP_SB_TYPE_EN")["calculated_ppsqm"]
        .median()
        .to_dict()
    )

    # --------------------------------------------------------
    # Overall market median price/m²
    # --------------------------------------------------------

    market_ppsqm = (
        data["calculated_ppsqm"]
        .median()
    )

    # --------------------------------------------------------
    # Historical transaction counts
    # --------------------------------------------------------

    project_count = (
        data
        .dropna(subset=["PROJECT_EN"])
        .groupby("PROJECT_EN")
        .size()
        .to_dict()
    )

    area_count = (
        data
        .groupby("AREA_EN")
        .size()
        .to_dict()
    )

    property_type_count = (
        data
        .groupby("PROP_SB_TYPE_EN")
        .size()
        .to_dict()
    )

    market_count = len(data)

    return {
        "project_ppsqm": project_ppsqm,
        "area_ppsqm": area_ppsqm,
        "property_type_ppsqm": property_type_ppsqm,
        "market_ppsqm": market_ppsqm,

        "project_count": project_count,
        "area_count": area_count,
        "property_type_count": property_type_count,
        "market_count": market_count
    }


# ============================================================
# ADD DEPLOYMENT FEATURES
# ============================================================

def add_deployment_features(df, lookups):
    """
    Add historical market features to a dataframe.

    These features are generated from the saved market
    lookup tables.
    """

    data = df.copy()

    # --------------------------------------------------------
    # Historical price/m²
    # --------------------------------------------------------

    data["historical_project_ppsqm"] = (
        data["PROJECT_EN"]
        .map(lookups["project_ppsqm"])
    )

    data["historical_area_ppsqm"] = (
        data["AREA_EN"]
        .map(lookups["area_ppsqm"])
    )

    data["historical_property_type_ppsqm"] = (
        data["PROP_SB_TYPE_EN"]
        .map(lookups["property_type_ppsqm"])
    )

    data["historical_market_ppsqm"] = (
        lookups["market_ppsqm"]
    )

    # --------------------------------------------------------
    # Historical transaction counts
    # --------------------------------------------------------

    data["historical_project_count"] = (
        data["PROJECT_EN"]
        .map(lookups["project_count"])
    )

    data["historical_area_count"] = (
        data["AREA_EN"]
        .map(lookups["area_count"])
    )

    data["historical_property_type_count"] = (
        data["PROP_SB_TYPE_EN"]
        .map(lookups["property_type_count"])
    )

    data["historical_market_count"] = (
        lookups["market_count"]
    )

    # --------------------------------------------------------
    # Best available historical price
    # --------------------------------------------------------

    data["historical_best_ppsqm"] = (
        data["historical_project_ppsqm"]
        .fillna(
            data["historical_area_ppsqm"]
        )
        .fillna(
            data["historical_property_type_ppsqm"]
        )
        .fillna(
            data["historical_market_ppsqm"]
        )
    )

    # --------------------------------------------------------
    # Identify which historical source was used
    # --------------------------------------------------------

    data["historical_price_source"] = np.select(
        [
            data["historical_project_ppsqm"].notna(),
            data["historical_area_ppsqm"].notna(),
            data["historical_property_type_ppsqm"].notna()
        ],
        [
            "project",
            "area",
            "property_type"
        ],
        default="overall_market"
    )

    return data


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor():

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_COLUMNS
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS
            )
        ]
    )

    return preprocessor


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("DUBAI PROPERTY PRICE PREDICTION")
    print("FINAL DEPLOYMENT MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    df = load_data()

    print(
        "\nTraining deployment model using "
        f"all {len(df):,} transactions."
    )

    # --------------------------------------------------------
    # 2. Prepare raw data
    # --------------------------------------------------------

    print("\nPreparing data...")

    prepared_df = prepare_data(df)

    # --------------------------------------------------------
    # 3. Create market lookup tables
    # --------------------------------------------------------

    print("\nCreating market statistics...")

    lookups = create_market_lookups(df)

    print(
        f"Projects: "
        f"{len(lookups['project_ppsqm']):,}"
    )

    print(
        f"Areas: "
        f"{len(lookups['area_ppsqm']):,}"
    )

    print(
        f"Property types: "
        f"{len(lookups['property_type_ppsqm']):,}"
    )

    print(
        f"Overall median price/m²: "
        f"AED {lookups['market_ppsqm']:,.2f}"
    )

    # --------------------------------------------------------
    # 4. Add historical features
    # --------------------------------------------------------

    print("\nAdding market features...")

    prepared_df = add_deployment_features(
        prepared_df,
        lookups
    )

    # --------------------------------------------------------
    # 5. Separate X and y
    # --------------------------------------------------------

    X = prepared_df[
        CATEGORICAL_COLUMNS +
        NUMERICAL_COLUMNS
    ].copy()

    y = prepared_df[
        TARGET_COLUMN
    ].copy()

    # --------------------------------------------------------
    # 6. Build preprocessing pipeline
    # --------------------------------------------------------

    print("\nBuilding preprocessing pipeline...")

    preprocessor = build_preprocessor()

    X_processed = preprocessor.fit_transform(
        X
    )

    print(
        f"Processed feature matrix: "
        f"{X_processed.shape}"
    )

    # --------------------------------------------------------
    # 7. Build final XGBoost model
    # --------------------------------------------------------

    print("\nTraining final XGBoost model...")

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # --------------------------------------------------------
    # 8. Log-transform target
    # --------------------------------------------------------

    log_y = np.log1p(y)

    # --------------------------------------------------------
    # 9. Train
    # --------------------------------------------------------

    model.fit(
        X_processed,
        log_y
    )

    # --------------------------------------------------------
    # 10. Create deployment artifact
    # --------------------------------------------------------

    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "lookups": lookups,

        "categorical_columns":
            CATEGORICAL_COLUMNS,

        "numerical_columns":
            NUMERICAL_COLUMNS,

        "target_column":
            TARGET_COLUMN,

        "target_transformation":
            "log1p",

        "metadata": {
            "dataset_rows": len(df),

            "dataset_start":
                str(
                    pd.to_datetime(
                        df["INSTANCE_DATE"]
                    ).min().date()
                ),

            "dataset_end":
                str(
                    pd.to_datetime(
                        df["INSTANCE_DATE"]
                    ).max().date()
                ),

            "model":
                "XGBoost",

            "n_estimators":
                500,

            "learning_rate":
                0.05,

            "max_depth":
                8,

            "random_state":
                RANDOM_STATE
        }
    }

    # --------------------------------------------------------
    # 11. Save artifact
    # --------------------------------------------------------

    joblib.dump(
        artifact,
        MODEL_PATH
    )

    # --------------------------------------------------------
    # 12. Confirmation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DEPLOYMENT MODEL CREATED")
    print("=" * 70)

    print(
        f"\nModel saved to:"
    )

    print(
        MODEL_PATH
    )

    print(
        f"\nTraining rows: "
        f"{len(X):,}"
    )

    print(
        f"Processed features: "
        f"{X_processed.shape[1]:,}"
    )

    print(
        "\nThe model artifact contains:"
    )

    print(
        "  ✓ XGBoost model"
    )

    print(
        "  ✓ Preprocessing pipeline"
    )

    print(
        "  ✓ Historical market statistics"
    )

    print(
        "  ✓ Feature configuration"
    )

    print(
        "  ✓ Dataset metadata"
    )

    print(
        "\nReady to build the prediction application."
    )


if __name__ == "__main__":
    main()