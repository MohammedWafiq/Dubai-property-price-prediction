import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from data_loader import load_data
from historical_features import add_historical_price_features
from preprocessing import prepare_data


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TARGET_COLUMN = "TRANS_VALUE"

TRAIN_END_DATE = "2026-05-16"

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

    # Date features
    "transaction_month",
    "transaction_day",
    "transaction_day_of_week",
    "transaction_day_of_year",

    # Historical price features
    "historical_project_ppsqm",
    "historical_area_ppsqm",
    "historical_property_type_ppsqm",
    "historical_market_ppsqm",
    "historical_best_ppsqm",

    # Historical reliability/count features
    "historical_project_count",
    "historical_area_count",
    "historical_property_type_count",
    "historical_market_count"
]


# ============================================================
# CREATE TRAINING-ONLY HISTORICAL FEATURES
# ============================================================

def create_training_historical_features(train_df, test_df):
    """
    Create historical market features without target leakage.

    Training:
        Each transaction only uses information from earlier dates.

    Test:
        Historical information comes ONLY from the training period.

    No target values from the test period are used.
    """

    print("\nCreating historical features...")

    # --------------------------------------------------------
    # Add historical features to the training data.
    #
    # The historical_features module already ensures that
    # each training transaction only sees previous dates.
    # --------------------------------------------------------

    train_with_history = add_historical_price_features(
        train_df
    )

    # --------------------------------------------------------
    # Calculate training-period historical statistics.
    #
    # These are used to generate historical features for the
    # test period.
    # --------------------------------------------------------

    training = train_df.copy()

    training["INSTANCE_DATE"] = pd.to_datetime(
        training["INSTANCE_DATE"],
        errors="coerce"
    )

    training["_date"] = training[
        "INSTANCE_DATE"
    ].dt.date

    training["_ppsqm"] = (
        training["TRANS_VALUE"]
        / training["ACTUAL_AREA"]
    )

    training["_ppsqm"] = training[
        "_ppsqm"
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Last historical price per group
    # --------------------------------------------------------

    def get_group_median(column):
        return (
            training
            .groupby(
                column,
                dropna=False
            )["_ppsqm"]
            .median()
        )

    project_price = get_group_median(
        "PROJECT_EN"
    )

    area_price = get_group_median(
        "AREA_EN"
    )

    property_type_price = get_group_median(
        "PROP_SB_TYPE_EN"
    )

    market_price = training[
        "_ppsqm"
    ].median()

    # --------------------------------------------------------
    # Historical transaction counts
    # --------------------------------------------------------

    project_count = (
        training
        .groupby(
            "PROJECT_EN",
            dropna=False
        )
        .size()
    )

    area_count = (
        training
        .groupby(
            "AREA_EN",
            dropna=False
        )
        .size()
    )

    property_type_count = (
        training
        .groupby(
            "PROP_SB_TYPE_EN",
            dropna=False
        )
        .size()
    )

    market_count = len(training)

    # --------------------------------------------------------
    # Create test historical features
    # --------------------------------------------------------

    test_with_history = test_df.copy()

    test_with_history[
        "historical_project_ppsqm"
    ] = test_with_history[
        "PROJECT_EN"
    ].map(project_price)

    test_with_history[
        "historical_area_ppsqm"
    ] = test_with_history[
        "AREA_EN"
    ].map(area_price)

    test_with_history[
        "historical_property_type_ppsqm"
    ] = test_with_history[
        "PROP_SB_TYPE_EN"
    ].map(property_type_price)

    test_with_history[
        "historical_market_ppsqm"
    ] = market_price

    # --------------------------------------------------------
    # Historical counts
    # --------------------------------------------------------

    test_with_history[
        "historical_project_count"
    ] = test_with_history[
        "PROJECT_EN"
    ].map(project_count)

    test_with_history[
        "historical_area_count"
    ] = test_with_history[
        "AREA_EN"
    ].map(area_count)

    test_with_history[
        "historical_property_type_count"
    ] = test_with_history[
        "PROP_SB_TYPE_EN"
    ].map(property_type_count)

    test_with_history[
        "historical_market_count"
    ] = market_count

    # --------------------------------------------------------
    # Hierarchical historical price
    # --------------------------------------------------------

    test_with_history[
        "historical_best_ppsqm"
    ] = (
        test_with_history[
            "historical_project_ppsqm"
        ]
        .fillna(
            test_with_history[
                "historical_area_ppsqm"
            ]
        )
        .fillna(
            test_with_history[
                "historical_property_type_ppsqm"
            ]
        )
        .fillna(
            test_with_history[
                "historical_market_ppsqm"
            ]
        )
    )

    # --------------------------------------------------------
    # Identify historical price source
    # --------------------------------------------------------

    test_with_history[
        "historical_price_source"
    ] = np.select(
        [
            test_with_history[
                "historical_project_ppsqm"
            ].notna(),

            test_with_history[
                "historical_area_ppsqm"
            ].notna(),

            test_with_history[
                "historical_property_type_ppsqm"
            ].notna(),

            test_with_history[
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

    return (
        train_with_history,
        test_with_history
    )


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
# BUILD MODELS
# ============================================================

def build_models():

    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "XGBoost":
            XGBRegressor(
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
    }

    return models


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test
):

    # --------------------------------------------------------
    # Log-transform target
    # --------------------------------------------------------

    log_y_train = np.log1p(
        y_train
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        log_y_train
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    log_predictions = model.predict(
        X_test
    )

    predictions = np.expm1(
        log_predictions
    )

    # Avoid negative predictions
    predictions = np.maximum(
        predictions,
        0
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "predictions": predictions
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "DUBAI PROPERTY PRICE PREDICTION"
    )

    print(
        "STRICT TIME-BASED EVALUATION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    df["INSTANCE_DATE"] = pd.to_datetime(
        df["INSTANCE_DATE"],
        errors="coerce"
    )

    df = df.sort_values(
        "INSTANCE_DATE"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    train_end = pd.Timestamp(
        TRAIN_END_DATE
    )

    train_df = df[
        df["INSTANCE_DATE"]
        <= train_end
    ].copy()

    test_df = df[
        df["INSTANCE_DATE"]
        > train_end
    ].copy()

    print(
        "\nChronological split:"
    )

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    print(
        f"Training period: "
        f"{train_df['INSTANCE_DATE'].min().date()} "
        f"to "
        f"{train_df['INSTANCE_DATE'].max().date()}"
    )

    print(
        f"Test period: "
        f"{test_df['INSTANCE_DATE'].min().date()} "
        f"to "
        f"{test_df['INSTANCE_DATE'].max().date()}"
    )

    # --------------------------------------------------------
    # Historical features
    # --------------------------------------------------------

    (
        train_df,
        test_df
    ) = create_training_historical_features(
        train_df,
        test_df
    )

    # --------------------------------------------------------
    # Standard preprocessing
    # --------------------------------------------------------

    train_prepared = prepare_data(
        train_df
    )

    test_prepared = prepare_data(
        test_df
    )

    # --------------------------------------------------------
    # Ensure required historical features survive
    # --------------------------------------------------------

    historical_columns = [
        "historical_project_ppsqm",
        "historical_area_ppsqm",
        "historical_property_type_ppsqm",
        "historical_market_ppsqm",
        "historical_best_ppsqm",
        "historical_project_count",
        "historical_area_count",
        "historical_property_type_count",
        "historical_market_count",
        "historical_price_source"
    ]

    for column in historical_columns:

        if column not in train_prepared.columns:
            raise ValueError(
                f"Missing historical feature: "
                f"{column}"
            )

        if column not in test_prepared.columns:
            raise ValueError(
                f"Missing historical feature: "
                f"{column}"
            )

    # --------------------------------------------------------
    # Separate target
    # --------------------------------------------------------

    y_train = train_prepared[
        TARGET_COLUMN
    ]

    y_test = test_prepared[
        TARGET_COLUMN
    ]

    X_train = train_prepared.drop(
        columns=[TARGET_COLUMN]
    )

    X_test = test_prepared.drop(
        columns=[TARGET_COLUMN]
    )

    # --------------------------------------------------------
    # Keep only model features
    # --------------------------------------------------------

    required_features = (
        NUMERICAL_COLUMNS
        + CATEGORICAL_COLUMNS
    )

    X_train = X_train[
        required_features
    ]

    X_test = X_test[
        required_features
    ]

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    preprocessor = build_preprocessor()

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        "\nProcessed feature matrix:"
    )

    print(
        f"Training shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Test shape: "
        f"{X_test_processed.shape}"
    )

    # --------------------------------------------------------
    # Train and evaluate
    # --------------------------------------------------------

    models = build_models()

    results = []

    trained_xgb = None
    xgb_predictions = None

    for model_name, model in models.items():

        print(
            "\n" + "-" * 60
        )

        print(
            f"Training {model_name}..."
        )

        result = evaluate_model(
            model,
            X_train_processed,
            y_train,
            X_test_processed,
            y_test
        )

        print(
            f"{model_name}"
        )

        print(
            f"MAE: "
            f"AED {result['MAE']:,.2f}"
        )

        print(
            f"RMSE: "
            f"AED {result['RMSE']:,.2f}"
        )

        print(
            f"R²: "
            f"{result['R2']:.4f}"
        )

        results.append(
            {
                "Model": model_name,
                "MAE": result["MAE"],
                "RMSE": result["RMSE"],
                "R2": result["R2"]
            }
        )

        if model_name == "XGBoost":
            trained_xgb = model
            xgb_predictions = result[
                "predictions"
            ]

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_path = (
        REPORTS_DIR
        / "historical_feature_model_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to:"
    )

    print(
        results_path
    )


if __name__ == "__main__":
    main()