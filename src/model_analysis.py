import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

from data_loader import load_data
from preprocessing import prepare_data
from historical_features import add_historical_price_features


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

TARGET_COLUMN = "TRANS_VALUE"
RANDOM_STATE = 42

TRAIN_END_DATE = pd.Timestamp("2026-05-17")
TEST_START_DATE = pd.Timestamp("2026-05-18")


# ============================================================
# LOAD DATA
# ============================================================

def load_and_prepare():

    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df = load_data()

    df["INSTANCE_DATE"] = pd.to_datetime(
        df["INSTANCE_DATE"],
        errors="coerce"
    )

    df = df.sort_values(
        "INSTANCE_DATE"
    ).reset_index(drop=True)

    print(
        f"\nFull dataset period: "
        f"{df['INSTANCE_DATE'].min().date()} → "
        f"{df['INSTANCE_DATE'].max().date()}"
    )

    return df


# ============================================================
# TIME-BASED SPLIT
# ============================================================

def create_time_split(df):

    print("\n" + "=" * 60)
    print("DATE-BASED TRAIN / TEST SPLIT")
    print("=" * 60)

    train_df = df[
        df["INSTANCE_DATE"] <= TRAIN_END_DATE
    ].copy()

    test_df = df[
        df["INSTANCE_DATE"] >= TEST_START_DATE
    ].copy()

    print(
        f"\nTraining rows: {len(train_df):,}"
    )

    print(
        f"Testing rows:  {len(test_df):,}"
    )

    print(
        f"\nTraining period: "
        f"{train_df['INSTANCE_DATE'].min().date()} → "
        f"{train_df['INSTANCE_DATE'].max().date()}"
    )

    print(
        f"Testing period:  "
        f"{test_df['INSTANCE_DATE'].min().date()} → "
        f"{test_df['INSTANCE_DATE'].max().date()}"
    )

    return train_df, test_df


# ============================================================
# HISTORICAL MARKET FEATURES
# ============================================================

def add_train_history_to_test(
    train_df,
    test_df
):

    print("\n" + "=" * 60)
    print("CREATING HISTORICAL MARKET FEATURES")
    print("=" * 60)

    # --------------------------------------------------------
    # Training historical features
    # --------------------------------------------------------

    print(
        "\nProcessing training history..."
    )

    train_with_history = (
        add_historical_price_features(
            train_df
        )
    )

    # --------------------------------------------------------
    # Calculate price/m² using training data only
    # --------------------------------------------------------

    train_ppsqm = (
        train_df["TRANS_VALUE"]
        / train_df["ACTUAL_AREA"]
    )

    train_ppsqm = train_ppsqm.replace(
        [np.inf, -np.inf],
        np.nan
    )

    history_df = train_df.copy()

    history_df["_ppsqm"] = train_ppsqm

    # --------------------------------------------------------
    # Project history
    # --------------------------------------------------------

    project_history = (
        history_df
        .groupby("PROJECT_EN")["_ppsqm"]
        .median()
    )

    # --------------------------------------------------------
    # Area history
    # --------------------------------------------------------

    area_history = (
        history_df
        .groupby("AREA_EN")["_ppsqm"]
        .median()
    )

    # --------------------------------------------------------
    # Property type history
    # --------------------------------------------------------

    property_type_history = (
        history_df
        .groupby(
            "PROP_SB_TYPE_EN"
        )["_ppsqm"]
        .median()
    )

    # --------------------------------------------------------
    # Overall market history
    # --------------------------------------------------------

    overall_history = (
        history_df["_ppsqm"].median()
    )

    # --------------------------------------------------------
    # Apply historical information to test data
    # --------------------------------------------------------

    test_with_history = test_df.copy()

    test_with_history[
        "historical_project_ppsqm"
    ] = test_with_history[
        "PROJECT_EN"
    ].map(project_history)

    test_with_history[
        "historical_area_ppsqm"
    ] = test_with_history[
        "AREA_EN"
    ].map(area_history)

    test_with_history[
        "historical_property_type_ppsqm"
    ] = test_with_history[
        "PROP_SB_TYPE_EN"
    ].map(
        property_type_history
    )

    test_with_history[
        "historical_market_ppsqm"
    ] = overall_history

    return (
        train_with_history,
        test_with_history
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    train_df,
    test_df
):

    y_train = train_df[
        TARGET_COLUMN
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].copy()

    X_train = train_df.drop(
        columns=[TARGET_COLUMN]
    ).copy()

    X_test = test_df.drop(
        columns=[TARGET_COLUMN]
    ).copy()

    # Add target temporarily because
    # prepare_data() expects the original schema.
    X_train_temp = X_train.copy()
    X_train_temp[
        TARGET_COLUMN
    ] = y_train.values

    X_test_temp = X_test.copy()
    X_test_temp[
        TARGET_COLUMN
    ] = y_test.values

    X_train = prepare_data(
        X_train_temp
    ).drop(
        columns=[TARGET_COLUMN]
    )

    X_test = prepare_data(
        X_test_temp
    ).drop(
        columns=[TARGET_COLUMN]
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(X_train):

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

    numerical_columns = [
        column
        for column in X_train.columns
        if column not in categorical_columns
    ]

    print("\n" + "=" * 60)
    print("FEATURE TYPES")
    print("=" * 60)

    print("\nCategorical columns:")

    for column in categorical_columns:
        print(f"  - {column}")

    print("\nNumerical columns:")

    for column in numerical_columns:
        print(f"  - {column}")

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
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            ),
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            )
        ]
    )

    return preprocessor


# ============================================================
# TRAIN XGBOOST
# ============================================================

def train_model(
    preprocessor,
    X_train,
    y_train
):

    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

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

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    y_train_log = np.log1p(
        y_train
    )

    print(
        "\nTraining with "
        "log-transformed target..."
    )

    pipeline.fit(
        X_train,
        y_train_log
    )

    print(
        "Training complete."
    )

    return pipeline


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    pipeline,
    X_test,
    y_test
):

    print(
        "\nGenerating predictions..."
    )

    predictions_log = pipeline.predict(
        X_test
    )

    predictions = np.expm1(
        predictions_log
    )

    predictions = np.maximum(
        predictions,
        0
    )

    results = X_test.copy()

    results[
        "actual_value"
    ] = y_test.values

    results[
        "predicted_value"
    ] = predictions

    results[
        "absolute_error"
    ] = np.abs(
        results["actual_value"]
        -
        results["predicted_value"]
    )

    results[
        "percentage_error"
    ] = (
        results["absolute_error"]
        /
        results["actual_value"]
        *
        100
    )

    return results


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def print_model_performance(
    results
):

    mae = mean_absolute_error(
        results["actual_value"],
        results["predicted_value"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            results["actual_value"],
            results["predicted_value"]
        )
    )

    r2 = r2_score(
        results["actual_value"],
        results["predicted_value"]
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"\nMAE:  AED {mae:,.2f}"
    )

    print(
        f"RMSE: AED {rmse:,.2f}"
    )

    print(
        f"R²:   {r2:.4f}"
    )


# ============================================================
# LARGEST ERRORS
# ============================================================

def inspect_largest_errors(
    results
):

    print("\n" + "=" * 60)
    print("LARGEST PREDICTION ERRORS")
    print("=" * 60)

    columns = [
        "actual_value",
        "predicted_value",
        "absolute_error",
        "percentage_error",
        "ACTUAL_AREA",
        "AREA_EN",
        "PROP_SB_TYPE_EN",
        "ROOMS_EN",
        "PROJECT_EN"
    ]

    available_columns = [
        column
        for column in columns
        if column in results.columns
    ]

    largest_errors = (
        results
        .sort_values(
            "absolute_error",
            ascending=False
        )
        .head(20)
    )

    print(
        largest_errors[
            available_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# ERROR BY PROPERTY TYPE
# ============================================================

def analyze_property_type_errors(
    results
):

    print("\n" + "=" * 60)
    print("ERROR BY PROPERTY TYPE")
    print("=" * 60)

    summary = (
        results
        .groupby(
            "PROP_SB_TYPE_EN"
        )
        .agg(
            transactions=(
                "actual_value",
                "count"
            ),
            mean_actual_value=(
                "actual_value",
                "mean"
            ),
            mean_absolute_error=(
                "absolute_error",
                "mean"
            ),
            median_absolute_error=(
                "absolute_error",
                "median"
            ),
            mean_percentage_error=(
                "percentage_error",
                "mean"
            )
        )
        .sort_values(
            "mean_absolute_error",
            ascending=False
        )
    )

    print(
        summary.to_string()
    )

    output_path = (
        REPORT_DIR
        / "improved_error_by_property_type.csv"
    )

    summary.to_csv(
        output_path
    )

    print(
        f"\nSaved to: {output_path}"
    )


# ============================================================
# ERROR BY PRICE SEGMENT
# ============================================================

def analyze_price_segments(
    results
):

    print("\n" + "=" * 60)
    print("ERROR BY PRICE SEGMENT")
    print("=" * 60)

    bins = [
        0,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        np.inf
    ]

    labels = [
        "< AED 500K",
        "AED 500K–1M",
        "AED 1M–2M",
        "AED 2M–5M",
        "AED 5M–10M",
        "> AED 10M"
    ]

    results = results.copy()

    results[
        "price_segment"
    ] = pd.cut(
        results["actual_value"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    summary = (
        results
        .groupby(
            "price_segment",
            observed=True
        )
        .agg(
            transactions=(
                "actual_value",
                "count"
            ),
            mean_actual_value=(
                "actual_value",
                "mean"
            ),
            mean_absolute_error=(
                "absolute_error",
                "mean"
            ),
            median_absolute_error=(
                "absolute_error",
                "median"
            ),
            mean_percentage_error=(
                "percentage_error",
                "mean"
            )
        )
    )

    print(
        summary.to_string()
    )

    output_path = (
        REPORT_DIR
        / "improved_error_by_price_segment.csv"
    )

    summary.to_csv(
        output_path
    )

    print(
        f"\nSaved to: {output_path}"
    )


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

def plot_actual_vs_predicted(
    results
):

    plt.figure(
        figsize=(10, 7)
    )

    plt.scatter(
        results["actual_value"],
        results["predicted_value"],
        alpha=0.35
    )

    max_value = max(
        results["actual_value"].max(),
        results["predicted_value"].max()
    )

    plt.plot(
        [0, max_value],
        [0, max_value],
        linestyle="--"
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(
        "Actual Transaction Value (AED)"
    )

    plt.ylabel(
        "Predicted Transaction Value (AED)"
    )

    plt.title(
        "XGBoost: Actual vs Predicted Property Values"
    )

    plt.tight_layout()

    output_path = (
        REPORT_DIR
        / "improved_actual_vs_predicted.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nSaved plot: {output_path}"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def analyze_feature_importance(
    pipeline
):

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    preprocessor = (
        pipeline
        .named_steps[
            "preprocessor"
        ]
    )

    model = (
        pipeline
        .named_steps[
            "model"
        ]
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance = (
        model.feature_importances_
    )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print(
        "\nTop 30 features:"
    )

    print(
        feature_importance
        .head(30)
        .to_string(
            index=False
        )
    )

    output_path = (
        REPORT_DIR
        / "improved_feature_importance.csv"
    )

    feature_importance.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved to: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # 1. Load data
    df = load_and_prepare()

    # 2. Chronological split
    train_df, test_df = (
        create_time_split(df)
    )

    # 3. Add historical market features
    (
        train_df,
        test_df
    ) = add_train_history_to_test(
        train_df,
        test_df
    )

    # 4. Prepare model features
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_features(
        train_df,
        test_df
    )

    # 5. Create preprocessing pipeline
    preprocessor = (
        create_preprocessor(
            X_train
        )
    )

    # 6. Train XGBoost
    pipeline = train_model(
        preprocessor,
        X_train,
        y_train
    )

    # 7. Generate predictions
    results = generate_predictions(
        pipeline,
        X_test,
        y_test
    )

    # 8. Overall performance
    print_model_performance(
        results
    )

    # 9. Error analysis
    inspect_largest_errors(
        results
    )

    analyze_property_type_errors(
        results
    )

    analyze_price_segments(
        results
    )

    # 10. Visual analysis
    plot_actual_vs_predicted(
        results
    )

    # 11. Feature importance
    analyze_feature_importance(
        pipeline
    )


if __name__ == "__main__":
    main()