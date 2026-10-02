import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dubai Property Cost Calculator",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "property_price_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "dubai_residential_data_2026.csv"


# ============================================================
# LOAD MODEL
# ============================================================

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
preprocessor = artifact["preprocessor"]
lookups = artifact["lookups"]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["INSTANCE_DATE"] = pd.to_datetime(
    df["INSTANCE_DATE"],
    errors="coerce"
)

LATEST_DATE = df["INSTANCE_DATE"].max()


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       REMOVE STREAMLIT TOP HEADER
       ======================================================== */

    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }

    #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }


    /* ========================================================
       FORCE LIGHT MODE
       ======================================================== */

    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section {
        color-scheme: light !important;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(255,255,255,0.95) 0%,
                rgba(255,255,255,0.45) 22%,
                transparent 48%
            ),
            radial-gradient(
                circle at 90% 90%,
                rgba(255,255,255,0.40) 0%,
                transparent 42%
            ),
            linear-gradient(
                135deg,
                #E7F5FF 0%,
                #D3EDFF 50%,
                #B0DBFF 100%
            );

        background-attachment: fixed;
        min-height: 100vh;
    }


    /* ========================================================
       MAIN CONTENT
       ======================================================== */

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
       HEADINGS
       ======================================================== */

    h1 {
        color: #123B57 !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2,
    h3 {
        color: #173B56 !important;
        font-weight: 700 !important;
    }

    p {
        color: #3C5A6E !important;
    }

    label {
        color: #234D67 !important;
        font-weight: 600 !important;
    }

    .stCaption {
        color: #55758A !important;
    }


    /* ========================================================
       SELECTBOX
       ======================================================== */

    [data-testid="stSelectbox"] {
        color-scheme: light !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;

        border-radius: 12px !important;

        color: #173B56 !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;

        border-radius: 12px !important;

        color: #173B56 !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div > div,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div > div > div,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div > div > div > div {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;

        color: #173B56 !important;
    }


    /* ========================================================
       SELECTBOX TEXT
       ======================================================== */

    [data-testid="stSelectbox"] [data-baseweb="select"] span,
    [data-testid="stSelectbox"] [data-baseweb="select"] div {
        color: #173B56 !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] span {
        -webkit-text-fill-color: #173B56 !important;
    }


    /* ========================================================
       SELECTBOX BORDER
       ======================================================== */

    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        border: 1px solid #D5E4EE !important;

        box-shadow:
            0 3px 10px rgba(40,90,120,0.06) !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"]:hover > div {
        border-color: #9CCBE8 !important;

        background: #FFFFFF !important;
    }


    /* ========================================================
       SELECTBOX ARROW
       ======================================================== */

    [data-testid="stSelectbox"]
    [data-baseweb="select"]
    svg {
        fill: #173B56 !important;
        color: #173B56 !important;
    }

    [data-testid="stSelectbox"]
    [data-baseweb="select"] > div > div:last-child,

    [data-testid="stSelectbox"]
    [data-baseweb="select"] > div > div > div:last-child,

    [data-testid="stSelectbox"]
    [data-baseweb="select"] > div > div > div > div:last-child {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
    }


    /* ========================================================
       DROPDOWN MENU
       ======================================================== */

    [data-baseweb="popover"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
    }

    [data-baseweb="popover"] > div {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
    }

    [data-baseweb="menu"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;

        color: #173B56 !important;
    }

    [data-baseweb="menu"] > div,
    [data-baseweb="menu"] ul {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
    }


    /* ========================================================
       DROPDOWN OPTIONS
       ======================================================== */

    [data-baseweb="menu"] [role="option"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;

        color: #173B56 !important;
    }

    [data-baseweb="menu"] [role="option"] * {
        background: transparent !important;

        color: #173B56 !important;

        -webkit-text-fill-color: #173B56 !important;
    }

    [data-baseweb="menu"] [role="option"]:hover {
        background: #E7F5FF !important;
        background-color: #E7F5FF !important;
    }

    [data-baseweb="menu"] [role="option"][aria-selected="true"] {
        background: #F4FAFE !important;
        background-color: #F4FAFE !important;
    }


    /* ========================================================
       NUMBER INPUT
       ======================================================== */

    [data-testid="stNumberInput"] {
        color-scheme: light !important;
    }

    [data-testid="stNumberInput"] > div,
    [data-testid="stNumberInput"] > div > div,
    [data-testid="stNumberInput"] > div > div > div {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;

        border-radius: 12px !important;
    }

    [data-testid="stNumberInput"] input {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;

        color: #173B56 !important;

        -webkit-text-fill-color: #173B56 !important;

        border: none !important;

        box-shadow: none !important;
    }

    [data-testid="stNumberInput"] button,
    [data-testid="stNumberInput"] button > div {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;

        color: #173B56 !important;

        border: none !important;

        box-shadow: none !important;
    }

    [data-testid="stNumberInput"] button:hover,
    [data-testid="stNumberInput"] button:hover > div {
        background: #E7F5FF !important;
        background-color: #E7F5FF !important;
    }

    [data-testid="stNumberInput"] button svg {
        fill: #173B56 !important;
        color: #173B56 !important;
    }

    [data-testid="stNumberInput"] > div {
        border: 1px solid #D5E4EE !important;

        box-shadow:
            0 3px 10px rgba(40,90,120,0.06) !important;

        overflow: hidden !important;
    }


    /* ========================================================
       GENERAL INPUTS
       ======================================================== */

    input,
    textarea {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;

        color: #173B56 !important;

        -webkit-text-fill-color: #173B56 !important;
    }


    /* ========================================================
       PREDICT BUTTON
       ======================================================== */

    div.stButton > button {
        width: 100%;

        background:
            linear-gradient(
                135deg,
                #E7F5FF 0%,
                #B0DBFF 50%,
                #7FC4F5 100%
            ) !important;

        color: #123B57 !important;

        border: none !important;

        border-radius: 14px !important;

        padding: 0.75rem 1rem !important;

        font-size: 1rem !important;

        font-weight: 700 !important;

        box-shadow:
            0 8px 20px rgba(45,100,140,0.15) !important;

        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background:
            linear-gradient(
                135deg,
                #D3EDFF 0%,
                #A5D8FA 50%,
                #70BCEB 100%
            ) !important;

        color: #123B57 !important;

        transform: translateY(-1px);

        box-shadow:
            0 12px 25px rgba(45,100,140,0.22) !important;
    }

    div.stButton > button:focus {
        outline: none !important;

        box-shadow:
            0 8px 20px rgba(45,100,140,0.15) !important;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.45);

        border-radius: 16px;

        padding: 1rem;

        border: 1px solid rgba(255,255,255,0.65);
    }

    [data-testid="stMetricLabel"] {
        color: #55758A !important;
    }

    [data-testid="stMetricValue"] {
        color: #123B57 !important;
        font-weight: 800 !important;
    }


    /* ========================================================
       INFO BOX
       ======================================================== */

    [data-testid="stAlert"] {
        background: rgba(255,255,255,0.65) !important;

        border: 1px solid rgba(255,255,255,0.85) !important;

        border-radius: 16px !important;

        color: #31556B !important;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# ============================================================

with st.container():

    st.title("Dubai Property Cost Calculator")

    st.write(
        "Estimate Dubai property transaction values using "
        "machine learning and historical market patterns."
    )

st.write("")


# ============================================================
# PROPERTY DETAILS
# ============================================================

with st.container():

    st.subheader("Property details")

    st.caption(
        "Select the property characteristics below. "
        "Dropdowns reduce manual input errors."
    )

    st.write("")


    # ========================================================
    # FIRST ROW — 3 INPUTS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        areas = sorted(
            df["AREA_EN"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        default_area = (
            "Jumeirah Village Circle"
            if "Jumeirah Village Circle" in areas
            else areas[0]
        )

        area = st.selectbox(
            "Area",
            areas,
            index=areas.index(default_area)
        )


    with col2:

        projects = sorted(
            df["PROJECT_EN"]
            .dropna()
            .astype(str)
            .loc[
                lambda x: x.str.strip() != ""
            ]
            .unique()
            .tolist()
        )

        projects = ["Unknown"] + projects

        project = st.selectbox(
            "Project",
            projects
        )


    with col3:

        property_types = sorted(
            df["PROP_SB_TYPE_EN"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        property_type = st.selectbox(
            "Property type",
            property_types
        )


    # ========================================================
    # SECOND ROW — 3 INPUTS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        rooms = sorted(
            df["ROOMS_EN"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        rooms = ["Unknown"] + rooms

        selected_rooms = st.selectbox(
            "Rooms",
            rooms
        )


    with col2:

        actual_area = st.number_input(
            "Property area",
            min_value=10.0,
            max_value=5000.0,
            value=80.0,
            step=1.0,
            format="%.0f"
        )

        st.caption("Area in square metres (m²)")


    with col3:

        procedure_options = sorted(
            df["PROCEDURE_EN"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        default_procedure = (
            "Sale"
            if "Sale" in procedure_options
            else procedure_options[0]
        )

        procedure = st.selectbox(
            "Transaction type",
            procedure_options,
            index=procedure_options.index(
                default_procedure
            )
        )


    # ========================================================
    # THIRD ROW — 3 INPUTS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        freehold_options = sorted(
            df["IS_FREE_HOLD_EN"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        freehold = st.selectbox(
            "Ownership",
            freehold_options
        )


    with col2:

        metro_options = sorted(
            df["NEAREST_METRO_EN"]
            .dropna()
            .astype(str)
            .loc[
                lambda x: x.str.strip() != ""
            ]
            .unique()
            .tolist()
        )

        metro_options = ["Unknown"] + metro_options

        metro = st.selectbox(
            "Nearest metro",
            metro_options
        )


    with col3:

        mall_options = sorted(
            df["NEAREST_MALL_EN"]
            .dropna()
            .astype(str)
            .loc[
                lambda x: x.str.strip() != ""
            ]
            .unique()
            .tolist()
        )

        mall_options = ["Unknown"] + mall_options

        mall = st.selectbox(
            "Nearest mall",
            mall_options
        )


    # ========================================================
    # FOURTH ROW — 1 INPUT
    # ========================================================

    landmark_options = sorted(
        df["NEAREST_LANDMARK_EN"]
        .dropna()
        .astype(str)
        .loc[
            lambda x: x.str.strip() != ""
        ]
        .unique()
        .tolist()
    )

    landmark_options = ["Unknown"] + landmark_options

    landmark = st.selectbox(
        "Nearest landmark",
        landmark_options
    )


    st.write("")


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    predict = st.button(
        "Estimate property value  →"
    )


# ============================================================
# PREDICTION
# ============================================================

if predict:

    # --------------------------------------------------------
    # MARKET LOOKUPS
    # --------------------------------------------------------

    project_ppsqm = lookups["project_ppsqm"].get(
        project,
        np.nan
    )

    area_ppsqm = lookups["area_ppsqm"].get(
        area,
        np.nan
    )

    property_type_ppsqm = lookups[
        "property_type_ppsqm"
    ].get(
        property_type,
        np.nan
    )


    # --------------------------------------------------------
    # OVERALL MARKET FALLBACK
    # --------------------------------------------------------

    if "overall_market_ppsqm" in lookups:

        market_ppsqm = lookups[
            "overall_market_ppsqm"
        ]

    elif "market_ppsqm" in lookups:

        market_ppsqm = lookups[
            "market_ppsqm"
        ]

    elif "overall_ppsqm" in lookups:

        market_ppsqm = lookups[
            "overall_ppsqm"
        ]

    else:

        market_ppsqm = (
            df["TRANS_VALUE"] /
            df["ACTUAL_AREA"]
        ).median()


    # --------------------------------------------------------
    # HISTORICAL BEST PRICE
    # --------------------------------------------------------

    historical_values = [
        project_ppsqm,
        area_ppsqm,
        property_type_ppsqm,
        market_ppsqm
    ]

    historical_values = [
        value
        for value in historical_values
        if pd.notna(value)
    ]

    if len(historical_values) > 0:

        historical_best_ppsqm = max(
            historical_values
        )

    else:

        historical_best_ppsqm = market_ppsqm


    # --------------------------------------------------------
    # HISTORICAL SOURCE
    # --------------------------------------------------------

    if pd.notna(project_ppsqm):

        historical_source = "project"

    elif pd.notna(area_ppsqm):

        historical_source = "area"

    elif pd.notna(property_type_ppsqm):

        historical_source = "property_type"

    else:

        historical_source = "overall_market"


    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    project_count = lookups[
        "project_count"
    ].get(
        project,
        0
    )

    area_count = lookups[
        "area_count"
    ].get(
        area,
        0
    )

    property_type_count = lookups[
        "property_type_count"
    ].get(
        property_type,
        0
    )

    market_count = lookups.get(
        "market_count",
        len(df)
    )


    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    transaction_date = LATEST_DATE

    transaction_month = transaction_date.month

    transaction_day = transaction_date.day

    transaction_day_of_week = (
        transaction_date.dayofweek
    )

    transaction_day_of_year = (
        transaction_date.dayofyear
    )


    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        {
            "PROCEDURE_EN": [procedure],

            "IS_FREE_HOLD_EN": [freehold],

            "AREA_EN": [area],

            "PROP_SB_TYPE_EN": [property_type],

            "ROOMS_EN": [selected_rooms],

            "NEAREST_METRO_EN": [metro],

            "NEAREST_MALL_EN": [mall],

            "NEAREST_LANDMARK_EN": [landmark],

            "PROJECT_EN": [project],

            "historical_price_source": [
                historical_source
            ],

            "ACTUAL_AREA": [actual_area],

            "transaction_month": [
                transaction_month
            ],

            "transaction_day": [
                transaction_day
            ],

            "transaction_day_of_week": [
                transaction_day_of_week
            ],

            "transaction_day_of_year": [
                transaction_day_of_year
            ],

            "historical_project_ppsqm": [
                project_ppsqm
            ],

            "historical_area_ppsqm": [
                area_ppsqm
            ],

            "historical_property_type_ppsqm": [
                property_type_ppsqm
            ],

            "historical_market_ppsqm": [
                market_ppsqm
            ],

            "historical_best_ppsqm": [
                historical_best_ppsqm
            ],

            "historical_project_count": [
                project_count
            ],

            "historical_area_count": [
                area_count
            ],

            "historical_property_type_count": [
                property_type_count
            ],

            "historical_market_count": [
                market_count
            ]
        }
    )


    # --------------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------------

    processed_input = preprocessor.transform(
        input_data
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predicted_log_price = model.predict(
        processed_input
    )[0]

    prediction = np.expm1(
        predicted_log_price
    )

    prediction = max(
        0,
        prediction
    )


    # --------------------------------------------------------
    # PRICE PER SQUARE METRE
    # --------------------------------------------------------

    price_per_sqm = (
        prediction /
        actual_area
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.write("")

    with st.container():

        st.subheader(
            "Estimated transaction value"
        )

        st.metric(
            label="Predicted property value",
            value=f"AED {prediction:,.0f}"
        )

        st.write("")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Estimated AED / m²",
                f"AED {price_per_sqm:,.0f}"
            )

        with col2:

            st.metric(
                "Property size",
                f"{actual_area:,.0f} m²"
            )

        with col3:

            st.metric(
                "Model",
                "XGBoost"
            )


# ============================================================
# TRANSPARENCY NOTE
# ============================================================

st.write("")

st.info(
    "This estimate is generated from historical Dubai "
    "property transaction patterns. It should be treated "
    "as a model-based estimate rather than an official "
    "property valuation."
)


# ============================================================
# FOOTER
# ============================================================

st.write("")

st.markdown(
    """
    <div style="text-align: center; margin-top: 2rem; padding-bottom: 1rem;">
        <p style="
            color: #55758A;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        ">
            Made By Mohammed Wafiq
        </p>
        <p style="margin-top: 0;">
            <a
                href="https://www.linkedin.com/in/mohammed-wafiq"
                target="_blank"
                style="
                    color: #126FB0;
                    font-size: 0.85rem;
                    text-decoration: none;
                    font-weight: 600;
                "
            >
                www.linkedin.com/in/mohammed-wafiq
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
