import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f7f8fa;
}

.block-container {
    max-width: 1150px;
    padding-top: 35px;
    padding-bottom: 40px;
}


/* Hide Streamlit branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* Input labels */

label {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #344054 !important;
}


/* Number input */

div[data-baseweb="input"] {
    border-radius: 10px !important;
    border: 1px solid #d9dee8 !important;
    background: #ffffff !important;
}


/* Select box */

div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border: 1px solid #d9dee8 !important;
    background: #ffffff !important;
}


/* Input focus */

div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #34558c !important;
    box-shadow: 0 0 0 2px rgba(52, 85, 140, 0.10) !important;
}


/* Prediction button */

.stButton > button {
    width: 100%;
    height: 52px;

    border-radius: 11px;
    border: none;

    background: #243b64;
    color: white;

    font-family: 'Inter', sans-serif;
    font-size: 16px;
    font-weight: 700;

    transition: 0.2s;
}

.stButton > button:hover {
    background: #182d50;
    color: white;
    transform: translateY(-1px);
}


/* Section heading */

.section-heading {
    font-size: 25px;
    font-weight: 750;
    color: #172033;
    margin-top: 32px;
    margin-bottom: 4px;
}

.section-subtitle {
    font-size: 14px;
    color: #7a8495;
    margin-bottom: 18px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

model_data = joblib.load(
    "model/house_price_model.pkl"
)

model = model_data["model"]

features = model_data["features"]


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "data/enhanced_house_price_dataset.csv"
)


# ============================================================
# HERO SECTION
# ============================================================

st.html("""
<div style="
    background: linear-gradient(135deg, #172033, #263d63);
    border-radius: 22px;
    padding: 48px 50px;
    margin-bottom: 30px;
    box-shadow: 0 12px 30px rgba(23,32,51,0.12);
">

    <div style="
        color: white;
        font-family: 'Inter', sans-serif;
        font-size: 46px;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -1.5px;
    ">
        House Price Predictor
    </div>

    <div style="
        color: #dce5f2;
        font-family: 'Inter', sans-serif;
        font-size: 19px;
        font-weight: 500;
        margin-top: 14px;
    ">
        Powered by Linear Regression
    </div>

</div>
""")


# ============================================================
# PROPERTY DETAILS
# ============================================================

st.markdown(
    '<div class="section-heading">🏠 Property Details</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Tell us about the property'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT CONTAINER
# ============================================================

with st.container(border=True):

    # --------------------------------------------------------
    # ROW 1
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        area = st.number_input(
            "📐 Area (sq ft)",
            min_value=0,
            value=3000,
            step=100
        )

    with col2:

        bedrooms = st.number_input(
            "🛏️ Bedrooms",
            min_value=1,
            value=3,
            step=1
        )


    # --------------------------------------------------------
    # ROW 2
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        bathrooms = st.number_input(
            "🛁 Bathrooms",
            min_value=1,
            value=2,
            step=1
        )

    with col2:

        stories = st.number_input(
            "🏢 Stories",
            min_value=1,
            value=2,
            step=1
        )


    # --------------------------------------------------------
    # ROW 3
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        parking = st.number_input(
            "🚗 Parking",
            min_value=0,
            value=1,
            step=1
        )

    with col2:

        city = st.selectbox(
            "📍 City",
            sorted(
                df["City"]
                .dropna()
                .unique()
            )
        )


    # --------------------------------------------------------
    # ROW 4
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        main_road = st.selectbox(
            "🛣️ Main Road",
            sorted(
                df["Main Road"]
                .dropna()
                .unique()
            )
        )

    with col2:

        guest_room = st.selectbox(
            "🛋️ Guest Room",
            sorted(
                df["Guest Room"]
                .dropna()
                .unique()
            )
        )


    # --------------------------------------------------------
    # ROW 5
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        water_supply = st.selectbox(
            "💧 Water Supply",
            sorted(
                df["Water Supply"]
                .dropna()
                .unique()
            )
        )

    with col2:

        preferred_tenant = st.selectbox(
            "👤 Preferred Tenant",
            sorted(
                df["Preferred Tenant"]
                .dropna()
                .unique()
            )
        )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.write("")

button_left, button_center, button_right = st.columns(
    [1, 2, 1]
)

with button_center:

    predict_button = st.button(
        "✨ Predict House Price"
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # Existing model features that are not visible
    # to the user are kept at their existing defaults.

    age = 20
    furnishing = "Furnished"
    basement = "No"
    air_conditioning = "No"
    locality_rating = 5


    # ========================================================
    # CREATE INPUT DATA
    # ========================================================

    input_data = pd.DataFrame([{

        "Area": area,

        "Bedrooms": bedrooms,

        "Bathrooms": bathrooms,

        "Stories": stories,

        "Parking": parking,

        "Age": age,

        "City": city,

        "Furnishing": furnishing,

        "Main Road": main_road,

        "Guest Room": guest_room,

        "Basement": basement,

        "Water Supply": water_supply,

        "Air Conditioning": air_conditioning,

        "Preferred Tenant": preferred_tenant,

        "Locality Rating": locality_rating

    }])


    # ========================================================
    # FEATURE ENCODING
    # ========================================================

    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )


    # ========================================================
    # MATCH TRAINING FEATURES
    # ========================================================

    input_data = input_data.reindex(
        columns=features,
        fill_value=0
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(
        input_data
    )[0]


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.html(f"""
    <div style="
        background: #172033;
        border-radius: 22px;
        padding: 38px;
        text-align: center;
        margin-top: 28px;
        box-shadow: 0 12px 32px rgba(23,32,51,0.15);
    ">

        <div style="
            color: #aeb9cb;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
        ">
            🏡 Estimated Property Value
        </div>

        <div style="
            color: white;
            font-size: 46px;
            font-weight: 800;
            margin-top: 8px;
            letter-spacing: -1px;
        ">
            ₹{prediction:,.2f}
        </div>

        <div style="
            color: #aeb9cb;
            font-size: 13px;
            margin-top: 7px;
        ">
            Based on the property details provided.
        </div>

    </div>
    """)


    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-heading">Prediction Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Details about your prediction'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "MODEL",
            "Linear Regression"
        )


    with col2:

        st.metric(
            "PROPERTY",
            f"{bedrooms} Bed • {bathrooms} Bath"
        )


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div style="
    text-align: center;
    color: #98a1b2;
    font-size: 12px;
    margin-top: 35px;
    padding-top: 18px;
    border-top: 1px solid #e5e8ee;
">
    Built with Python • Streamlit • Scikit-learn
</div>
""")