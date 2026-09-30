
import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD MODEL AND SCALER
# ---------------------------------------------------------

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666666;
    margin-bottom: 30px;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    background-color: #f7f9fc;
    border: 1px solid #e5e7eb;
    text-align: center;
    margin-top: 20px;
}

.result-title {
    font-size: 28px;
    font-weight: 700;
}

.info-card {
    padding: 18px;
    border-radius: 12px;
    background-color: #f7f9fc;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🩺 Diabetes Risk Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Logistic Regression based diabetes prediction system'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("👤 Patient Information")

    st.write(
        "Enter the patient's medical information below."
    )

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        value=1,
        step=1
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )

    st.divider()

    st.caption(
        "This application is a machine-learning demonstration "
        "and should not be used as a medical diagnosis."
    )


# ---------------------------------------------------------
# MAIN INPUT SECTION
# ---------------------------------------------------------

st.subheader("📋 Medical Measurements")

col1, col2 = st.columns(2)

with col1:

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        value=120.0,
        help="Plasma glucose concentration"
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        value=70.0,
        help="Diastolic blood pressure"
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        value=20.0,
        help="Triceps skin fold thickness"
    )

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        value=80.0,
        help="Serum insulin level"
    )


with col2:

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        value=25.0,
        help="Body Mass Index"
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=0.5,
        help="Diabetes hereditary risk score"
    )

    st.info(
        "💡 Enter the patient's values and click "
        "**Assess Diabetes Risk**."
    )


st.divider()


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

predict_button = st.button(
    "🔍 Assess Diabetes Risk",
    use_container_width=True
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if predict_button:

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Scale input using the same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get probability of diabetes
    probability = model.predict_proba(input_scaled)[0][1]

    probability_percentage = probability * 100


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.subheader("📊 Assessment Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:

            st.error("### ⚠️ Diabetes Predicted")

            st.write(
                "The model predicts the positive diabetes class."
            )

        else:

            st.success("### ✅ No Diabetes Predicted")

            st.write(
                "The model predicts the negative diabetes class."
            )


    with result_col2:

        st.metric(
            "Predicted Diabetes Probability",
            f"{probability_percentage:.2f}%"
        )

        st.progress(
            probability
        )


    # -----------------------------------------------------
    # RISK LEVEL
    # -----------------------------------------------------

    st.markdown("### Risk Indicator")

    if probability < 0.30:

        st.success(
            "🟢 Lower predicted probability"
        )

    elif probability < 0.60:

        st.warning(
            "🟡 Moderate predicted probability"
        )

    else:

        st.error(
            "🔴 Higher predicted probability"
        )


    # -----------------------------------------------------
    # INPUT SUMMARY
    # -----------------------------------------------------

    with st.expander("📋 View Patient Input Summary"):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ---------------------------------------------------------
# ABOUT THE MODEL
# ---------------------------------------------------------

st.divider()

with st.expander("ℹ️ About This Model"):

    st.write(
        """
        This application uses a Logistic Regression classification
        model trained on diabetes-related medical features.

        The model uses the following inputs:

        • Pregnancies
        • Glucose
        • Blood Pressure
        • Skin Thickness
        • Insulin
        • BMI
        • Diabetes Pedigree Function
        • Age

        The input values are standardized using the same scaler
        used during model training before making predictions.
        """
    )

