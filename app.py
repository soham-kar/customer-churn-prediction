import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# --- Load model and preprocessing objects ---
@st.cache_resource
def load_model_and_assets():
    model = tf.keras.models.load_model('models/my_model.keras')

    with open('models/label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('models/onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_assets()


# --- App Layout ---
st.title('Customer Churn Prediction')
st.markdown("""
Predict whether a bank customer is likely to churn based on their demographics and banking activity.
""")

# --- Collect user input ---
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, value=30)
balance = st.number_input('Balance', min_value=0.0, step=100.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600, step=1)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=1000.0)
tenure = st.slider('Tenure (Years)', 0, 10, value=3)
num_of_products = st.slider('Number of Products', 1, 4, value=1)

has_cr_card_input = st.selectbox('Has Credit Card', ['Yes', 'No'])
has_cr_card = 1 if has_cr_card_input == 'Yes' else 0

is_active_member_input = st.selectbox('Is Active Member', ['Yes', 'No'])
is_active_member = 1 if is_active_member_input == 'Yes' else 0

# --- Preprocessing ---
def preprocess_input():
    try:
        gender_encoded = label_encoder_gender.transform([gender])[0]
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
        )

        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [gender_encoded],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        input_combined = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        input_scaled = scaler.transform(input_combined)

        return input_scaled
    except Exception as e:
        st.error(f"❌ Error during preprocessing: {e}")
        return None


# --- Predict on Button Click ---
if st.button('Predict Churn'):
    with st.spinner('Predicting...'):
        processed_input = preprocess_input()

        if processed_input is not None:
            prediction = model.predict(processed_input)
            prediction_proba = prediction[0][0]

            st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}")

            if prediction_proba > 0.5:
                st.error("⚠️ The customer is likely to churn.")
            else:
                st.success("✅ The customer is not likely to churn.")

# --- Optional: Model Info ---
with st.expander("ℹ️ Model & Data Info"):
    st.markdown("""
    - **Model**: Trained Keras Neural Network  
    - **Input Features**: Credit Score, Age, Balance, Tenure, etc.  
    - **Encoders**: LabelEncoder for Gender, OneHotEncoder for Geography  
    - **Scaler**: StandardScaler  
    - **Threshold**: > 50% probability means likely to churn.
    """)
