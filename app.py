import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
# Using st.cache_resource to load them only once
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# Page configuration
st.set_page_config(
    page_title="SUV Purchase Prediction",
    page_icon="🚗",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
    }
    .failure {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🚗 SUV Purchase Prediction")
st.write("Enter the customer's details below to predict if they will purchase an SUV.")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    
    with col2:
        salary = st.number_input("Estimated Salary ($)", min_value=10000, max_value=200000, value=50000, step=500)
    
    submitted = st.form_submit_button("Predict Purchase")

# Prediction logic
if submitted:
    # Prepare input data
    input_data = np.array([[age, salary]])
    
    # Scale the input using the loaded scaler
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    st.markdown("---")
    
    if prediction[0] == 1:
        st.markdown(f"""
            <div class="prediction-box success">
                <h3>Likely to Purchase! ✅</h3>
                <p>Confidence: {probability:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-box failure">
                <h3>Unlikely to Purchase ❌</h3>
                <p>Probability of purchase: {probability:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Machine Learning Model: Logistic Regression | Built with Streamlit")
