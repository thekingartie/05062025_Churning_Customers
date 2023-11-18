import streamlit as st
import pickle 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# Load the pre-trained machine learning model and scaler 
# with open('model.h5') as file:
#     loaded_model = pickle.load(file)

loaded_model = load_model('model.h5')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Create a Streamlit app
st.title("Churn Prediction Application")
st.write("Please enter customer attributes to predict churn")

# Define input fields for user input
tenure = st.number_input("tenure", min_value = 0, max_value = 100)
MonthlyCharges = st.number_input("MonthlyCharges", min_value = 0, max_value = 1000)
customerID = st.number_input("customerID", min_value = 0, max_value = 9999)
Contract = st.selectbox("Contract", ['One year', 'Two year', 'Monthly-to-month'])
OnlineSecurity = st.selectbox("OnlineSecurity", ['No', 'Yes'])
PaymentMethod = st.selectbox("PaymentMethod", ['Electronic Check', 'Mailed Check', 'Bank Transfer (automatic)', 'Credit Card (automatic)'])
TechSupport = st.selectbox("TechSupport", ['No', 'Yes'])
InternetService = st.selectbox("InternetService", ["DSL", "Fiber Optic"])
OnlineBackup = st.selectbox("OnlineBackup", ['No', 'Yes'])
gender = st.selectbox("gender", ['M', 'F'])
PaperlessBilling = st.selectbox("PaperlessBilling", ['No', 'Yes'])
MultipleLines = st.selectbox("MultipleLines", ['No', 'Yes'])
Partner = st.selectbox("Partner", ['No', 'Yes'])
DeviceProtection = st.selectbox("DeviceProtection", ['No', 'Yes'])
SeniorCitizen = st.selectbox("SeniorCitizen", ['No', 'Yes'])
Dependents = st.number_input("Dependents", min_value = 0, max_value = 20)
StreamingTV = st.selectbox("StreamingTV", ['No', 'Yes'])
StreamingMovies = st.selectbox("StreamingMovies", ['No', 'Yes'])
PhoneService = st.selectbox("PhoneService", ['No', 'Yes'])
TotalCharges = st.number_input("TotalCharges", min_value = 0, max_value = 1000)


# Create a button to trigger the prediction
if st.button("Predict Churn (given the input)"):
    # Prepare the input data for prediction
    input_data = {
        "tenure": [tenure],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges],
        "customerID": [1 if customerID == '123456' else 0],
        "Contract_Two Year": [1 if Contract == 'Two year' else 0],
        "OnlineSecurity_Yes": [1 if OnlineSecurity == 'Yes' else 0],
        "PaymentMethod_CreditCard (automatic)": [1 if PaymentMethod == 'Credit Card (automatic)' else 0],
        "TechSupport_Yes": [1 if TechSupport == 'Yes' else 0],
        "InternetService_Fiber optic": [1 if InternetService == 'Fiber Optic' else 0],
        "OnlineBackup_Yes": [1 if OnlineBackup == 'Yes' else 0],
        "gender_M": [1 if gender == 'M' else 0],
        "PaperlessBilling _Yes": [1 if PaperlessBilling == 'Yes' else 0],
        "MultipleLines_Yes": [1 if MultipleLines == 'Yes' else 0],
        "Partner_Yes": [1 if Partner == 'Yes' else 0],
        "DeviceProtection_Yes": [1 if DeviceProtection == 'Yes' else 0],
        "SeniorCitizen_Yes": [1 if SeniorCitizen == 'Yes' else 0],
        "Dependents": [Dependents],     
        "StreamingTV_Yes": [1 if StreamingTV == 'Yes' else 0],
        "StreamingMovies_Yes": [1 if StreamingMovies == 'Yes' else 0],
        "PhoneService_Yes": [1 if PhoneService == 'Yes' else 0],
    }

    input_dataset = pd.DataFrame(input_data)

    # Ensure feature names match the ones used during training
    input_dataset.columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'customerID', 'Contract', 'OnlineSecurity', 'PaymentMethod', 'TechSupport', 'InternetService', 'OnlineBackup', 'gender', 'PaperlessBilling', 'MultipleLines', 'Partner', 'DeviceProtection', 'SeniorCitizen', 'Dependents', 'StreamingTV', 'StreamingMovies', 'PhoneService']
    
    scaled_input_data = scaler.transform(input_dataset)

    # Make the prediction
    predicted_churn = loaded_model.predict(scaled_input_data)

    # Calculate the confidence factor
    confidence_factor = predicted_churn.squeeze()

    # Display the prediction and confidence factor 
    st.write(f"Predicted Churn: {int(round(float(confidence_factor)))}")
    st.write(f"confidence Factor: {confidence_factor:.2f}")

    # Display the prediction
    if confidence_factor > 0.5:
        st.warning("Churn: Yes")
    else:
        st.success("Churn: No")
    
# Create a reset button to clear the input fields
if st.button("Reset"):
    st.experimental_rerun()