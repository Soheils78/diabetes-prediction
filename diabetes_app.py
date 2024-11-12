import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load your dataset and preprocess it
data = pd.read_csv('diabetes.csv')

# Replace zeros with NaN for specific columns
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)

# Fill NaN values with the mean of each column
data.fillna(data.mean(), inplace=True)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])
y = data['Outcome']

# Train the Logistic Regression model with an 80-20 split
model = LogisticRegression(random_state=42)
model.fit(X, y)

# Streamlit app
st.title("Diabetes Prediction Apps")
st.write("This app predicts the likelihood of diabetes based on user inputs.")
st.write("Implemented by Soheil Salemi")


# User input fields
glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=120.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Insulin Level", min_value=0.0, max_value=900.0, value=80.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)

# Predict button
if st.button("Predict"):
    # Preprocess the user input
    user_data = np.array([[glucose, blood_pressure, skin_thickness, insulin, bmi]])
    user_data_scaled = scaler.transform(user_data)

    # Make the prediction
    prediction = model.predict(user_data_scaled)
    probability = model.predict_proba(user_data_scaled)[0][1]

    # Display the prediction
    if prediction[0] == 1:
        st.success(f"The model predicts that the person is likely to have diabetes with a probability of {probability:.2f}.")
    else:
        st.success(f"The model predicts that the person is unlikely to have diabetes with a probability of {1 - probability:.2f}.")
