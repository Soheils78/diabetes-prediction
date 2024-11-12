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

# Standardize the features, including Age
scaler = StandardScaler()
X = scaler.fit_transform(data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']])
y = data['Outcome']

# Train the Logistic Regression model with an 80-20 split
model = LogisticRegression(random_state=42)
model.fit(X, y)

# Streamlit app
st.title("Diabetes Prediction App")
st.write("This app predicts the likelihood of diabetes based on user inputs.")
st.write("Implemented by Soheil Salemi")

# User input fields
glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=120.0)
blood_pressure = st.number_input("Blood Pressure (Diastolic blood pressure in mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
skin_thickness = st.number_input("Skin Thickness (Triceps skin fold thickness in mm)", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Insulin Level (2-Hour serum insulin in mu U/ml)", min_value=0.0, max_value=900.0, value=80.0)
bmi = st.number_input("BMI (Body mass index in kg/m²)", min_value=0.0, max_value=100.0, value=25.0)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    try:
        # Preprocess the user input
        user_data = np.array([[glucose, blood_pressure, skin_thickness, insulin, bmi, age]])
        user_data_scaled = scaler.transform(user_data)

        # Make the prediction
        prediction = model.predict(user_data_scaled)
        probability = model.predict_proba(user_data_scaled)[0][1]

        # Display the prediction with different background colors
        if len(prediction) > 0 and prediction[0] == 1:
            st.markdown(
                f"<div style='padding: 15px; color: white; background-color: red; text-align: center; border-radius: 10px;'>"
                f"The model predicts that the person is <strong>likely</strong> to have diabetes with a probability of {probability:.2f}."
                f"</div>",
                unsafe_allow_html=True,
            )
        elif len(prediction) > 0 and prediction[0] == 0:
            st.markdown(
                f"<div style='padding: 15px; color: white; background-color: green; text-align: center; border-radius: 10px;'>"
                f"The model predicts that the person is <strong>unlikely</strong> to have diabetes with a probability of {1 - probability:.2f}."
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.error("Unable to make a prediction. Please check the input values.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
