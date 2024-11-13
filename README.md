# Diabetes Prediction Project

This project is a web-based application that predicts the likelihood of diabetes using a Logistic Regression model. The app takes user input for various health metrics and provides a prediction based on a trained machine learning model.

## Project Structure

- `diabetes_app.py`: Python script for the Streamlit web application.
- `diabetes.csv`: Dataset used for training the Logistic Regression model.
- `requirements.txt`: List of dependencies needed to run the app.
- `predict_diabets.ipynb`: Jupyter Notebook containing data preprocessing, model training, and evaluation steps.

## Dataset

The dataset used is the **Pima Indians Diabetes Dataset**, which contains the following features:

- `Glucose`: Plasma glucose concentration.
- `BloodPressure`: Diastolic blood pressure (mm Hg).
- `SkinThickness`: Triceps skin fold thickness (mm).
- `Insulin`: 2-Hour serum insulin (mu U/ml).
- `BMI`: Body mass index (weight in kg/(height in m)^2).
- `Age`: Age of the patient (years).
- `Outcome`: Indicates if the patient has diabetes (1) or not (0).

## App Description

The app uses a Logistic Regression model trained on the Pima Indians Diabetes dataset. It allows users to input values for the following features:

- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- BMI
- Age

After the user inputs these values and clicks the **Predict** button, the app will display:

- **Prediction Result**: Whether the person is likely or unlikely to have diabetes.
- **Probability**: The model’s confidence in its prediction.

## How to Run the App from Browser

Just  need to click to https://diabetes-prediction-8iwbxyaae6rueawreskvno.streamlit.app

## Example of Usage

1. Enter values for Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, and Age.
2. Click the **Predict** button.
3. The app will display whether the person is likely or unlikely to have diabetes, along with the prediction probability.


## Technologies Used
	•	Python
	•	Streamlit
	•	Scikit-Learn
	•	Pandas
	•	NumPy

 ## Prject Analysis and Experiments
 In this project, several machine learning experiments were conducted to find the best model
 for predicting diabetes using the Pima Indians Diabetes dataset. Below is a summary of the analysis
 and experimentation which is available in `predict_diabets.ipynb` file :
 
 #### 1. Data Preprocessing
 The dataset contained some missing or zero values in key features such as **Glucose, Blood Pressure, Skin Thickness, Insulin**, and **BMI**. 
 To handle these issues:

 	•	Zero values in these features were replaced with NaN (Not a Number).
	•	Missing values were filled with the mean of each column.
	•	Features were standardized using StandardScaler to ensure the data has a mean of 0 and a standard deviation of 1.

### 2. Logistic Regression Model

A Logistic Regression model was trained on the dataset to predict the likelihood of diabetes:
	•	Initially, the data was split into 70% training and 30% testing.
	•	The model achieved an accuracy of 75.32% on the test set.


 Key metrics:
	•	Precision: 0.66
	•	Recall: 0.59
	•	F1-Score: 0.62

 ### 3. Experimenting with Random Forest

 A Random Forest Classifier was also trained and evaluated for comparison:
	•	The model achieved an accuracy of 71% on the test set.

 Key metrics:
	•	Precision: 0.58 for the positive class (diabetes)
	•	Recall: 0.59 for the positive class
	•	F1-Score: 0.58

 **Conclusion:** The Logistic Regression model outperformed the Random Forest model
 in terms of both accuracy and recall.

 ### 4. Data Split Experiment

 To improve model performance, different data splits were tested:
	•	When the data was split into **80% training ** and **20% testing**, the Logistic Regression model’s accuracy increased to **77.27%.**

 ### 5. Cross-Validation
 To ensure the model’s robustness, 5-fold cross-validation was performed:
	•	Cross-Validation Scores for each fold: [0.779, 0.740, 0.779, 0.797, 0.752]
	•	Mean Cross-Validation Accuracy: 76.95%

 ### Summary of Findings

 	•	The best performing model was the Logistic Regression with an 80-20 split.
	•	Cross-validation confirmed the model’s reliability with a mean accuracy of 76.95%.
	•	The model was implemented in a Streamlit app for easy user interaction and diabetes prediction based on user inputs.


 ## Author

 This project was implemented by Soheil Salemi.
 
