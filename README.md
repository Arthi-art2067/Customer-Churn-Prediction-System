# Customer Churn Prediction System

## 1. Project Overview
The Customer Churn Prediction System uses Machine Learning
to predict whether a bank customer is likely to leave the bank.

## 2. Objectives
- Predict customer churn.
- Analyze customer information.
- Display prediction probability.
- Evaluate model performance.
- Provide an interactive web application.

## 3. Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## 4. Dataset
Dataset: Churn_Modelling.csv

Total Records: 10,000

Target Column: Exited

- 0: Customer stayed
- 1: Customer left

Dataset Source:
https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers

## 5. Data Preprocessing
- Removed unnecessary columns.
- Removed duplicate records.
- Converted categorical features into numerical features.
- Separated input features and target variable.

## 6. Machine Learning Algorithm
Random Forest Classifier

## 7. Model Evaluation
Accuracy: 87.15%
Precision: 80.24%
Recall: 48.89%
F1 Score: 60.76%

These are the measured results from the current test run.

## 8. Application Features
- View customer dataset.
- View dataset statistics.
- Predict customer churn.
- Display churn probability.
- Download customer dataset.

## 9. How to Run

Install dependencies:

pip install -r requirements.txt

Train the model:

python train_model.py

Run the application:

streamlit run app.py

## 10. Conclusion
This project demonstrates how Machine Learning
can help predict customer churn using historical
customer data.