# Customer Churn Prediction System

## 1. Project Overview

The Customer Churn Prediction System is a Machine Learning application that predicts whether a bank customer is likely to leave the bank.

The system uses historical customer data to identify churn risk and displays prediction results through an interactive Streamlit web application.

## 2. Objectives

- Predict customer churn using Machine Learning.
- Analyze customer information.
- Display churn probability and risk levels.
- Compare multiple Machine Learning algorithms.
- Evaluate model performance using standard metrics.
- Visualize customer data.
- Allow users to upload and validate CSV datasets.
- Download prediction results.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Matplotlib
- Unittest

## 4. Dataset

**Dataset:** Churn_Modelling.csv

**Total Records:** 10,000

**Target Column:** Exited

- 0: Customer stayed
- 1: Customer left

**Dataset Source:**

https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers

## 5. Data Preprocessing

The following preprocessing steps are performed:

- Removed unnecessary columns.
- Removed duplicate records.
- Created additional features.
- Converted categorical features into numerical features.
- Separated input features and target variable.
- Aligned prediction input columns with the trained model.

## 6. Machine Learning Algorithms

This project compares four Machine Learning algorithms:

1. Random Forest Classifier
2. Extra Trees Classifier
3. Gradient Boosting Classifier
4. XGBoost Classifier

XGBoost was selected as the best-performing model based on the measured test accuracy.

## 7. Model Evaluation

The selected model achieved the following results:

| Metric | Score |
|---|---:|
| Accuracy | 87.15% |
| Precision | 80.24% |
| Recall | 48.89% |
| F1 Score | 60.76% |

These are the measured results from the current test run.

The model's recall indicates that some customers who actually leave may not be identified correctly.

## 8. Application Features

### Dataset Management
- View customer dataset records.
- Display dataset statistics.
- Upload customer data in CSV format.
- Validate required columns and target values.
- Detect missing values and invalid datasets.

### Data Visualization
- Customer churn distribution chart.
- Customer distribution by geography.
- Customer age distribution chart.

### Churn Prediction
- Predict whether a customer may leave.
- Display churn probability.
- Classify churn risk as Low, Medium, or High.

Risk categories:
- Low Risk: Probability below 30%.
- Medium Risk: Probability from 30% to below 60%.
- High Risk: Probability of 60% or above.

These are application-defined probability thresholds, not guaranteed measures of actual customer behaviour.

### Download Options
- Download the customer dataset.
- Download an individual customer's prediction result.

## 9. Project Structure

- `app.py` - Streamlit web application.
- `preprocessing.py` - Data loading and preprocessing.
- `train_model.py` - Model training and comparison.
- `results.py` - Model evaluation results.
- `test_project.py` - Automated tests.
- `Churn_Modelling.csv` - Customer dataset.
- `churn_model.pkl` - Trained Machine Learning model.
- `model_columns.pkl` - Model input column definitions.
- `model_evaluation_results.csv` - Model evaluation results.
- `confusion_matrix.png` - Confusion matrix visualization.
- `requirements.txt` - Python dependencies.

## 10. Installation and Execution

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
