import streamlit as st
import pandas as pd
import joblib
from preprocessing import load_data, preprocess_data

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction System")
st.write("Predict whether a customer is likely to leave the bank.")

try:
    # Load trained model
    model = joblib.load("churn_model.pkl")
    model_columns = joblib.load("model_columns.pkl")

    # CSV upload
    st.subheader("Upload Customer Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.error("The uploaded CSV file is empty.")
                st.stop()

            if len(df.columns) != len(set(df.columns)):
                st.error("Duplicate column names found in the CSV.")
                st.stop()

            st.success("CSV file uploaded successfully!")

        except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
            st.error("Invalid CSV file. Please upload a valid CSV.")
            st.stop()

    else:
        df = load_data()
        st.info("Using the default customer dataset.")

    # Dataset validation
    required_columns = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            "Required columns missing: "
            + ", ".join(missing_columns)
        )
        st.stop()

    if df[required_columns].isnull().any().any():
        st.error(
            "Missing values found. Please clean the dataset "
            "before uploading."
        )
        st.stop()

    if not df["Exited"].isin([0, 1]).all():
        st.error("Exited column must contain only 0 or 1.")
        st.stop()

    # Dataset overview
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Customer Dataset")
    st.dataframe(df.head(10), width="stretch")
        # Data Visualization
    st.subheader("Customer Data Visualizations")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("### Customer Churn Distribution")
        churn_counts = df["Exited"].value_counts().sort_index()
        churn_counts.index = [
            "Stayed" if value == 0 else "Churned"
            for value in churn_counts.index
        ]
        st.bar_chart(churn_counts)

    with chart_col2:
        st.write("### Customers by Geography")
        geography_counts = df["Geography"].value_counts()
        st.bar_chart(geography_counts)

    st.write("### Age Distribution")
    age_counts = df["Age"].value_counts().sort_index()
    st.line_chart(age_counts)

    # Preprocessing
    X, y = preprocess_data(df)
    X = X.reindex(columns=model_columns, fill_value=0)

    # Customer prediction
    st.subheader("Customer Churn Prediction")

    customer_index = st.number_input(
        "Enter Customer Row Number",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    if st.button("Predict Churn"):
        customer = X.iloc[[int(customer_index)]]

        prediction = model.predict(customer)[0]
        probabilities = model.predict_proba(customer)[0]

        churn_class_index = list(model.classes_).index(1)
        probability = probabilities[churn_class_index]

        if prediction == 1:
            st.warning("Prediction: Customer may leave the bank.")
        else:
            st.success("Prediction: Customer is likely to stay.")

        st.metric("Churn Probability", f"{probability:.2%}")

        if probability < 0.30:
            st.success("Churn Risk Level: Low Risk")
        elif probability < 0.60:
            st.warning("Churn Risk Level: Medium Risk")
        else:
            st.error("Churn Risk Level: High Risk")

        # Download individual prediction
        result = df.iloc[[int(customer_index)]].copy()
        result["Predicted_Churn"] = int(prediction)
        result["Churn_Probability"] = round(float(probability), 4)

        st.download_button(
            label="Download Prediction Result",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="customer_churn_prediction.csv",
            mime="text/csv"
        )

    # Download dataset
    st.subheader("Download Customer Dataset")

    st.download_button(
        label="Download Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="customer_churn_results.csv",
        mime="text/csv"
    )

except FileNotFoundError:
    st.error(
        "Required project files are missing. "
        "Check churn_model.pkl, model_columns.pkl and the dataset."
    )

except Exception as e:
    st.error(f"An error occurred: {e}")