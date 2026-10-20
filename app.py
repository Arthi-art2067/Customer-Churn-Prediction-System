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
    df = load_data()

    # Dataset overview
    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)
    col1.metric("Total Customers", len(df))
    col2.metric("Total Columns", len(df.columns))

    st.subheader("Customer Dataset")
    st.dataframe(df.head(10), use_container_width=True)

    # Load trained model
    model = joblib.load("churn_model.pkl")
    model_columns = joblib.load("model_columns.pkl")

    # Preprocess dataset
    X, y = preprocess_data(df)
    X = X.reindex(columns=model_columns, fill_value=0)

    # Prediction
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
        probability = model.predict_proba(customer)[0][1]

        if prediction == 1:
            st.warning("Prediction: Customer may leave the bank.")
        else:
            st.success("Prediction: Customer is likely to stay.")

        st.metric("Churn Probability", f"{probability:.2%}")

    # Download dataset
    st.subheader("Download Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Customer Dataset",
        data=csv,
        file_name="customer_churn_results.csv",
        mime="text/csv"
    )

except FileNotFoundError:
    st.error(
        "Required file not found. "
        "Check that Churn_Modelling.csv, churn_model.pkl, "
        "and model_columns.pkl are in the project folder."
    )

except Exception as e:
    st.error(f"An error occurred: {e}")