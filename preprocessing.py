import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = [
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

NUMERIC_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited"
]


def load_data(file_path="Churn_Modelling.csv"):
    """Load the customer churn dataset safely."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df


def validate_data(df):
    """Validate uploaded customer dataset."""

    errors = []

    # Check empty dataset
    if df is None or df.empty:
        return ["The uploaded CSV file is empty."]

    # Check duplicate column names
    if not df.columns.is_unique:
        errors.append("Duplicate column names found.")

    # Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Stop additional checks if required columns are missing
    if missing_columns:
        return errors

    # Check missing values in required columns
    if df[REQUIRED_COLUMNS].isnull().any().any():
        errors.append(
            "Missing values found in required columns."
        )

    # Check numeric columns
    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[column], errors="coerce")

        if converted.isnull().any():
            errors.append(
                f"{column} contains invalid or missing numeric values."
            )

    # Validate target values
    exited = pd.to_numeric(df["Exited"], errors="coerce")

    if exited.isnull().any() or not exited.isin([0, 1]).all():
        errors.append(
            "Exited column must contain only 0 or 1."
        )

    # Validate categorical values
    if not df["Gender"].isin(["Male", "Female"]).all():
        errors.append(
            "Gender must contain Male or Female."
        )

    if not df["Geography"].isin(
        ["France", "Germany", "Spain"]
    ).all():
        errors.append(
            "Geography must contain France, Germany or Spain."
        )

    # Check numeric values are within reasonable ranges
    numeric_df = df[NUMERIC_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    if (numeric_df["Age"].dropna() <= 0).any():
        errors.append("Age must be greater than zero.")

    if (numeric_df["Tenure"].dropna() < 0).any():
        errors.append("Tenure cannot be negative.")

    if (numeric_df["Balance"].dropna() < 0).any():
        errors.append("Balance cannot be negative.")

    if (numeric_df["EstimatedSalary"].dropna() < 0).any():
        errors.append("EstimatedSalary cannot be negative.")

    if not numeric_df["HasCrCard"].dropna().isin([0, 1]).all():
        errors.append("HasCrCard must contain only 0 or 1.")

    if not numeric_df["IsActiveMember"].dropna().isin([0, 1]).all():
        errors.append("IsActiveMember must contain only 0 or 1.")

    return list(dict.fromkeys(errors))


def preprocess_data(df):
    """Clean dataset and prepare features and target."""

    validation_errors = validate_data(df)

    if validation_errors:
        raise ValueError(
            "Dataset validation failed: "
            + " ".join(validation_errors)
        )

    # Remove unnecessary columns
    df = df.drop(
        columns=["RowNumber", "CustomerId", "Surname"],
        errors="ignore"
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Separate target and input features
    target = "Exited"

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    # Convert categorical columns into numeric columns
    X = pd.get_dummies(X, drop_first=True, dtype=int)

    return X, y


if __name__ == "__main__":
    df = load_data()

    errors = validate_data(df)

    if errors:
        print("Dataset validation failed:")
        for error in errors:
            print("-", error)
    else:
        X, y = preprocess_data(df)

        print("Dataset loaded and validated successfully!")
        print("Total records:", len(df))
        print("Input features:", X.shape[1])
        print("\nTarget distribution:")
        print(y.value_counts())