import pandas as pd


def load_data(file_path="Churn_Modelling.csv"):
    """Load the customer churn dataset."""
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    """Clean the dataset and prepare features and target."""

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
    y = df[target]

    # Convert categorical columns into numeric columns
    X = pd.get_dummies(X, drop_first=True, dtype=int)

    return X, y


if __name__ == "__main__":
    df = load_data()
    X, y = preprocess_data(df)

    print("Dataset loaded successfully!")
    print("Total records:", len(df))
    print("Input features:", X.shape[1])
    print("\nTarget distribution:")
    print(y.value_counts())