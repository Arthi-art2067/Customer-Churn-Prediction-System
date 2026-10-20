import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

from preprocessing import load_data, preprocess_data


def train_model():

    # Step 1: Load and preprocess dataset
    print("Loading dataset...")
    df = load_data()
    X, y = preprocess_data(df)

    print(f"Total records: {len(df)}")
    print(f"Total features: {X.shape[1]}")

    # Step 2: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # Step 3: Define machine learning models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced"
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced"
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    # Step 4: Train and compare models
    results = []
    best_model = None
    best_accuracy = -1

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        results.append((name, accuracy))

        print(f"Accuracy: {accuracy * 100:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    # Step 5: Predict using the selected model
    predictions = best_model.predict(X_test)

    # Step 6: Display model comparison
    print("\nMODEL COMPARISON")
    print("----------------")

    for name, accuracy in results:
        print(f"{name}: {accuracy * 100:.2f}%")

    # Step 7: Evaluate the best model
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print("\nBEST MODEL RESULTS")
    print("------------------")

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    # Step 8: Generate confusion matrix
    cm = confusion_matrix(y_test, predictions)

    print("\nCONFUSION MATRIX")
    print("----------------")
    print(cm)

    # Step 9: Save confusion matrix as an image
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stay", "Churn"]
    )

    display.plot()
    plt.title("Customer Churn Prediction - Confusion Matrix")
    plt.savefig(
        "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\nConfusion matrix image saved successfully!")
    print("Image file: confusion_matrix.png")

    # Step 10: Display classification report
    print("\nCLASSIFICATION REPORT")
    print("---------------------")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Stay", "Churn"],
            zero_division=0
        )
    )

    # Step 11: Save model and feature columns
    joblib.dump(best_model, "churn_model.pkl")
    joblib.dump(list(X.columns), "model_columns.pkl")

    print("Model saved: churn_model.pkl")
    print("Feature columns saved: model_columns.pkl")

    print("\nModel training and evaluation completed!")


if __name__ == "__main__":
    train_model()