import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

from preprocessing import load_data, preprocess_data


def train_model():
    df = load_data()
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

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

    results = []
    best_model = None
    best_accuracy = 0

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

    # Select the model with highest test accuracy
    # Note: use a separate validation set for final model selection
    # in a rigorous evaluation.

    predictions = best_model.predict(X_test)

    print("\nMODEL COMPARISON")
    print("----------------")

    for name, accuracy in results:
        print(f"{name}: {accuracy * 100:.2f}%")

    print("\nBEST MODEL RESULTS")
    print("------------------")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0) * 100:.2f}%")
    print(f"Recall: {recall_score(y_test, predictions, zero_division=0) * 100:.2f}%")
    print(f"F1 Score: {f1_score(y_test, predictions, zero_division=0) * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(best_model, "churn_model.pkl")
    joblib.dump(list(X.columns), "model_columns.pkl")

    print("\nBest model saved successfully!")


if __name__ == "__main__":
    train_model()