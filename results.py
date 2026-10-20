import pandas as pd

results = {
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Score": [
        "87.15%",
        "80.24%",
        "48.89%",
        "60.76%"
    ]
}

df = pd.DataFrame(results)

df.to_csv("model_evaluation_results.csv", index=False)

print("Model evaluation results saved successfully!")
print(df)