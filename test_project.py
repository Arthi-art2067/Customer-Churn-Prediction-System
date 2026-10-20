
import unittest
import pandas as pd
import numpy as np
import pickle
import csv
from pathlib import Path
from io import StringIO

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "Churn_Modelling.csv"
MODEL_PATH = BASE_DIR / "churn_model.pkl"
COLUMNS_PATH = BASE_DIR / "model_columns.pkl"
RESULTS_PATH = BASE_DIR / "model_evaluation_results.csv"


class TestCustomerChurnProject(unittest.TestCase):

    # Dataset Tests

    def test_1_dataset_exists(self):
        self.assertTrue(DATASET_PATH.exists())

    def test_2_dataset_has_enough_records(self):
        df = pd.read_csv(DATASET_PATH)
        self.assertGreaterEqual(len(df), 5000)

    def test_3_target_column_exists(self):
        df = pd.read_csv(DATASET_PATH)
        self.assertIn("Exited", df.columns)

    def test_4_target_values_are_valid(self):
        df = pd.read_csv(DATASET_PATH)
        self.assertTrue(
            set(df["Exited"].dropna().unique()).issubset({0, 1})
        )

    def test_5_dataset_has_no_missing_values(self):
        df = pd.read_csv(DATASET_PATH)
        self.assertEqual(int(df.isnull().sum().sum()), 0)

    def test_6_model_file_exists(self):
        self.assertTrue(MODEL_PATH.exists())

    def test_7_model_columns_match_model_features(self):
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

        with open(COLUMNS_PATH, "rb") as file:
            columns = pickle.load(file)

        self.assertEqual(model.n_features_in_, len(columns))

    def test_8_model_can_make_prediction(self):
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

        with open(COLUMNS_PATH, "rb") as file:
            columns = pickle.load(file)

        sample = np.zeros((1, len(columns)))
        prediction = model.predict(sample)

        self.assertIn(int(prediction[0]), [0, 1])

    def test_9_evaluation_results_exist(self):
        self.assertTrue(RESULTS_PATH.exists())

    def test_10_evaluation_results_have_metrics(self):
        results = pd.read_csv(RESULTS_PATH)

        self.assertFalse(results.empty)
        self.assertGreaterEqual(len(results.columns), 2)

    # Additional CSV Validation Tests

    def test_11_csv_can_be_read(self):
        df = pd.read_csv(DATASET_PATH)
        self.assertIsInstance(df, pd.DataFrame)

    def test_12_dataset_has_unique_column_names(self):
        df = pd.read_csv(DATASET_PATH)
        self.assertEqual(len(df.columns), len(set(df.columns)))

    def test_13_dataset_has_no_duplicate_rows(self):
        df = pd.read_csv(DATASET_PATH)
        self.assertEqual(int(df.duplicated().sum()), 0)

    def test_14_empty_csv_is_detected(self):
        empty_df = pd.DataFrame()
        self.assertTrue(empty_df.empty)

    def test_15_missing_values_are_detected(self):
        test_df = pd.DataFrame({
            "CreditScore": [600, None],
            "Exited": [0, 1]
        })

        self.assertGreater(
            int(test_df.isnull().sum().sum()), 0
        )

    def test_16_duplicate_columns_are_detected(self):
        csv_data = "Age,Age,Exited\n25,26,0\n30,31,1\n"

        header = next(csv.reader(StringIO(csv_data)))

        self.assertNotEqual(
            len(header),
            len(set(header))
        )

    def test_17_invalid_target_values_are_detected(self):
        test_df = pd.DataFrame({"Exited": [0, 1, 2]})

        self.assertFalse(
            test_df["Exited"].isin([0, 1]).all()
        )

    def test_18_model_probability_is_valid(self):
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

        with open(COLUMNS_PATH, "rb") as file:
            columns = pickle.load(file)

        sample = np.zeros((1, len(columns)))
        probabilities = model.predict_proba(sample)[0]

        self.assertAlmostEqual(
            float(np.sum(probabilities)), 1.0, places=5
        )

        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)