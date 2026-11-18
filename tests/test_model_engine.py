import unittest
import pandas as pd
from model_engine import validate_and_prepare, add_targets, train_and_evaluate, CLASS_FEATURES

class ModelEngineTests(unittest.TestCase):
    def test_accuracy_is_derived_when_missing(self):
        raw = pd.DataFrame([{"student_id":"s1", "week":1, "topic":"Arrays", "attempts":4,
                             "problems_solved":3, "hints_used":1, "avg_time_min":12}])
        result = validate_and_prepare(raw)
        self.assertAlmostEqual(result.loc[0, "accuracy"], 75.0)

    def test_duplicate_sessions_are_aggregated_by_week(self):
        row = {"student_id":"s1", "week":1, "topic":"Arrays", "attempts":4,
               "problems_solved":3, "hints_used":1, "avg_time_min":12}
        raw = pd.DataFrame([row, {**row, "attempts":2, "problems_solved":1, "hints_used":1, "avg_time_min":18}])
        result = validate_and_prepare(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "attempts"], 6)
        self.assertEqual(result.loc[0, "problems_solved"], 4)
        self.assertAlmostEqual(result.loc[0, "accuracy"], 400/6)

    def test_accuracy_is_not_a_classifier_input(self):
        self.assertNotIn("accuracy", CLASS_FEATURES)

    def test_demo_model_pipeline(self):
        raw = pd.read_csv("data/demo_coding_practice.csv")
        data = validate_and_prepare(raw)
        result = train_and_evaluate(data)
        self.assertGreater(result["classification_results"].iloc[0]["Test_Accuracy"], .70)
        self.assertLess(result["regression_results"].iloc[0]["Test_MAE"], 6.5)

if __name__ == "__main__":
    unittest.main()
