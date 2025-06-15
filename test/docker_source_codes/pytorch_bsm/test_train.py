import unittest
import os
import json
from nlp_pipeline.docker_source_code_examples.pytorch_bsm.train import load_parse_hyperparameters

class TestTrain(unittest.TestCase):
    def setUp(self):
        # Create a temporary hyperparameters file for testing
        self.temp_hparam_path = "temp_hyperparameters.json"
        self.hyperparameters = {
            "id_name": "test_id",
            "text_name": "test_text",
            "label_name": "test_label",
            "batch_size": 16,
            "max_sen_len": 128,
            "is_binary": True,
            "num_classes": 2,
        }
        with open(self.temp_hparam_path, "w") as f:
            json.dump(self.hyperparameters, f)

    def tearDown(self):
        # Remove the temporary hyperparameters file
        if os.path.exists(self.temp_hparam_path):
            os.remove(self.temp_hparam_path)

    def test_load_parse_hyperparameters(self):
        # Test if the function correctly parses the hyperparameters
        parsed_hyperparameters = load_parse_hyperparameters(self.temp_hparam_path)
        self.assertEqual(parsed_hyperparameters["id_name"], self.hyperparameters["id_name"])
        self.assertEqual(parsed_hyperparameters["batch_size"], self.hyperparameters["batch_size"])
        self.assertEqual(parsed_hyperparameters["is_binary"], self.hyperparameters["is_binary"])

if __name__ == "__main__":
    unittest.main()
