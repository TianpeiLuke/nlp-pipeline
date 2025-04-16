import ast
import json
import unittest



def serialize_config(config):
    safe_config = {}
    for k, v in config.items():
        if isinstance(v, (list, dict, bool)):
            safe_config[k] = json.dumps(v)
        else:
            safe_config[k] = str(v)
    return safe_config


def safe_cast(val):
    # Only process if it's a string
    if isinstance(val, str):
        val = val.strip()

        # Handle common string representations of booleans
        if val.lower() == "true":
            return True
        elif val.lower() == "false":
            return False

        # Handle JSON-style lists or dictionaries
        if (val.startswith("[") and val.endswith("]")) or (val.startswith("{") and val.endswith("}")):
            try:
                return json.loads(val)  # Safe parse for JSON-formatted strings
            except Exception:
                pass  # If JSON fails, fall back

        # Handle numbers, tuples, and literals (e.g. int, float, None, etc.)
        try:
            return ast.literal_eval(val)  # More flexible, but still safe
        except Exception:
            pass  # If it fails, treat as plain string

    # Return original if it's not a string or nothing parsed
    return val


def sanitize_config(config):
    for key, val in config.items():
        if isinstance(val, str) and val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        config[key] = safe_cast(val)
    return config




class TestConfigSanitization(unittest.TestCase):

    def test_serialize_and_sanitize_config(self):
        original_config = {
            "tokenizer": "bert-base-uncased",          # string
            "use_gpu": True,                           # boolean
            "dropout": 0.1,                            # float
            "tab_field_list": ["f1", "f2", "f3"],      # list
            "class_weights": [1.0, 0.5],               # list of floats
            "extra_settings": {"a": 1, "b": False}     # dict
        }

        serialized = serialize_config(original_config)
        sanitized = sanitize_config(serialized)

        # All values should match original
        self.assertEqual(sanitized["tokenizer"], original_config["tokenizer"])
        self.assertEqual(sanitized["use_gpu"], original_config["use_gpu"])
        self.assertEqual(sanitized["dropout"], original_config["dropout"])
        self.assertEqual(sanitized["tab_field_list"], original_config["tab_field_list"])
        self.assertEqual(sanitized["class_weights"], original_config["class_weights"])
        self.assertEqual(sanitized["extra_settings"], original_config["extra_settings"])

        # All types should be preserved
        self.assertIsInstance(sanitized["tokenizer"], str)
        self.assertIsInstance(sanitized["use_gpu"], bool)
        self.assertIsInstance(sanitized["dropout"], float)
        self.assertIsInstance(sanitized["tab_field_list"], list)
        self.assertIsInstance(sanitized["class_weights"], list)
        self.assertIsInstance(sanitized["extra_settings"], dict)

if __name__ == "__main__":
    unittest.main()