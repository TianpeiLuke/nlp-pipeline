import unittest
from src.pipeline_deps import SemanticMatcher, OutputSpec, DependencyType

class TestSemanticMatcher(unittest.TestCase):
    def test_calculate_similarity_with_aliases(self):
        """Test that calculate_similarity_with_aliases returns the highest similarity."""
        matcher = SemanticMatcher()
        
        # Create test output spec with aliases
        output_spec = OutputSpec(
            logical_name="processed_data",
            aliases=["training_data", "input_features", "ModelData"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
        
        # Test with various dependency names
        test_cases = [
            # (dep_name, expected_min_score, expected_best_match)
            ("training_data", 1.0, "training_data"),        # Exact match with alias
            ("train_data", 0.7, "training_data"),           # Close match with alias
            ("processed_data", 1.0, "processed_data"),      # Exact match with logical_name
            ("model_data", 0.5, "ModelData"),               # Close match with alias
            ("feature_data", 0.5, "input_features"),        # Moderate match with alias
            ("unrelated_name", 0.1, "processed_data")       # No good match (defaults to logical)
        ]
        
        for dep_name, expected_min_score, expected_best_match in test_cases:
            score = matcher.calculate_similarity_with_aliases(dep_name, output_spec)
            self.assertGreaterEqual(score, expected_min_score, f"Score for '{dep_name}' below expectation")

    def test_weight_calculation(self):
        """Test that the weight calculation is correct."""
        matcher = SemanticMatcher()
        
        # Test with known values
        name1 = "training_data"
        name2 = "train_data"
        
        # Calculate scores
        norm1 = matcher._normalize_name(name1)
        norm2 = matcher._normalize_name(name2)
        string_sim = matcher._calculate_string_similarity(norm1, norm2)
        token_sim = matcher._calculate_token_similarity(norm1, norm2)
        semantic_sim = matcher._calculate_semantic_similarity(norm1, norm2)
        substring_sim = matcher._calculate_substring_similarity(norm1, norm2)
        
        # Expected score
        expected_score = (
            string_sim * 0.3 +
            token_sim * 0.25 +
            semantic_sim * 0.25 +
            substring_sim * 0.2
        )
        
        # Actual score
        actual_score = matcher.calculate_similarity(name1, name2)
        
        # Assert that the scores are equal
        self.assertAlmostEqual(expected_score, actual_score, places=5)

if __name__ == '__main__':
    unittest.main()
