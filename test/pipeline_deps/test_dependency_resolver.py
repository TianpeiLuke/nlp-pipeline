import unittest
from src.pipeline_deps import (
    UnifiedDependencyResolver,
    SpecificationRegistry,
    StepSpecification,
    OutputSpec,
    DependencySpec,
    DependencyType,
    NodeType,
)

class TestDependencyResolver(unittest.TestCase):
    def test_dependency_resolution_with_aliases(self):
        """Test that dependency resolution uses aliases for matching."""
        from src.pipeline_deps import SemanticMatcher
        registry = SpecificationRegistry()
        semantic_matcher = SemanticMatcher()
        resolver = UnifiedDependencyResolver(registry, semantic_matcher)
        
        # Create producer step specification with aliases
        producer_spec = StepSpecification(
            step_type="PreprocessingStep",
            node_type=NodeType.SOURCE,
            dependencies=[],  # No dependencies for this test
            outputs=[
                OutputSpec(
                    logical_name="processed_data",
                    aliases=["training_data", "model_input"],
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
                    data_type="S3Uri"
                )
            ]
        )
        
        # Create consumer step specification
        consumer_spec = StepSpecification(
            step_type="TrainingStep",
            node_type=NodeType.SINK,
            dependencies=[
                DependencySpec(
                    logical_name="training_data",
                    dependency_type=DependencyType.PROCESSING_OUTPUT,
                    required=True,
                    compatible_sources=["PreprocessingStep"],
                    data_type="S3Uri"
                )
            ],
            outputs=[]  # No outputs needed for this test
        )
        
        # Register step specifications
        registry.register("producer", producer_spec)
        registry.register("consumer", consumer_spec)
        
        # Resolve dependencies
        resolved = resolver.resolve_step_dependencies("consumer", ["producer"])
        
        # Assert that resolution was successful
        self.assertIn("training_data", resolved)
        self.assertEqual(resolved["training_data"].step_name, "producer")
        self.assertEqual(resolved["training_data"].output_spec.logical_name, "processed_data")

    def test_weight_calculation(self):
        """Test that the weight calculation is correct."""
        from src.pipeline_deps import SemanticMatcher
        registry = SpecificationRegistry()
        semantic_matcher = SemanticMatcher()
        resolver = UnifiedDependencyResolver(registry, semantic_matcher)
        
        # Create producer step specification
        producer_spec = StepSpecification(
            step_type="PreprocessingStep",
            node_type=NodeType.SOURCE,
            dependencies=[],
            outputs=[
                OutputSpec(
                    logical_name="processed_data",
                    output_type=DependencyType.PROCESSING_OUTPUT,
                    property_path="properties.test",
                    data_type="S3Uri"
                )
            ]
        )
        
        # Create consumer step specification
        consumer_spec = StepSpecification(
            step_type="TrainingStep",
            node_type=NodeType.SINK,
            dependencies=[
                DependencySpec(
                    logical_name="training_data",
                    dependency_type=DependencyType.PROCESSING_OUTPUT,
                    data_type="S3Uri",
                    compatible_sources=["PreprocessingStep"],
                    semantic_keywords=["data"]
                )
            ],
            outputs=[]
        )
        
        # Calculate scores
        dep_spec = consumer_spec.dependencies["training_data"]
        output_spec = producer_spec.outputs["processed_data"]
        
        type_compatibility_score = 0.4  # Exact match
        data_type_score = 0.2  # Exact match
        semantic_score = resolver.semantic_matcher.calculate_similarity_with_aliases(
            dep_spec.logical_name, output_spec
        ) * 0.25
        direct_match_bonus = 0.0
        compatible_source_score = 0.1
        keyword_match_score = resolver._calculate_keyword_match(dep_spec.semantic_keywords, output_spec.logical_name) * 0.05
        
        # Expected score
        expected_score = (
            type_compatibility_score +
            data_type_score +
            semantic_score +
            direct_match_bonus +
            compatible_source_score +
            keyword_match_score
        )
        
        # Actual score
        actual_score = resolver._calculate_compatibility(dep_spec, output_spec, producer_spec)
        
        # Assert that the scores are equal
        self.assertAlmostEqual(expected_score, actual_score, places=5)

if __name__ == '__main__':
    unittest.main()
