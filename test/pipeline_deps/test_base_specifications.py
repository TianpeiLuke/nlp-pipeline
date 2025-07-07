"""
Unit tests for base specifications module.

Tests the Pydantic V2 migration of base specifications including:
- DependencySpec validation and functionality
- OutputSpec validation and functionality  
- PropertyReference functionality
- StepSpecification validation and functionality
- SpecificationRegistry functionality
- Serialization/deserialization capabilities
"""

import unittest
import json
from typing import List, Dict

from src.pipeline_deps.base_specifications import (
    DependencySpec, OutputSpec, PropertyReference, StepSpecification,
    DependencyType, NodeType
)
from src.pipeline_deps.specification_registry import SpecificationRegistry


class TestDependencySpec(unittest.TestCase):
    """Test cases for DependencySpec class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a fresh instance of the enum for each test to ensure isolation
        self.dependency_type = DependencyType.PROCESSING_OUTPUT
        self.valid_dependency_data = {
            "logical_name": "training_data",
            "dependency_type": DependencyType.PROCESSING_OUTPUT,
            "required": True,
            "compatible_sources": ["DataLoadingStep", "PreprocessingStep"],
            "semantic_keywords": ["data", "dataset", "input"],
            "data_type": "S3Uri",
            "description": "Training dataset for model training"
        }
    
    def test_valid_dependency_creation(self):
        """Test creating a valid DependencySpec."""
        dep_spec = DependencySpec(**self.valid_dependency_data)
        
        self.assertEqual(dep_spec.logical_name, "training_data")
        self.assertEqual(dep_spec.dependency_type, DependencyType.PROCESSING_OUTPUT)
        self.assertTrue(dep_spec.required)
        self.assertIn("DataLoadingStep", dep_spec.compatible_sources)
        self.assertIn("data", dep_spec.semantic_keywords)
        self.assertEqual(dep_spec.data_type, "S3Uri")
        self.assertEqual(dep_spec.description, "Training dataset for model training")
    
    def test_dependency_with_string_enum(self):
        """Test creating DependencySpec with string enum values."""
        data = self.valid_dependency_data.copy()
        data["dependency_type"] = "processing_output"  # String instead of enum
        
        dep_spec = DependencySpec(**data)
        self.assertEqual(dep_spec.dependency_type, DependencyType.PROCESSING_OUTPUT)
    
    def test_dependency_defaults(self):
        """Test DependencySpec with default values."""
        dep_spec = DependencySpec(
            logical_name="test_dep",
            dependency_type=DependencyType.MODEL_ARTIFACTS
        )
        
        self.assertTrue(dep_spec.required)  # Default is True
        self.assertEqual(dep_spec.compatible_sources, [])  # Default empty list
        self.assertEqual(dep_spec.semantic_keywords, [])  # Default empty list
        self.assertEqual(dep_spec.data_type, "S3Uri")  # Default
        self.assertEqual(dep_spec.description, "")  # Default empty string
    
    def test_logical_name_validation(self):
        """Test logical name validation."""
        # Empty logical name should fail
        with self.assertRaises(ValueError) as context:
            DependencySpec(
                logical_name="",
                dependency_type=DependencyType.PROCESSING_OUTPUT
            )
        # Pydantic V2 uses different error messages
        self.assertTrue("String should have at least 1 character" in str(context.exception) or 
                       "logical_name cannot be empty" in str(context.exception))
        
        # Whitespace-only logical name should fail
        with self.assertRaises(ValueError) as context:
            DependencySpec(
                logical_name="   ",
                dependency_type=DependencyType.PROCESSING_OUTPUT
            )
        self.assertIn("logical_name cannot be empty", str(context.exception))
        
        # Invalid characters should fail
        with self.assertRaises(ValueError) as context:
            DependencySpec(
                logical_name="invalid@name",
                dependency_type=DependencyType.PROCESSING_OUTPUT
            )
        self.assertIn("alphanumeric characters", str(context.exception))
    
    def test_dependency_type_validation(self):
        """Test dependency type validation."""
        # Invalid string enum value should fail
        with self.assertRaises(ValueError) as context:
            DependencySpec(
                logical_name="test_dep",
                dependency_type="invalid_type"
            )
        # Check for either custom or Pydantic error message
        error_msg = str(context.exception)
        self.assertTrue("dependency_type must be one of" in error_msg or 
                       "Input should be" in error_msg)
        
        # Invalid type should fail
        with self.assertRaises(ValueError) as context:
            DependencySpec(
                logical_name="test_dep",
                dependency_type=123
            )
        error_msg = str(context.exception)
        self.assertTrue("dependency_type must be a DependencyType" in error_msg or
                       "Input should be" in error_msg)
    
    def test_compatible_sources_validation(self):
        """Test compatible sources list validation."""
        # Non-list should fail
        with self.assertRaises(ValueError) as context:
            DependencySpec(
                logical_name="test_dep",
                dependency_type=DependencyType.PROCESSING_OUTPUT,
                compatible_sources="not_a_list"
            )
        error_msg = str(context.exception)
        self.assertTrue("compatible_sources must be a list" in error_msg or
                       "Input should be a valid list" in error_msg)
        
        # Empty strings should be filtered out
        dep_spec = DependencySpec(
            logical_name="test_dep",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            compatible_sources=["ValidStep", "", "  ", "AnotherStep"]
        )
        self.assertEqual(len(dep_spec.compatible_sources), 2)
        self.assertIn("ValidStep", dep_spec.compatible_sources)
        self.assertIn("AnotherStep", dep_spec.compatible_sources)
    
    def test_semantic_keywords_validation(self):
        """Test semantic keywords list validation."""
        # Non-list should fail
        with self.assertRaises(ValueError) as context:
            DependencySpec(
                logical_name="test_dep",
                dependency_type=DependencyType.PROCESSING_OUTPUT,
                semantic_keywords="not_a_list"
            )
        error_msg = str(context.exception)
        self.assertTrue("semantic_keywords must be a list" in error_msg or
                       "Input should be a valid list" in error_msg)
        
        # Keywords should be cleaned and lowercased
        dep_spec = DependencySpec(
            logical_name="test_dep",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            semantic_keywords=["DATA", "  Dataset  ", "", "Input", "DATA"]  # Duplicates and case
        )
        self.assertEqual(len(dep_spec.semantic_keywords), 3)  # Duplicates removed
        self.assertIn("data", dep_spec.semantic_keywords)  # Lowercased
        self.assertIn("dataset", dep_spec.semantic_keywords)  # Trimmed and lowercased
        self.assertIn("input", dep_spec.semantic_keywords)
    
    def test_serialization(self):
        """Test Pydantic serialization capabilities."""
        dep_spec = DependencySpec(**self.valid_dependency_data)
        
        # Test JSON serialization
        json_str = dep_spec.model_dump_json()
        self.assertIsInstance(json_str, str)
        
        # Test dict serialization
        dict_data = dep_spec.model_dump()
        self.assertIsInstance(dict_data, dict)
        self.assertEqual(dict_data["logical_name"], "training_data")
        
        # Convert the string dependency_type back to enum for validation
        dict_data_copy = dict_data.copy()
        dict_data_copy["dependency_type"] = DependencyType(dict_data["dependency_type"])
        
        # Test deserialization
        new_spec = DependencySpec.model_validate(dict_data_copy)
        self.assertEqual(new_spec.logical_name, dep_spec.logical_name)
        self.assertEqual(new_spec.dependency_type, dep_spec.dependency_type)
    
    def test_json_schema_generation(self):
        """Test JSON schema generation."""
        schema = DependencySpec.model_json_schema()
        
        self.assertIsInstance(schema, dict)
        self.assertIn("properties", schema)
        self.assertIn("logical_name", schema["properties"])
        self.assertIn("dependency_type", schema["properties"])
        self.assertIn("examples", schema)


class TestOutputSpec(unittest.TestCase):
    """Test cases for OutputSpec class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a fresh instance of the enum for each test to ensure isolation
        self.output_type = DependencyType.PROCESSING_OUTPUT
        self.valid_output_data = {
            "logical_name": "processed_data",
            "output_type": DependencyType.PROCESSING_OUTPUT,
            "property_path": "properties.ProcessingOutputConfig.Outputs['ProcessedData'].S3Output.S3Uri",
            "data_type": "S3Uri",
            "description": "Processed training data output"
        }
    
    def test_valid_output_creation(self):
        """Test creating a valid OutputSpec."""
        output_spec = OutputSpec(**self.valid_output_data)
        
        self.assertEqual(output_spec.logical_name, "processed_data")
        self.assertEqual(output_spec.output_type, DependencyType.PROCESSING_OUTPUT)
        self.assertTrue(output_spec.property_path.startswith("properties."))
        self.assertEqual(output_spec.data_type, "S3Uri")
        self.assertEqual(output_spec.description, "Processed training data output")
    
    def test_output_with_string_enum(self):
        """Test creating OutputSpec with string enum values."""
        data = self.valid_output_data.copy()
        data["output_type"] = "model_artifacts"  # String instead of enum
        
        output_spec = OutputSpec(**data)
        self.assertEqual(output_spec.output_type, DependencyType.MODEL_ARTIFACTS)
    
    def test_output_defaults(self):
        """Test OutputSpec with default values."""
        output_spec = OutputSpec(
            logical_name="test_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts"
        )
        
        self.assertEqual(output_spec.data_type, "S3Uri")  # Default
        self.assertEqual(output_spec.description, "")  # Default empty string
    
    def test_logical_name_validation(self):
        """Test logical name validation."""
        # Empty logical name should fail
        with self.assertRaises(ValueError) as context:
            OutputSpec(
                logical_name="",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.test"
            )
        self.assertTrue("String should have at least 1 character" in str(context.exception) or 
                       "logical_name cannot be empty" in str(context.exception))
        
        # Invalid characters should fail
        with self.assertRaises(ValueError) as context:
            OutputSpec(
                logical_name="invalid@name",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.test"
            )
        self.assertIn("alphanumeric characters", str(context.exception))
    
    def test_output_type_validation(self):
        """Test output type validation."""
        # Invalid string enum value should fail
        with self.assertRaises(ValueError) as context:
            OutputSpec(
                logical_name="test_output",
                output_type="invalid_type",
                property_path="properties.test"
            )
        error_msg = str(context.exception)
        self.assertTrue("output_type must be one of" in error_msg or
                       "Input should be" in error_msg)
    
    def test_property_path_validation(self):
        """Test property path validation."""
        # Empty property path should fail
        with self.assertRaises(ValueError) as context:
            OutputSpec(
                logical_name="test_output",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path=""
            )
        self.assertTrue("String should have at least 1 character" in str(context.exception) or
                       "property_path cannot be empty" in str(context.exception))
        
        # Property path not starting with 'properties.' should fail
        with self.assertRaises(ValueError) as context:
            OutputSpec(
                logical_name="test_output",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="invalid.path"
            )
        self.assertIn("property_path should start with 'properties.'", str(context.exception))
    
    def test_serialization(self):
        """Test Pydantic serialization capabilities."""
        output_spec = OutputSpec(**self.valid_output_data)
        
        # Test JSON serialization
        json_str = output_spec.model_dump_json()
        self.assertIsInstance(json_str, str)
        
        # Test dict serialization
        dict_data = output_spec.model_dump()
        self.assertIsInstance(dict_data, dict)
        self.assertEqual(dict_data["logical_name"], "processed_data")
        
        # Convert the string output_type back to enum for validation
        dict_data_copy = dict_data.copy()
        dict_data_copy["output_type"] = DependencyType(dict_data["output_type"])
        
        # Test deserialization
        new_spec = OutputSpec.model_validate(dict_data_copy)
        self.assertEqual(new_spec.logical_name, output_spec.logical_name)
        self.assertEqual(new_spec.output_type, output_spec.output_type)


class TestPropertyReference(unittest.TestCase):
    """Test cases for PropertyReference class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.output_spec = OutputSpec(
            logical_name="model_artifacts",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts"
        )
    
    def test_valid_property_reference_creation(self):
        """Test creating a valid PropertyReference."""
        prop_ref = PropertyReference(
            step_name="training_step",
            output_spec=self.output_spec
        )
        
        self.assertEqual(prop_ref.step_name, "training_step")
        self.assertEqual(prop_ref.output_spec.logical_name, "model_artifacts")
    
    def test_step_name_validation(self):
        """Test step name validation."""
        # Empty step name should fail
        with self.assertRaises(ValueError) as context:
            PropertyReference(
                step_name="",
                output_spec=self.output_spec
            )
        self.assertTrue("String should have at least 1 character" in str(context.exception) or
                       "step_name cannot be empty" in str(context.exception))
        
        # Whitespace-only step name should fail
        with self.assertRaises(ValueError) as context:
            PropertyReference(
                step_name="   ",
                output_spec=self.output_spec
            )
        self.assertIn("step_name cannot be empty", str(context.exception))
    
    def test_sagemaker_property_conversion(self):
        """Test conversion to SageMaker property format."""
        prop_ref = PropertyReference(
            step_name="training_step",
            output_spec=self.output_spec
        )
        
        sagemaker_prop = prop_ref.to_sagemaker_property()
        expected = {"Get": "Steps.training_step.properties.ModelArtifacts.S3ModelArtifacts"}
        self.assertEqual(sagemaker_prop, expected)
    
    def test_string_representation(self):
        """Test string representation methods."""
        prop_ref = PropertyReference(
            step_name="training_step",
            output_spec=self.output_spec
        )
        
        # Test __str__
        self.assertEqual(str(prop_ref), "training_step.model_artifacts")
        
        # Test __repr__
        expected_repr = "PropertyReference(step='training_step', output='model_artifacts')"
        self.assertEqual(repr(prop_ref), expected_repr)


class TestStepSpecification(unittest.TestCase):
    """Test cases for StepSpecification class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create fresh instances of the enums for each test to ensure isolation
        self.node_type_source = NodeType.SOURCE
        self.node_type_internal = NodeType.INTERNAL
        self.node_type_sink = NodeType.SINK
        self.node_type_singular = NodeType.SINGULAR
        self.dependency_type = DependencyType.PROCESSING_OUTPUT
        self.dep_spec = DependencySpec(
            logical_name="input_data",
            dependency_type=self.dependency_type,
            required=True,
            compatible_sources=["DataLoadingStep"]
        )
        
        self.output_spec = OutputSpec(
            logical_name="processed_data",
            output_type=self.dependency_type,
            property_path="properties.ProcessingOutputConfig.Outputs['ProcessedData'].S3Output.S3Uri"
        )
    
    def test_internal_node_creation(self):
        """Test creating an INTERNAL node (has both dependencies and outputs)."""
        step_spec = StepSpecification(
            step_type="DataProcessingStep",
            node_type=self.node_type_internal,
            dependencies=[self.dep_spec],
            outputs=[self.output_spec]
        )
        
        self.assertEqual(step_spec.step_type, "DataProcessingStep")
        self.assertEqual(step_spec.node_type, NodeType.INTERNAL)
        self.assertEqual(len(step_spec.dependencies), 1)
        self.assertEqual(len(step_spec.outputs), 1)
        
        # Test dictionary access
        self.assertIn("input_data", step_spec.dependencies)
        self.assertIn("processed_data", step_spec.outputs)
    
    def test_source_node_creation(self):
        """Test creating a SOURCE node (no dependencies, has outputs)."""
        step_spec = StepSpecification(
            step_type="DataLoadingStep",
            node_type=self.node_type_source,
            dependencies=[],
            outputs=[self.output_spec]
        )
        
        self.assertEqual(step_spec.node_type, self.node_type_source)
        self.assertEqual(len(step_spec.dependencies), 0)
        self.assertEqual(len(step_spec.outputs), 1)
    
    def test_sink_node_creation(self):
        """Test creating a SINK node (has dependencies, no outputs)."""
        step_spec = StepSpecification(
            step_type="ModelRegistrationStep",
            node_type=self.node_type_sink,
            dependencies=[self.dep_spec],
            outputs=[]
        )
        
        self.assertEqual(step_spec.node_type, self.node_type_sink)
        self.assertEqual(len(step_spec.dependencies), 1)
        self.assertEqual(len(step_spec.outputs), 0)
    
    def test_singular_node_creation(self):
        """Test creating a SINGULAR node (no dependencies, no outputs)."""
        step_spec = StepSpecification(
            step_type="StandaloneStep",
            node_type=self.node_type_singular,
            dependencies=[],
            outputs=[]
        )
        
        self.assertEqual(step_spec.node_type, self.node_type_singular)
        self.assertEqual(len(step_spec.dependencies), 0)
        self.assertEqual(len(step_spec.outputs), 0)
    
    def test_node_type_validation_failures(self):
        """Test node type validation failures."""
        # SOURCE node with dependencies should fail
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="InvalidSource",
                node_type=self.node_type_source,
                dependencies=[self.dep_spec],
                outputs=[self.output_spec]
            )
        self.assertIn("SOURCE node", str(context.exception))
        self.assertIn("cannot have dependencies", str(context.exception))
        
        # SOURCE node without outputs should fail
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="InvalidSource",
                node_type=self.node_type_source,
                dependencies=[],
                outputs=[]
            )
        self.assertIn("SOURCE node", str(context.exception))
        self.assertIn("must have outputs", str(context.exception))
        
        # INTERNAL node without dependencies should fail
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="InvalidInternal",
                node_type=self.node_type_internal,
                dependencies=[],
                outputs=[self.output_spec]
            )
        self.assertIn("INTERNAL node", str(context.exception))
        self.assertIn("must have dependencies", str(context.exception))
        
        # SINK node with outputs should fail
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="InvalidSink",
                node_type=self.node_type_sink,
                dependencies=[self.dep_spec],
                outputs=[self.output_spec]
            )
        self.assertIn("SINK node", str(context.exception))
        self.assertIn("cannot have outputs", str(context.exception))
    
    def test_duplicate_logical_names(self):
        """Test validation of duplicate logical names."""
        # Duplicate dependency logical names should fail
        dep_spec2 = DependencySpec(
            logical_name="input_data",  # Same as self.dep_spec
            dependency_type=DependencyType.MODEL_ARTIFACTS
        )
        
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="TestStep",
                node_type=NodeType.INTERNAL,
                dependencies=[self.dep_spec, dep_spec2],
                outputs=[self.output_spec]
            )
        self.assertIn("Duplicate dependency logical names", str(context.exception))
        
        # Duplicate output logical names should fail
        output_spec2 = OutputSpec(
            logical_name="processed_data",  # Same as self.output_spec
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts"
        )
        
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="TestStep",
                node_type=NodeType.INTERNAL,
                dependencies=[self.dep_spec],
                outputs=[self.output_spec, output_spec2]
            )
        self.assertIn("Duplicate output logical names", str(context.exception))
    
    def test_step_type_validation(self):
        """Test step type validation."""
        # Empty step type should fail
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="",
                node_type=NodeType.SOURCE,
                dependencies=[],
                outputs=[self.output_spec]
            )
        self.assertTrue("String should have at least 1 character" in str(context.exception) or
                       "step_type cannot be empty" in str(context.exception))
    
    def test_node_type_string_validation(self):
        """Test node type validation with string values."""
        step_spec = StepSpecification(
            step_type="TestStep",
            node_type="source",  # String instead of enum
            dependencies=[],
            outputs=[self.output_spec]
        )
        # With use_enum_values=False, we expect an enum instance
        self.assertIsInstance(step_spec.node_type, NodeType)
        self.assertEqual(step_spec.node_type, NodeType.SOURCE)
    
    def test_dependency_and_output_access(self):
        """Test dependency and output access methods."""
        step_spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal,
            dependencies=[self.dep_spec],
            outputs=[self.output_spec]
        )
        
        # Test get_dependency
        retrieved_dep = step_spec.get_dependency("input_data")
        self.assertIsNotNone(retrieved_dep)
        self.assertEqual(retrieved_dep.logical_name, "input_data")
        
        # Test get_output
        retrieved_output = step_spec.get_output("processed_data")
        self.assertIsNotNone(retrieved_output)
        self.assertEqual(retrieved_output.logical_name, "processed_data")
        
        # Test non-existent dependency/output
        self.assertIsNone(step_spec.get_dependency("non_existent"))
        self.assertIsNone(step_spec.get_output("non_existent"))
    
    def test_dependency_filtering_methods(self):
        """Test dependency filtering methods."""
        optional_dep = DependencySpec(
            logical_name="optional_data",
            dependency_type=DependencyType.HYPERPARAMETERS,
            required=False
        )
        
        step_spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal,
            dependencies=[self.dep_spec, optional_dep],
            outputs=[self.output_spec]
        )
        
        # Test list_required_dependencies
        required_deps = step_spec.list_required_dependencies()
        self.assertEqual(len(required_deps), 1)
        self.assertEqual(required_deps[0].logical_name, "input_data")
        
        # Test list_optional_dependencies
        optional_deps = step_spec.list_optional_dependencies()
        self.assertEqual(len(optional_deps), 1)
        self.assertEqual(optional_deps[0].logical_name, "optional_data")
        
        # Test list_dependencies_by_type
        processing_deps = step_spec.list_dependencies_by_type(DependencyType.PROCESSING_OUTPUT)
        self.assertEqual(len(processing_deps), 1)
        self.assertEqual(processing_deps[0].logical_name, "input_data")
        
        hyperparam_deps = step_spec.list_dependencies_by_type(DependencyType.HYPERPARAMETERS)
        self.assertEqual(len(hyperparam_deps), 1)
        self.assertEqual(hyperparam_deps[0].logical_name, "optional_data")
    
    def test_output_filtering_methods(self):
        """Test output filtering methods."""
        model_output = OutputSpec(
            logical_name="model_artifacts",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts"
        )
        
        step_spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal,
            dependencies=[self.dep_spec],
            outputs=[self.output_spec, model_output]
        )
        
        # Test list_outputs_by_type
        processing_outputs = step_spec.list_outputs_by_type(DependencyType.PROCESSING_OUTPUT)
        self.assertEqual(len(processing_outputs), 1)
        self.assertEqual(processing_outputs[0].logical_name, "processed_data")
        
        model_outputs = step_spec.list_outputs_by_type(DependencyType.MODEL_ARTIFACTS)
        self.assertEqual(len(model_outputs), 1)
        self.assertEqual(model_outputs[0].logical_name, "model_artifacts")
    
    def test_legacy_validate_method(self):
        """Test legacy validate method for backward compatibility."""
        step_spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal,
            dependencies=[self.dep_spec],
            outputs=[self.output_spec]
        )
        
        errors = step_spec.validate()
        self.assertEqual(len(errors), 0)  # Should have no errors
    
    def test_repr_method(self):
        """Test __repr__ method."""
        step_spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal,
            dependencies=[self.dep_spec],
            outputs=[self.output_spec]
        )
        
        repr_str = repr(step_spec)
        self.assertIn("TestStep", repr_str)
        self.assertIn("dependencies=1", repr_str)
        self.assertIn("outputs=1", repr_str)


class TestSpecificationRegistry(unittest.TestCase):
    """Test cases for SpecificationRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = SpecificationRegistry()
        
        # Create fresh instances of the enums for each test to ensure isolation
        self.node_type_source = NodeType.SOURCE
        self.dependency_type = DependencyType.PROCESSING_OUTPUT
        
        self.output_spec = OutputSpec(
            logical_name="test_output",
            output_type=self.dependency_type,
            property_path="properties.ProcessingOutputConfig.Outputs['TestOutput'].S3Output.S3Uri"
        )
        
        self.step_spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_source,
            dependencies=[],
            outputs=[self.output_spec]
        )
    
    def test_register_specification(self):
        """Test registering a specification."""
        self.registry.register("test_step", self.step_spec)
        
        # Test retrieval
        retrieved_spec = self.registry.get_specification("test_step")
        self.assertIsNotNone(retrieved_spec)
        self.assertEqual(retrieved_spec.step_type, "TestStep")
    
    def test_register_invalid_specification(self):
        """Test registering an invalid specification."""
        with self.assertRaises(ValueError) as context:
            self.registry.register("invalid_step", "not_a_specification")
        self.assertIn("specification must be a StepSpecification instance", str(context.exception))
    
    def test_list_methods(self):
        """Test listing methods."""
        self.registry.register("test_step", self.step_spec)
        
        # Test list_step_names
        step_names = self.registry.list_step_names()
        self.assertIn("test_step", step_names)
        
        # Test list_step_types
        step_types = self.registry.list_step_types()
        self.assertIn("TestStep", step_types)
    
    def test_get_specifications_by_type(self):
        """Test getting specifications by type."""
        self.registry.register("test_step", self.step_spec)
        
        specs = self.registry.get_specifications_by_type("TestStep")
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].step_type, "TestStep")
        
        # Test non-existent type
        specs = self.registry.get_specifications_by_type("NonExistentType")
        self.assertEqual(len(specs), 0)
    
    def test_find_compatible_outputs(self):
        """Test finding compatible outputs."""
        # Register the specification
        self.registry.register("test_step", self.step_spec)
        
        # Create a dependency that should match
        dep_spec = DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            data_type="S3Uri",
            compatible_sources=["TestStep"],
            semantic_keywords=["test", "output"]
        )
        
        compatible = self.registry.find_compatible_outputs(dep_spec)
        self.assertEqual(len(compatible), 1)
        
        step_name, output_name, output_spec, score = compatible[0]
        self.assertEqual(step_name, "test_step")
        self.assertEqual(output_name, "test_output")
        self.assertGreater(score, 0.5)  # Should have a good compatibility score
    
    def test_compatibility_scoring(self):
        """Test compatibility scoring logic."""
        # Register the specification
        self.registry.register("test_step", self.step_spec)
        
        # Test with compatible source bonus
        dep_spec_with_source = DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            data_type="S3Uri",
            compatible_sources=["TestStep"]
        )
        
        compatible_with_source = self.registry.find_compatible_outputs(dep_spec_with_source)
        score_with_source = compatible_with_source[0][3]
        
        # Test without compatible source
        dep_spec_without_source = DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            data_type="S3Uri",
            compatible_sources=["OtherStep"]
        )
        
        compatible_without_source = self.registry.find_compatible_outputs(dep_spec_without_source)
        score_without_source = compatible_without_source[0][3]
        
        # Score with compatible source should be higher
        self.assertGreater(score_with_source, score_without_source)


class TestEnumValidation(unittest.TestCase):
    """Test cases for enum validation across all classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create fresh instances of the enums for each test to ensure isolation
        self.node_type_source = NodeType.SOURCE
        self.node_type_internal = NodeType.INTERNAL
        self.node_type_sink = NodeType.SINK
        self.node_type_singular = NodeType.SINGULAR
    
    def test_dependency_type_enum_values(self):
        """Test all DependencyType enum values."""
        valid_values = [
            "model_artifacts",
            "processing_output", 
            "training_data",
            "hyperparameters",
            "payload_samples",
            "custom_property"
        ]
        
        for value in valid_values:
            # Test in DependencySpec
            dep_spec = DependencySpec(
                logical_name="test_dep",
                dependency_type=value
            )
            self.assertIsInstance(dep_spec.dependency_type, DependencyType)
            
            # Test in OutputSpec
            output_spec = OutputSpec(
                logical_name="test_output",
                output_type=value,
                property_path="properties.test.path"
            )
            self.assertIsInstance(output_spec.output_type, DependencyType)
    
    def test_node_type_enum_values(self):
        """Test all NodeType enum values."""
        valid_values = [
            self.node_type_source,
            self.node_type_internal,
            self.node_type_sink,
            self.node_type_singular
        ]
        
        for value in valid_values:
            if value == NodeType.SOURCE:
                # SOURCE nodes need outputs but no dependencies
                step_spec = StepSpecification(
                    step_type="TestStep",
                    node_type=value,
                    dependencies=[],
                    outputs=[OutputSpec(
                        logical_name="test_output",
                        output_type=DependencyType.PROCESSING_OUTPUT,
                        property_path="properties.test.path"
                    )]
                )
            elif value == NodeType.SINK:
                # SINK nodes need dependencies but no outputs
                step_spec = StepSpecification(
                    step_type="TestStep",
                    node_type=value,
                    dependencies=[DependencySpec(
                        logical_name="test_dep",
                        dependency_type=DependencyType.PROCESSING_OUTPUT
                    )],
                    outputs=[]
                )
            elif value == NodeType.SINGULAR:
                # SINGULAR nodes have no dependencies or outputs
                step_spec = StepSpecification(
                    step_type="TestStep",
                    node_type=value,
                    dependencies=[],
                    outputs=[]
                )
            else:  # internal
                # INTERNAL nodes need both dependencies and outputs
                step_spec = StepSpecification(
                    step_type="TestStep",
                    node_type=value,
                    dependencies=[DependencySpec(
                        logical_name="test_dep",
                        dependency_type=DependencyType.PROCESSING_OUTPUT
                    )],
                    outputs=[OutputSpec(
                        logical_name="test_output",
                        output_type=DependencyType.PROCESSING_OUTPUT,
                        property_path="properties.test.path"
                    )]
                )
            
            self.assertIsInstance(step_spec.node_type, NodeType)


class TestPydanticFeatures(unittest.TestCase):
    """Test cases for Pydantic V2 specific features."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create fresh instances of the enums for each test to ensure isolation
        self.node_type_internal = NodeType.INTERNAL
    
    def test_model_dump_and_validate(self):
        """Test model_dump and model_validate functionality."""
        # Create a complex specification
        dep_spec = DependencySpec(
            logical_name="training_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["DataLoadingStep"],
            semantic_keywords=["data", "training"],
            description="Training dataset"
        )
        
        output_spec = OutputSpec(
            logical_name="model_artifacts",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            description="Trained model artifacts"
        )
        
        step_spec = StepSpecification(
            step_type="TrainingStep",
            node_type=self.node_type_internal,
            dependencies=[dep_spec],
            outputs=[output_spec]
        )
        
        # Test model_dump
        dict_data = step_spec.model_dump()
        self.assertIsInstance(dict_data, dict)
        self.assertEqual(dict_data["step_type"], "TrainingStep")
        self.assertIn("dependencies", dict_data)
        self.assertIn("outputs", dict_data)
        
        # Test model_validate
        new_step_spec = StepSpecification.model_validate(dict_data)
        self.assertEqual(new_step_spec.step_type, step_spec.step_type)
        self.assertEqual(new_step_spec.node_type, step_spec.node_type)
        self.assertEqual(len(new_step_spec.dependencies), len(step_spec.dependencies))
        self.assertEqual(len(new_step_spec.outputs), len(step_spec.outputs))
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        dep_spec = DependencySpec(
            logical_name="test_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            semantic_keywords=["test", "data"]
        )
        
        # Test JSON serialization
        json_str = dep_spec.model_dump_json()
        self.assertIsInstance(json_str, str)
        
        # Parse JSON to verify it's valid
        import json
        parsed_data = json.loads(json_str)
        self.assertEqual(parsed_data["logical_name"], "test_data")
        self.assertEqual(parsed_data["dependency_type"], "processing_output")
        
        # Test deserialization from JSON
        new_dep_spec = DependencySpec.model_validate_json(json_str)
        self.assertEqual(new_dep_spec.logical_name, dep_spec.logical_name)
        self.assertEqual(new_dep_spec.dependency_type, dep_spec.dependency_type)
    
    def test_field_validation_errors(self):
        """Test that field validation provides helpful error messages."""
        # Test logical name validation error
        try:
            DependencySpec(
                logical_name="invalid@name!",
                dependency_type=DependencyType.PROCESSING_OUTPUT
            )
            self.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            self.assertIn("alphanumeric characters", error_msg)
        
        # Test property path validation error
        try:
            OutputSpec(
                logical_name="test_output",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="invalid_path"
            )
            self.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            self.assertIn("should start with 'properties.'", error_msg)
    
    def test_model_config_settings(self):
        """Test that model configuration settings work correctly."""
        # Test use_enum_values setting
        dep_spec = DependencySpec(
            logical_name="test_dep",
            dependency_type="processing_output"  # String value
        )
        self.assertEqual(dep_spec.dependency_type, DependencyType.PROCESSING_OUTPUT)
        
        # Test validate_assignment setting
        dep_spec.logical_name = "new_name"
        self.assertEqual(dep_spec.logical_name, "new_name")
        
        # Test that invalid assignment still fails
        with self.assertRaises(ValueError):
            dep_spec.logical_name = ""  # Should fail validation


class TestScriptContractIntegration(unittest.TestCase):
    """Test cases for script contract integration with StepSpecification."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create fresh instances of the enums for each test to ensure isolation
        self.node_type_internal = NodeType.INTERNAL
        self.node_type_source = NodeType.SOURCE
        self.dependency_type = DependencyType.PROCESSING_OUTPUT
        self.dep_spec = DependencySpec(
            logical_name="input_data",
            dependency_type=self.dependency_type,
            required=True,
            compatible_sources=["DataLoadingStep"]
        )
        
        self.output_spec = OutputSpec(
            logical_name="processed_data",
            output_type=self.dependency_type,
            property_path="properties.ProcessingOutputConfig.Outputs['ProcessedData'].S3Output.S3Uri"
        )
    
    def test_step_specification_without_script_contract(self):
        """Test StepSpecification without script contract (default behavior)."""
        step_spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal,
            dependencies=[self.dep_spec],
            outputs=[self.output_spec]
        )
        
        # script_contract should be None by default
        self.assertIsNone(step_spec.script_contract)
        
        # validate_script_compliance should return success when no contract is defined
        result = step_spec.validate_script_compliance("dummy_path.py")
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_step_specification_with_mock_script_contract(self):
        """Test StepSpecification with a mock script contract."""
        # Create a mock script contract
        from unittest.mock import Mock
        mock_contract = Mock()
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validation_result.errors = []
        mock_validation_result.warnings = []
        mock_contract.validate_implementation.return_value = mock_validation_result
        
        step_spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal,
            dependencies=[self.dep_spec],
            outputs=[self.output_spec],
            script_contract=mock_contract
        )
        
        # script_contract should be set
        self.assertIsNotNone(step_spec.script_contract)
        self.assertEqual(step_spec.script_contract, mock_contract)
        
        # validate_script_compliance should call the contract's validate_implementation
        result = step_spec.validate_script_compliance("test_script.py")
        self.assertTrue(result.is_valid)
        mock_contract.validate_implementation.assert_called_once_with("test_script.py")
    
    def test_step_specification_with_failing_script_contract(self):
        """Test StepSpecification with a script contract that fails validation."""
        # Create a mock script contract that fails validation
        from unittest.mock import Mock
        mock_contract = Mock()
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.errors = ["Script validation failed", "Missing required path"]
        mock_validation_result.warnings = ["Warning message"]
        mock_contract.validate_implementation.return_value = mock_validation_result
        
        step_spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal,
            dependencies=[self.dep_spec],
            outputs=[self.output_spec],
            script_contract=mock_contract
        )
        
        # validate_script_compliance should return the failing result
        result = step_spec.validate_script_compliance("failing_script.py")
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 2)
        self.assertIn("Script validation failed", result.errors)
        self.assertIn("Missing required path", result.errors)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("Warning message", result.warnings)
    
    def test_step_specification_serialization_with_script_contract(self):
        """Test that StepSpecification serialization works with script_contract field."""
        # Create a step spec without script contract
        step_spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal,
            dependencies=[self.dep_spec],
            outputs=[self.output_spec]
        )
        
        # Test serialization
        dict_data = step_spec.model_dump()
        self.assertIn("script_contract", dict_data)
        self.assertIsNone(dict_data["script_contract"])
        
        # Test deserialization
        new_step_spec = StepSpecification.model_validate(dict_data)
        self.assertIsNone(new_step_spec.script_contract)
        self.assertEqual(new_step_spec.step_type, step_spec.step_type)
    
    def test_backward_compatibility(self):
        """Test that existing code without script_contract still works."""
        # This tests that the new field doesn't break existing specifications
        step_spec = StepSpecification(
            step_type="LegacyStep",
            node_type=self.node_type_source,
            dependencies=[],
            outputs=[self.output_spec]
        )
        
        # All existing functionality should work
        self.assertEqual(step_spec.step_type, "LegacyStep")
        self.assertEqual(step_spec.node_type, NodeType.SOURCE)
        self.assertEqual(len(step_spec.dependencies), 0)
        self.assertEqual(len(step_spec.outputs), 1)
        
        # Legacy validate method should still work
        errors = step_spec.validate()
        self.assertEqual(len(errors), 0)
        
        # New script validation should work with no contract
        result = step_spec.validate_script_compliance("any_script.py")
        self.assertTrue(result.is_valid)


class TestStepSpecificationIntegration(unittest.TestCase):
    """Integration tests for updated step specifications with script contracts."""
    
    def test_model_eval_spec_integration(self):
        """Test that MODEL_EVAL_SPEC can be imported and has script contract."""
        try:
            from src.pipeline_step_specs.model_eval_spec import MODEL_EVAL_SPEC
            
            # Should be able to import without errors
            self.assertIsNotNone(MODEL_EVAL_SPEC)
            self.assertEqual(MODEL_EVAL_SPEC.step_type, "XGBoostModelEvaluation")
            self.assertEqual(MODEL_EVAL_SPEC.node_type, NodeType.INTERNAL)
            
            # Should have script contract
            self.assertIsNotNone(MODEL_EVAL_SPEC.script_contract)
            
            # Should be able to call validate_script_compliance
            result = MODEL_EVAL_SPEC.validate_script_compliance("dummy_script.py")
            # Result should be valid or invalid, but not error out
            self.assertIsInstance(result.is_valid, bool)
            
        except ImportError as e:
            self.fail(f"Failed to import MODEL_EVAL_SPEC: {e}")
    
    def test_preprocessing_training_spec_integration(self):
        """Test that PREPROCESSING_TRAINING_SPEC can be imported and has script contract."""
        try:
            from src.pipeline_step_specs.preprocessing_training_spec import PREPROCESSING_TRAINING_SPEC
            
            # Should be able to import without errors
            self.assertIsNotNone(PREPROCESSING_TRAINING_SPEC)
            self.assertEqual(PREPROCESSING_TRAINING_SPEC.step_type, "TabularPreprocessing_Training")
            self.assertEqual(PREPROCESSING_TRAINING_SPEC.node_type, NodeType.INTERNAL)
            
            # Should have script contract
            self.assertIsNotNone(PREPROCESSING_TRAINING_SPEC.script_contract)
            
            # Should be able to call validate_script_compliance
            result = PREPROCESSING_TRAINING_SPEC.validate_script_compliance("dummy_script.py")
            self.assertIsInstance(result.is_valid, bool)
            
        except ImportError as e:
            self.fail(f"Failed to import PREPROCESSING_TRAINING_SPEC: {e}")
    
    def test_xgboost_training_spec_integration(self):
        """Test that XGBOOST_TRAINING_SPEC can be imported and has script contract."""
        try:
            from src.pipeline_step_specs.xgboost_training_spec import XGBOOST_TRAINING_SPEC
            
            # Should be able to import without errors
            self.assertIsNotNone(XGBOOST_TRAINING_SPEC)
            self.assertEqual(XGBOOST_TRAINING_SPEC.step_type, "XGBoostTraining")
            self.assertEqual(XGBOOST_TRAINING_SPEC.node_type, NodeType.INTERNAL)
            
            # Should have script contract
            self.assertIsNotNone(XGBOOST_TRAINING_SPEC.script_contract)
            
            # Should be able to call validate_script_compliance
            result = XGBOOST_TRAINING_SPEC.validate_script_compliance("dummy_script.py")
            self.assertIsInstance(result.is_valid, bool)
            
        except ImportError as e:
            self.fail(f"Failed to import XGBOOST_TRAINING_SPEC: {e}")
    
    def test_no_circular_imports(self):
        """Test that there are no circular import issues."""
        try:
            # Import all the updated specifications
            from src.pipeline_step_specs.model_eval_spec import MODEL_EVAL_SPEC
            from src.pipeline_step_specs.preprocessing_training_spec import PREPROCESSING_TRAINING_SPEC
            from src.pipeline_step_specs.xgboost_training_spec import XGBOOST_TRAINING_SPEC
            
            # All should import successfully
            specs = [MODEL_EVAL_SPEC, PREPROCESSING_TRAINING_SPEC, XGBOOST_TRAINING_SPEC]
            for spec in specs:
                self.assertIsNotNone(spec)
                self.assertIsNotNone(spec.script_contract)
                
        except ImportError as e:
            self.fail(f"Circular import detected: {e}")


if __name__ == "__main__":
    unittest.main()
