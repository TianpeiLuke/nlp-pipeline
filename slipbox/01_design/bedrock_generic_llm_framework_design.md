---
tags:
  - design
  - bedrock
  - llm_framework
  - refactoring
  - architecture
keywords:
  - AWS Bedrock
  - generic LLM framework
  - prompt templates
  - batch processing
  - concurrent processing
  - pydantic models
  - modular architecture
  - structured output
topics:
  - LLM framework design
  - bedrock integration
  - prompt engineering
  - batch processing architecture
language: python
date of note: 2025-01-24
---

# Generic Bedrock LLM Framework Design

## Executive Summary

This document outlines the design for refactoring the existing bedrock scripts into a generic, modular LLM framework that supports structured prompt templates, batch processing with concurrency, and formatted structured output using AWS Bedrock, boto3, and Pydantic.

## Current State Analysis

### Existing Issues
- **Code Redundancy**: ~40% duplicate code across agent implementations
- **Tight Coupling**: Hard-coded prompt templates and task-specific logic
- **Limited Reusability**: RnR-specific implementations not generalizable
- **Complex Threading**: Overly complex batch processing (removed in cleanup)
- **Inconsistent Architecture**: Multiple patterns for similar functionality

### Cleanup Completed
- ✅ Removed `main.py` (overly complex CLI)
- ✅ Removed `bedrock_batch_inference.py` (unnecessary threading complexity)
- ✅ Identified `bedrock_rnr_agent_v2.py` as most robust implementation
- ✅ Identified `rnr_bedrock_main.py` as most efficient processor

## Design Goals

### Primary Objectives
1. **Generic Framework**: Support multiple LLM tasks beyond RnR classification
2. **Modular Architecture**: Each component focuses on single responsibility
3. **Template Independence**: Configurable prompt structure without code changes
4. **Structured Output**: Consistent Pydantic-based response models
5. **Concurrent Processing**: Efficient batch processing with proper resource management
6. **Production Ready**: SageMaker compatible with proper error handling

### Secondary Objectives
- Maintain backward compatibility with existing RnR workflows
- Support multiple Bedrock models and inference profiles
- Enable easy extension for new task types
- Provide comprehensive logging and monitoring

## Prompt Template Structure

### Standardized Template Components

```yaml
# Prompt Template Schema
system_prompt:
  persona: "You are an expert in..."
  role: "Your task is to..."

task_selection:
  type: "categorization" | "summarization" | "planning" | "cataloging" | "validation"
  description: "Specific task description"

definition:
  categories: []  # For classification tasks
  key_points: []  # For summarization tasks
  requirements: []  # For planning tasks
  domain_knowledge: "Background information"

instruction:
  rules: []
  exceptions: []
  guidelines: []

user_input:
  data_schema:
    auto_generate: true  # Auto-generate from DataFrame
    fields: []  # Field definitions with types and descriptions
    required_fields: []  # Fields that must be present
    optional_fields: []  # Fields that are optional
  data_examples:
    auto_include: true  # Include sample data in prompt
    max_examples: 3  # Maximum number of examples to include
    anonymize: true  # Anonymize sensitive data in examples
  input_validation:
    type_checking: true  # Validate input data types
    range_validation: true  # Validate numeric ranges
    format_validation: true  # Validate string formats

format_structure:
  output_schema: "Pydantic model definition"
  required_fields: []
  optional_fields: []

reference:
  slipbox_links: []
  internal_docs: []
  external_resources: []
```

### Template Types by Task Category

#### 1. Categorization/Classification
```python
class ClassificationTemplate(BaseModel):
    system_prompt: SystemPrompt
    categories: List[CategoryDefinition]
    classification_rules: List[str]
    output_schema: Type[BaseModel]
    confidence_requirements: Dict[str, float]
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
```

#### 2. Summarization/Extraction
```python
class SummarizationTemplate(BaseModel):
    system_prompt: SystemPrompt
    extraction_targets: List[str]
    summary_style: str
    output_schema: Type[BaseModel]
    length_constraints: Dict[str, int]
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
```

#### 3. Planning
```python
class PlanningTemplate(BaseModel):
    system_prompt: SystemPrompt
    background_knowledge: str
    outcome_requirements: List[str]
    output_schema: Type[BaseModel]
    planning_constraints: Dict[str, Any]
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
```

#### 4. Cataloging
```python
class CatalogTemplate(BaseModel):
    system_prompt: SystemPrompt
    catalog_type: str  # "index", "taxonomy", "metadata", "classification_schema"
    indexing_rules: List[str]
    metadata_schema: Dict[str, Any]
    output_schema: Type[BaseModel]
    categorization_criteria: List[str]
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
```

#### 5. Validation/Review/Analysis
```python
class ValidationTemplate(BaseModel):
    system_prompt: SystemPrompt
    validation_criteria: List[str]
    analysis_dimensions: List[str]
    output_schema: Type[BaseModel]
    scoring_rubric: Dict[str, Any]
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
```

## Architecture Design

### Core Components

```
src/nlp_pipeline/
├── core/
│   ├── __init__.py
│   ├── base_agent.py              # Abstract base agent
│   ├── bedrock_client.py          # Bedrock client management
│   ├── prompt_manager.py          # Template loading and management
│   └── response_parser.py         # Generic response parsing
├── templates/
│   ├── __init__.py
│   ├── base_template.py           # Base template classes
│   ├── classification_template.py # Classification-specific templates
│   ├── summarization_template.py  # Summarization-specific templates
│   ├── planning_template.py       # Planning-specific templates
│   ├── catalog_template.py        # Cataloging-specific templates
│   └── validation_template.py     # Validation-specific templates
├── agents/
│   ├── __init__.py
│   ├── classification_agent.py    # Classification tasks
│   ├── summarization_agent.py     # Summarization tasks
│   ├── planning_agent.py          # Planning tasks
│   ├── catalog_agent.py           # Cataloging/indexing tasks
│   └── validation_agent.py        # Validation tasks
├── workflows/
│   ├── __init__.py
│   ├── base_workflow.py           # Base workflow orchestration
│   ├── classification_validation_workflow.py  # Classification + Validation
│   ├── summarization_catalog_workflow.py      # Summarization + Cataloging
│   ├── planning_validation_workflow.py        # Planning + Validation
│   └── custom_workflow_builder.py             # Dynamic workflow creation
├── langgraph/
│   ├── __init__.py
│   ├── graph_builder.py           # LangGraph integration
│   ├── node_definitions.py        # Agent nodes for graphs
│   ├── edge_conditions.py         # Conditional routing logic
│   └── state_management.py        # Workflow state handling
├── processors/
│   ├── __init__.py
│   ├── batch_processor.py         # Generic batch processing
│   ├── concurrent_processor.py    # Concurrent execution management
│   └── result_aggregator.py       # Result collection and formatting
├── models/
│   ├── __init__.py
│   ├── base_models.py             # Base Pydantic models
│   ├── classification_models.py   # Classification response models
│   ├── summarization_models.py    # Summarization response models
│   ├── planning_models.py         # Planning response models
│   ├── catalog_models.py          # Cataloging response models
│   └── validation_models.py       # Validation response models
├── utils/
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   ├── logging.py                 # Logging utilities
│   └── exceptions.py              # Custom exceptions
├── legacy_bedrock/
│   ├── __init__.py
│   ├── rnr_agent.py               # Backward compatibility
│   └── rnr_models.py              # Legacy RnR models
└── bedrock/                       # Existing bedrock scripts (to be migrated)
    ├── __init__.py
    ├── bedrock_rnr_agent.py       # Current RnR agent (v1)
    ├── bedrock_rnr_agent_v2.py    # Enhanced RnR agent (v2)
    ├── rnr_bedrock_main.py        # Current main processor
    ├── rnr_reason_code_models.py  # Current Pydantic models
    ├── invoke_bedrock.py          # Low-level API calls
    ├── prompt_rnr_parse.py        # Response parsing utilities
    ├── bedrock_batch_process_merge.py # Batch processing utilities
    ├── upload_s3.py               # S3 utilities
    ├── example_inference_profile_usage.py # Usage examples
    └── prompt_repo/               # Current prompt templates
```

### Component Responsibilities

#### 1. Core Components

**BaseAgent** (Abstract)
```python
class BaseAgent(ABC):
    def __init__(self, model_id: str, template: BaseTemplate):
        self.bedrock_client = BedrockClient(model_id)
        self.template = template
        self.prompt_manager = PromptManager()
        self.response_parser = ResponseParser()
    
    @abstractmethod
    async def process_single(self, input_data: Dict[str, Any]) -> BaseModel:
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        pass
```

**BedrockClient**
```python
class BedrockClient:
    def __init__(self, model_id: str, region: str = "us-west-2"):
        self.model_id = model_id
        self.client = self._create_client()
        self.inference_profile_manager = InferenceProfileManager()
    
    async def invoke_model(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Unified Bedrock API calling with retry logic
        pass
    
    def _handle_inference_profiles(self) -> str:
        # Auto-detect and configure inference profiles
        pass
```

**PromptManager**
```python
class PromptManager:
    def __init__(self, template_dir: Path = None):
        self.template_dir = template_dir or Path(__file__).parent.parent / "bedrock" / "prompt_repo"
        self.template_cache = {}
    
    def load_template(self, template_name: str) -> BaseTemplate:
        # Load and parse template files
        pass
    
    def render_prompt(self, template: BaseTemplate, **kwargs) -> str:
        # Render template with input data
        pass
    
    def validate_template(self, template: BaseTemplate) -> bool:
        # Validate template structure
        pass
```

#### 2. Template System

**BaseTemplate**
```python
class BaseTemplate(BaseModel):
    name: str
    task_type: TaskType
    system_prompt: SystemPrompt
    definition: Definition
    instruction: Instruction
    user_input: UserInputConfig
    format_structure: FormatStructure
    reference: Reference
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    
    def render(self, input_data: Dict[str, Any] = None, **kwargs) -> str:
        """Render complete prompt with user data integration."""
        prompt_parts = []
        
        # System prompt
        prompt_parts.append(self.system_prompt.render())
        
        # Task definition
        prompt_parts.append(self._render_definition())
        
        # Instructions
        prompt_parts.append(self._render_instructions())
        
        # User input schema and examples
        if input_data:
            prompt_parts.append(self._render_user_input_section(input_data))
        
        # Format structure
        prompt_parts.append(self._render_format_structure())
        
        # References
        if self.reference.slipbox_links or self.reference.internal_docs:
            prompt_parts.append(self._render_references())
        
        return "\n\n".join(prompt_parts)
    
    def _render_user_input_section(self, input_data: Dict[str, Any]) -> str:
        """Render user input section with schema and examples."""
        sections = []
        
        # Data schema section
        if self.user_input.data_schema:
            sections.append("## Input Data Schema")
            for field_name, field_schema in self.user_input.data_schema.items():
                sections.append(f"- **{field_name}** ({field_schema.field_type}): {field_schema.description}")
                if field_schema.example_values:
                    sections.append(f"  Examples: {', '.join(field_schema.example_values[:3])}")
        
        # Current input data
        sections.append("## Current Input Data")
        for key, value in input_data.items():
            if key in self.user_input.data_schema:
                field_schema = self.user_input.data_schema[key]
                sections.append(f"- **{key}**: {value}")
            else:
                sections.append(f"- **{key}**: {value}")
        
        # Data examples (if configured)
        if self.user_input.data_examples and self.user_input.max_examples > 0:
            sections.append("## Example Data Patterns")
            for i, example in enumerate(self.user_input.data_examples[:self.user_input.max_examples]):
                sections.append(f"### Example {i+1}")
                for key, value in example.items():
                    sections.append(f"- {key}: {value}")
        
        return "\n".join(sections)
    
    def auto_generate_schema_from_dataframe(self, df: 'pd.DataFrame') -> 'BaseTemplate':
        """Auto-generate data schema from DataFrame."""
        import pandas as pd
        
        data_schema = {}
        data_examples = []
        
        # Generate schema from DataFrame columns
        for column in df.columns:
            dtype = str(df[column].dtype)
            
            # Map pandas dtypes to human-readable types
            if 'int' in dtype:
                field_type = 'integer'
            elif 'float' in dtype:
                field_type = 'float'
            elif 'datetime' in dtype:
                field_type = 'datetime'
            elif 'bool' in dtype:
                field_type = 'boolean'
            else:
                field_type = 'string'
            
            # Get example values (non-null, unique)
            example_values = df[column].dropna().unique()[:5]
            example_values = [str(val) for val in example_values]
            
            # Create field schema
            data_schema[column] = DataFieldSchema(
                field_name=column,
                field_type=field_type,
                description=f"Data field: {column}",
                is_required=df[column].notna().all(),
                example_values=example_values,
                validation_rules={}
            )
        
        # Generate example data (anonymized if configured)
        sample_size = min(self.user_input.max_examples, len(df))
        if sample_size > 0:
            sample_df = df.sample(n=sample_size)
            for _, row in sample_df.iterrows():
                example = {}
                for column in df.columns:
                    value = row[column]
                    if pd.isna(value):
                        example[column] = "null"
                    elif self.user_input.anonymize_examples and self._is_sensitive_field(column):
                        example[column] = self._anonymize_value(value, data_schema[column].field_type)
                    else:
                        example[column] = str(value)
                data_examples.append(example)
        
        # Update template with generated schema
        updated_template = self.model_copy()
        updated_template.user_input.data_schema = data_schema
        updated_template.user_input.data_examples = data_examples
        updated_template.user_input.required_fields = [
            field for field, schema in data_schema.items() if schema.is_required
        ]
        updated_template.user_input.optional_fields = [
            field for field, schema in data_schema.items() if not schema.is_required
        ]
        
        return updated_template
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if field contains sensitive data."""
        sensitive_keywords = ['id', 'email', 'phone', 'address', 'name', 'ssn', 'credit']
        return any(keyword in field_name.lower() for keyword in sensitive_keywords)
    
    def _anonymize_value(self, value: Any, field_type: str) -> str:
        """Anonymize sensitive values."""
        if field_type == 'string':
            return f"[REDACTED_{field_type.upper()}]"
        elif field_type == 'integer':
            return "[REDACTED_NUMBER]"
        elif field_type == 'float':
            return "[REDACTED_DECIMAL]"
        else:
            return f"[REDACTED_{field_type.upper()}]"
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate required inputs are present and match schema."""
        # Check required fields
        for field in self.user_input.required_fields:
            if field not in inputs:
                return False
        
        # Type validation if enabled
        if self.user_input.input_validation.get("type_checking", False):
            for field, value in inputs.items():
                if field in self.user_input.data_schema:
                    expected_type = self.user_input.data_schema[field].field_type
                    if not self._validate_field_type(value, expected_type):
                        return False
        
        return True
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type matches expected type."""
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'integer':
            return isinstance(value, int)
        elif expected_type == 'float':
            return isinstance(value, (int, float))
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'datetime':
            return isinstance(value, (str, pd.Timestamp)) if 'pd' in globals() else isinstance(value, str)
        else:
            return True  # Allow any type for unknown types
```

**SystemPrompt**
```python
class SystemPrompt(BaseModel):
    persona: str
    role: str
    context: Optional[str] = None
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )
    
    def render(self) -> str:
        return f"{self.persona}\n\n{self.role}"
```

**Supporting Template Models**
```python
class CategoryDefinition(BaseModel):
    name: str = Field(description="Category name")
    description: str = Field(description="Category description")
    criteria: List[str] = Field(description="Classification criteria")
    examples: List[str] = Field(default_factory=list, description="Example cases")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class Definition(BaseModel):
    categories: List[CategoryDefinition] = Field(default_factory=list)
    key_points: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    domain_knowledge: str = Field(default="", description="Background information")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class Instruction(BaseModel):
    rules: List[str] = Field(description="Processing rules")
    exceptions: List[str] = Field(default_factory=list, description="Exception cases")
    guidelines: List[str] = Field(default_factory=list, description="Additional guidelines")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class FormatStructure(BaseModel):
    output_schema: str = Field(description="Pydantic model name for output")
    required_fields: List[str] = Field(description="Required output fields")
    optional_fields: List[str] = Field(default_factory=list, description="Optional output fields")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class DataFieldSchema(BaseModel):
    field_name: str = Field(description="Name of the data field")
    field_type: str = Field(description="Data type (string, int, float, datetime, etc.)")
    description: str = Field(description="Human-readable description of the field")
    is_required: bool = Field(default=True, description="Whether field is required")
    example_values: List[str] = Field(default_factory=list, description="Example values for this field")
    validation_rules: Dict[str, Any] = Field(default_factory=dict, description="Validation constraints")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class UserInputConfig(BaseModel):
    data_schema: Dict[str, DataFieldSchema] = Field(default_factory=dict, description="Auto-generated data schema")
    auto_generate_schema: bool = Field(default=True, description="Auto-generate schema from DataFrame")
    required_fields: List[str] = Field(default_factory=list, description="Required input fields")
    optional_fields: List[str] = Field(default_factory=list, description="Optional input fields")
    data_examples: List[Dict[str, Any]] = Field(default_factory=list, description="Sample data examples")
    max_examples: int = Field(default=3, description="Maximum examples to include in prompt")
    anonymize_examples: bool = Field(default=True, description="Anonymize sensitive data")
    input_validation: Dict[str, bool] = Field(
        default_factory=lambda: {
            "type_checking": True,
            "range_validation": True,
            "format_validation": True
        },
        description="Input validation settings"
    )
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class Reference(BaseModel):
    slipbox_links: List[str] = Field(default_factory=list, description="Internal slipbox references")
    internal_docs: List[str] = Field(default_factory=list, description="Internal documentation")
    external_resources: List[str] = Field(default_factory=list, description="External resources")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )
```

#### 3. Agent Implementations

**ClassificationAgent**
```python
class ClassificationAgent(BaseAgent):
    def __init__(self, model_id: str, template: ClassificationTemplate):
        super().__init__(model_id, template)
        self.decision_tree = DecisionTree(template.categories)
    
    async def process_single(self, input_data: Dict[str, Any]) -> ClassificationResult:
        prompt = self.prompt_manager.render_prompt(self.template, **input_data)
        response = await self.bedrock_client.invoke_model(prompt)
        return self.response_parser.parse_classification(response)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        required_fields = self.template.get_required_fields()
        return all(field in input_data for field in required_fields)
```

#### 4. Batch Processing

**BatchProcessor**
```python
class BatchProcessor:
    def __init__(self, agent: BaseAgent, batch_size: int = 10):
        self.agent = agent
        self.batch_size = batch_size
        self.concurrent_processor = ConcurrentProcessor()
        self.result_aggregator = ResultAggregator()
    
    async def process_dataframe(
        self, 
        df: pd.DataFrame, 
        max_workers: int = 5
    ) -> pd.DataFrame:
        batches = self._create_batches(df)
        results = await self.concurrent_processor.process_batches(
            batches, self.agent, max_workers
        )
        return self.result_aggregator.combine_results(results, df)
    
    def _create_batches(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        # Split DataFrame into batches
        pass
```

**ConcurrentProcessor**
```python
class ConcurrentProcessor:
    async def process_batches(
        self, 
        batches: List[pd.DataFrame], 
        agent: BaseAgent, 
        max_workers: int
    ) -> List[List[BaseModel]]:
        semaphore = asyncio.Semaphore(max_workers)
        tasks = [
            self._process_batch_with_semaphore(batch, agent, semaphore)
            for batch in batches
        ]
        return await asyncio.gather(*tasks)
    
    async def _process_batch_with_semaphore(
        self, 
        batch: pd.DataFrame, 
        agent: BaseAgent, 
        semaphore: asyncio.Semaphore
    ) -> List[BaseModel]:
        async with semaphore:
            return await self._process_batch(batch, agent)
```

### Model System

#### Base Models
```python
class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    PLANNING = "planning"
    CATALOGING = "cataloging"
    VALIDATION = "validation"

class BaseResult(BaseModel):
    task_type: TaskType
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in the result")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    model_id: str = Field(description="Bedrock model used")
    template_name: str = Field(description="Template used for processing")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class BaseEvidence(BaseModel):
    source: str = Field(description="Source of evidence")
    content: List[str] = Field(description="Evidence content items")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in evidence")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class BaseReasoning(BaseModel):
    primary_factors: List[str] = Field(description="Main reasoning factors")
    supporting_evidence: List[str] = Field(description="Supporting evidence")
    contradicting_evidence: List[str] = Field(default_factory=list, description="Contradicting evidence")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )
```

#### Task-Specific Models
```python
class ClassificationResult(BaseResult):
    category: str
    evidence: BaseEvidence
    reasoning: BaseReasoning
    decision_path: List[str] = Field(description="Decision tree path for classification")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class SummarizationResult(BaseResult):
    summary: str = Field(description="Generated summary text")
    key_points: List[str] = Field(description="Extracted key points")
    extracted_entities: Dict[str, List[str]] = Field(description="Named entities by type")
    source_references: List[str] = Field(description="Source document references")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class PlanStep(BaseModel):
    step_number: int
    description: str
    estimated_duration: str
    dependencies: List[str] = Field(default_factory=list)
    resources: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class RiskItem(BaseModel):
    risk_type: str
    description: str
    probability: float = Field(ge=0.0, le=1.0)
    impact: str
    mitigation: str
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class PlanningResult(BaseResult):
    plan_steps: List[PlanStep] = Field(description="Ordered list of plan steps")
    timeline: Dict[str, str] = Field(description="Timeline mapping")
    resources_required: List[str] = Field(description="Required resources")
    risk_assessment: List[RiskItem] = Field(description="Identified risks")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class ValidationIssue(BaseModel):
    issue_type: str
    severity: str
    description: str
    location: Optional[str] = None
    recommendation: str
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class ValidationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PARTIAL = "partial"

class ValidationResult(BaseResult):
    validation_status: ValidationStatus
    issues_found: List[ValidationIssue] = Field(description="Validation issues")
    recommendations: List[str] = Field(description="Improvement recommendations")
    compliance_score: float = Field(ge=0.0, le=1.0, description="Overall compliance score")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class CatalogEntry(BaseModel):
    entry_id: str = Field(description="Unique identifier for catalog entry")
    title: str = Field(description="Entry title")
    description: str = Field(description="Entry description")
    category: str = Field(description="Primary category")
    subcategories: List[str] = Field(default_factory=list, description="Subcategories")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

class CatalogResult(BaseResult):
    catalog_entries: List[CatalogEntry] = Field(description="Generated catalog entries")
    taxonomy: Dict[str, List[str]] = Field(description="Hierarchical taxonomy")
    index_structure: Dict[str, Any] = Field(description="Index organization")
    metadata_schema: Dict[str, str] = Field(description="Metadata field definitions")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )
```

## Multi-Agent Workflow System

### LangGraph Integration

The framework integrates with LangGraph to enable complex multi-agent workflows that combine different task types for comprehensive processing.

#### Workflow Components

**BaseWorkflow**
```python
class BaseWorkflow(BaseModel):
    workflow_id: str = Field(description="Unique workflow identifier")
    name: str = Field(description="Workflow name")
    description: str = Field(description="Workflow description")
    agents: List[BaseAgent] = Field(description="Agents in the workflow")
    graph: StateGraph = Field(description="LangGraph state graph")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Execute the workflow graph
        pass
```

**WorkflowState**
```python
class WorkflowState(BaseModel):
    input_data: Dict[str, Any] = Field(description="Original input data")
    current_step: str = Field(description="Current workflow step")
    agent_results: Dict[str, BaseResult] = Field(default_factory=dict)
    intermediate_data: Dict[str, Any] = Field(default_factory=dict)
    final_result: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )
```

#### Pre-built Workflow Templates

**1. Classification + Validation Workflow**
```python
class ClassificationValidationWorkflow(BaseWorkflow):
    """
    Workflow that first classifies content, then validates the classification.
    Use case: RnR classification with quality assurance
    """
    
    def __init__(self, classification_agent: ClassificationAgent, validation_agent: ValidationAgent):
        # Build LangGraph workflow
        graph = StateGraph(WorkflowState)
        
        # Add nodes
        graph.add_node("classify", self._classify_node)
        graph.add_node("validate", self._validate_node)
        graph.add_node("finalize", self._finalize_node)
        
        # Add edges
        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            self._should_validate,
            {"validate": "validate", "finalize": "finalize"}
        )
        graph.add_edge("validate", "finalize")
        graph.add_edge("finalize", END)
        
        super().__init__(
            workflow_id="classification_validation",
            name="Classification with Validation",
            description="Classify content and validate the results",
            agents=[classification_agent, validation_agent],
            graph=graph.compile()
        )
    
    async def _classify_node(self, state: WorkflowState) -> WorkflowState:
        classification_result = await self.agents[0].process_single(state.input_data)
        state.agent_results["classification"] = classification_result
        state.current_step = "classification_complete"
        return state
    
    async def _validate_node(self, state: WorkflowState) -> WorkflowState:
        # Prepare validation input with classification result
        validation_input = {
            **state.input_data,
            "classification_result": state.agent_results["classification"]
        }
        validation_result = await self.agents[1].process_single(validation_input)
        state.agent_results["validation"] = validation_result
        state.current_step = "validation_complete"
        return state
    
    def _should_validate(self, state: WorkflowState) -> str:
        classification_result = state.agent_results.get("classification")
        if classification_result and classification_result.confidence_score < 0.8:
            return "validate"
        return "finalize"
    
    async def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        # Combine results
        final_result = {
            "classification": state.agent_results.get("classification"),
            "validation": state.agent_results.get("validation"),
            "workflow_confidence": self._calculate_workflow_confidence(state)
        }
        state.final_result = final_result
        state.current_step = "complete"
        return state
```

**2. Summarization + Cataloging Workflow**
```python
class SummarizationCatalogWorkflow(BaseWorkflow):
    """
    Workflow that summarizes content and then catalogs it.
    Use case: Document processing and knowledge management
    """
    
    def __init__(self, summarization_agent: SummarizationAgent, catalog_agent: CatalogAgent):
        graph = StateGraph(WorkflowState)
        
        # Add nodes
        graph.add_node("summarize", self._summarize_node)
        graph.add_node("catalog", self._catalog_node)
        graph.add_node("finalize", self._finalize_node)
        
        # Add edges
        graph.add_edge(START, "summarize")
        graph.add_edge("summarize", "catalog")
        graph.add_edge("catalog", "finalize")
        graph.add_edge("finalize", END)
        
        super().__init__(
            workflow_id="summarization_catalog",
            name="Summarization with Cataloging",
            description="Summarize content and create catalog entries",
            agents=[summarization_agent, catalog_agent],
            graph=graph.compile()
        )
    
    async def _summarize_node(self, state: WorkflowState) -> WorkflowState:
        summary_result = await self.agents[0].process_single(state.input_data)
        state.agent_results["summarization"] = summary_result
        state.current_step = "summarization_complete"
        return state
    
    async def _catalog_node(self, state: WorkflowState) -> WorkflowState:
        # Use summary for cataloging
        catalog_input = {
            **state.input_data,
            "summary": state.agent_results["summarization"].summary,
            "key_points": state.agent_results["summarization"].key_points
        }
        catalog_result = await self.agents[1].process_single(catalog_input)
        state.agent_results["catalog"] = catalog_result
        state.current_step = "catalog_complete"
        return state
```

**3. Planning + Validation Workflow**
```python
class PlanningValidationWorkflow(BaseWorkflow):
    """
    Workflow that creates plans and validates their feasibility.
    Use case: Project planning with risk assessment
    """
    
    def __init__(self, planning_agent: PlanningAgent, validation_agent: ValidationAgent):
        graph = StateGraph(WorkflowState)
        
        # Add nodes with conditional loops
        graph.add_node("plan", self._plan_node)
        graph.add_node("validate_plan", self._validate_plan_node)
        graph.add_node("refine_plan", self._refine_plan_node)
        graph.add_node("finalize", self._finalize_node)
        
        # Add edges with conditional routing
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "validate_plan")
        graph.add_conditional_edges(
            "validate_plan",
            self._should_refine_plan,
            {"refine": "refine_plan", "finalize": "finalize"}
        )
        graph.add_edge("refine_plan", "validate_plan")  # Loop back for re-validation
        graph.add_edge("finalize", END)
        
        super().__init__(
            workflow_id="planning_validation",
            name="Planning with Validation",
            description="Create and validate project plans",
            agents=[planning_agent, validation_agent],
            graph=graph.compile()
        )
```

#### Custom Workflow Builder

**Dynamic Workflow Creation**
```python
class CustomWorkflowBuilder:
    """
    Builder for creating custom multi-agent workflows dynamically.
    """
    
    def __init__(self):
        self.agents: List[BaseAgent] = []
        self.nodes: List[Tuple[str, Callable]] = []
        self.edges: List[Tuple[str, str]] = []
        self.conditional_edges: List[Tuple[str, Callable, Dict[str, str]]] = []
    
    def add_agent(self, agent: BaseAgent, node_name: str) -> 'CustomWorkflowBuilder':
        """Add an agent as a workflow node."""
        self.agents.append(agent)
        self.nodes.append((node_name, self._create_agent_node(agent)))
        return self
    
    def add_edge(self, from_node: str, to_node: str) -> 'CustomWorkflowBuilder':
        """Add a direct edge between nodes."""
        self.edges.append((from_node, to_node))
        return self
    
    def add_conditional_edge(
        self, 
        from_node: str, 
        condition_func: Callable, 
        edge_map: Dict[str, str]
    ) -> 'CustomWorkflowBuilder':
        """Add a conditional edge with routing logic."""
        self.conditional_edges.append((from_node, condition_func, edge_map))
        return self
    
    def build(self, workflow_id: str, name: str, description: str) -> BaseWorkflow:
        """Build the complete workflow."""
        graph = StateGraph(WorkflowState)
        
        # Add all nodes
        for node_name, node_func in self.nodes:
            graph.add_node(node_name, node_func)
        
        # Add all edges
        for from_node, to_node in self.edges:
            graph.add_edge(from_node, to_node)
        
        # Add conditional edges
        for from_node, condition_func, edge_map in self.conditional_edges:
            graph.add_conditional_edges(from_node, condition_func, edge_map)
        
        return BaseWorkflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            agents=self.agents,
            graph=graph.compile()
        )
    
    def _create_agent_node(self, agent: BaseAgent) -> Callable:
        """Create a node function for an agent."""
        async def agent_node(state: WorkflowState) -> WorkflowState:
            try:
                result = await agent.process_single(state.input_data)
                state.agent_results[agent.__class__.__name__.lower()] = result
                return state
            except Exception as e:
                state.errors.append(f"Agent {agent.__class__.__name__} failed: {str(e)}")
                return state
        return agent_node
```

### Multi-Agent Workflow Usage Examples

#### Example 1: RnR Classification with Validation
```python
from nlp_pipeline.agents import ClassificationAgent, ValidationAgent
from nlp_pipeline.workflows import ClassificationValidationWorkflow
from nlp_pipeline.templates import ClassificationTemplate, ValidationTemplate

# Create agents
classification_template = ClassificationTemplate.from_config("config/templates/rnr_classification.yaml")
validation_template = ValidationTemplate.from_config("config/templates/rnr_validation.yaml")

classification_agent = ClassificationAgent(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    template=classification_template
)

validation_agent = ValidationAgent(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    template=validation_template
)

# Create workflow
workflow = ClassificationValidationWorkflow(classification_agent, validation_agent)

# Execute workflow
input_data = {
    "dialogue": "Customer claims package never arrived...",
    "shiptrack": "EVENT_301: Delivered to front door at 2:30 PM",
    "max_estimated_arrival_date": "2025-01-20"
}

result = await workflow.execute(input_data)
print(f"Classification: {result['classification'].category}")
print(f"Validation Status: {result['validation'].validation_status}")
print(f"Workflow Confidence: {result['workflow_confidence']}")
```

#### Example 2: Document Processing Pipeline
```python
from nlp_pipeline.agents import SummarizationAgent, CatalogAgent
from nlp_pipeline.workflows import SummarizationCatalogWorkflow

# Create agents
summarization_agent = SummarizationAgent(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    template=summarization_template
)

catalog_agent = CatalogAgent(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    template=catalog_template
)

# Create workflow
workflow = SummarizationCatalogWorkflow(summarization_agent, catalog_agent)

# Process document
document_data = {
    "content": "Long technical document content...",
    "document_type": "technical_specification",
    "source": "internal_docs/api_spec_v2.md"
}

result = await workflow.execute(document_data)
print(f"Summary: {result['summarization'].summary}")
print(f"Catalog Entries: {len(result['catalog'].catalog_entries)}")
```

#### Example 3: Custom Multi-Agent Workflow
```python
from nlp_pipeline.workflows import CustomWorkflowBuilder
from nlp_pipeline.agents import ClassificationAgent, SummarizationAgent, ValidationAgent

# Create custom workflow with multiple agents
builder = CustomWorkflowBuilder()

# Add agents as nodes
builder.add_agent(classification_agent, "classify")
builder.add_agent(summarization_agent, "summarize")
builder.add_agent(validation_agent, "validate")

# Define workflow logic
builder.add_edge("classify", "summarize")
builder.add_conditional_edge(
    "summarize",
    lambda state: "validate" if state.agent_results["summarization"].confidence_score > 0.7 else "end",
    {"validate": "validate", "end": "end"}
)

# Build workflow
custom_workflow = builder.build(
    workflow_id="custom_processing",
    name="Custom Processing Pipeline",
    description="Classification, summarization, and validation pipeline"
)

# Execute
result = await custom_workflow.execute(input_data)
```

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1-2)
1. **Create base components**
   - BaseAgent abstract class
   - BedrockClient with inference profile support
   - PromptManager for template handling
   - ResponseParser for generic parsing

2. **Implement template system**
   - BaseTemplate and task-specific templates
   - Template validation and rendering
   - Prompt component structure

3. **Set up model system**
   - Base Pydantic models
   - Task-specific response models
   - Validation and serialization

### Phase 2: Single Agent Implementation (Week 3-4)
1. **Classification Agent**
   - Migrate RnR logic to generic classification
   - Implement decision tree support
   - Add confidence scoring

2. **Batch Processing**
   - Generic BatchProcessor
   - Async ConcurrentProcessor
   - Result aggregation and formatting

3. **Testing Framework**
   - Unit tests for all components
   - Integration tests with Bedrock
   - Performance benchmarks

### Phase 3: Multi-Agent Workflows (Week 5-6)
1. **LangGraph Integration**
   - BaseWorkflow implementation
   - WorkflowState management
   - Graph builder utilities

2. **Pre-built Workflows**
   - Classification + Validation workflow
   - Summarization + Cataloging workflow
   - Planning + Validation workflow

3. **Custom Workflow Builder**
   - Dynamic workflow creation
   - Conditional routing logic
   - Error handling and recovery

### Phase 4: Additional Task Types (Week 7-8)
1. **Remaining Agents**
   - Summarization Agent
   - Planning Agent
   - Catalog Agent
   - Validation Agent

2. **Advanced Workflows**
   - Complex multi-step processes
   - Iterative refinement workflows
   - Parallel processing workflows

### Phase 5: Production Features (Week 9-10)
1. **Advanced Features**
   - Checkpoint/resume functionality
   - Progress tracking and monitoring
   - Error recovery and retry logic

2. **Integration**
   - SageMaker compatibility
   - S3 integration for large datasets
   - CloudWatch logging

3. **Legacy Support**
   - Backward compatibility layer
   - Migration utilities
   - Documentation and examples

## Configuration Management

### Template Configuration
```yaml
# config/templates/rnr_classification.yaml
name: "rnr_classification"
task_type: "classification"
system_prompt:
  persona: "You are an expert in analyzing buyer-seller messaging conversations"
  role: "Your task is to classify interactions into predefined categories"
definition:
  categories:
    - name: "TrueDNR"
      description: "Package marked as delivered but buyer claims non-receipt"
      criteria: ["Tracking shows delivery", "Buyer disputes receiving"]
instruction:
  rules:
    - "Choose exactly ONE category from the provided list"
    - "Provide confidence score between 0.00 and 1.00"
  exceptions:
    - "Do not classify without sufficient evidence"
format_structure:
  output_schema: "ClassificationResult"
  required_fields: ["category", "confidence_score", "evidence", "reasoning"]
reference:
  slipbox_links:
    - "slipbox/01_design/rnr_classification_guide.md"
```

### Agent Configuration
```yaml
# config/agents/rnr_agent.yaml
agent_type: "classification"
model_id: "anthropic.claude-3-5-sonnet-20241022-v2:0"
template: "rnr_classification"
batch_size: 10
max_workers: 5
retry_config:
  max_retries: 3
  backoff_factor: 2
  max_delay: 32
inference_profile:
  auto_detect: true
  fallback_model: "anthropic.claude-3-5-sonnet-20240620-v1:0"
```

## Usage Examples

### Basic Classification
```python
from nlp_pipeline.agents import ClassificationAgent
from nlp_pipeline.templates import ClassificationTemplate

# Load template
template = ClassificationTemplate.from_config("config/templates/rnr_classification.yaml")

# Create agent
agent = ClassificationAgent(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    template=template
)

# Process single item
result = await agent.process_single({
    "dialogue": "Customer says they never received the package...",
    "shiptrack": "EVENT_301: Delivered to front door",
    "max_estimated_arrival_date": "2025-01-20"
})

print(f"Category: {result.category}")
print(f"Confidence: {result.confidence_score}")
```

### Batch Processing
```python
from nlp_pipeline.processors import BatchProcessor
import pandas as pd

# Load data
df = pd.read_csv("rnr_cases.csv")

# Create batch processor
processor = BatchProcessor(agent, batch_size=20)

# Process all data
results_df = await processor.process_dataframe(df, max_workers=10)

# Save results
results_df.to_parquet("rnr_results.parquet")
```

### Auto-Generated Schema from DataFrame
```python
import pandas as pd
from nlp_pipeline.templates import ClassificationTemplate
from nlp_pipeline.agents import ClassificationAgent

# Load your data
df = pd.read_csv("customer_data.csv")
# DataFrame columns: ['dialogue', 'shiptrack', 'customer_id', 'order_date', 'delivery_status']

# Load base template
base_template = ClassificationTemplate.from_config("config/templates/rnr_classification.yaml")

# Auto-generate schema from DataFrame
template_with_schema = base_template.auto_generate_schema_from_dataframe(df)

# The template now includes:
# - Auto-detected field types (string, integer, datetime, etc.)
# - Example values for each field
# - Required/optional field classification
# - Anonymized examples for sensitive fields

# Create agent with schema-aware template
agent = ClassificationAgent(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    template=template_with_schema
)

# Process data - the prompt will include schema information
result = await agent.process_single({
    "dialogue": "Customer says package never arrived",
    "shiptrack": "EVENT_301: Delivered to front door",
    "customer_id": "CUST_12345",
    "order_date": "2025-01-15",
    "delivery_status": "delivered"
})

# The generated prompt will include:
# ## Input Data Schema
# - **dialogue** (string): Data field: dialogue
#   Examples: Customer says package never arrived, Item was damaged, Great service
# - **shiptrack** (string): Data field: shiptrack  
#   Examples: EVENT_301: Delivered, EVENT_200: In transit, EVENT_100: Shipped
# - **customer_id** (string): Data field: customer_id
#   Examples: [REDACTED_STRING], [REDACTED_STRING], [REDACTED_STRING]
# - **order_date** (datetime): Data field: order_date
#   Examples: 2025-01-15, 2025-01-14, 2025-01-13
# - **delivery_status** (string): Data field: delivery_status
#   Examples: delivered, in_transit, pending
#
# ## Current Input Data
# - **dialogue**: Customer says package never arrived
# - **shiptrack**: EVENT_301: Delivered to front door
# - **customer_id**: CUST_12345
# - **order_date**: 2025-01-15
# - **delivery_status**: delivered
```

### Custom Template with Manual Schema
```python
from nlp_pipeline.templates import BaseTemplate, SystemPrompt, Definition, CategoryDefinition
from nlp_pipeline.templates import UserInputConfig, DataFieldSchema
from nlp_pipeline.models import TaskType

# Create custom template with manual schema definition
template = BaseTemplate(
    name="custom_classification",
    task_type=TaskType.CLASSIFICATION,
    system_prompt=SystemPrompt(
        persona="You are a sentiment analysis expert",
        role="Classify text sentiment as positive, negative, or neutral"
    ),
    definition=Definition(
        categories=[
            CategoryDefinition(name="positive", description="Positive sentiment"),
            CategoryDefinition(name="negative", description="Negative sentiment"),
            CategoryDefinition(name="neutral", description="Neutral sentiment")
        ]
    ),
    user_input=UserInputConfig(
        data_schema={
            "text": DataFieldSchema(
                field_name="text",
                field_type="string",
                description="Text content to analyze for sentiment",
                is_required=True,
                example_values=["Great product!", "Terrible service", "It's okay"],
                validation_rules={"min_length": 1, "max_length": 1000}
            ),
            "source": DataFieldSchema(
                field_name="source",
                field_type="string", 
                description="Source of the text (review, comment, etc.)",
                is_required=False,
                example_values=["product_review", "customer_feedback", "social_media"]
            )
        },
        required_fields=["text"],
        optional_fields=["source"],
        max_examples=2,
        anonymize_examples=False
    ),
    # ... other components
)

# Use with agent
from nlp_pipeline.agents import ClassificationAgent
agent = ClassificationAgent(model_id="claude-3-sonnet", template=template)
```

### Batch Processing with Auto-Schema
```python
import pandas as pd
from nlp_pipeline.processors import BatchProcessor
from nlp_pipeline.templates import ClassificationTemplate

# Load large dataset
df = pd.read_csv("large_customer_dataset.csv")
print(f"Processing {len(df)} records")

# Load and auto-configure template
base_template = ClassificationTemplate.from_config("config/templates/rnr_classification.yaml")

# Auto-generate schema from the DataFrame
template_with_schema = base_template.auto_generate_schema_from_dataframe(df)

# Create agent with schema-aware template
agent = ClassificationAgent(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    template=template_with_schema
)

# Create batch processor
processor = BatchProcessor(agent, batch_size=50)

# Process all data with schema-aware prompts
results_df = await processor.process_dataframe(df, max_workers=10)

# Each batch will include:
# 1. Auto-detected schema information in the prompt
# 2. Example data patterns from the DataFrame
# 3. Proper field type validation
# 4. Anonymized sensitive fields

print(f"Processed {len(results_df)} records")
print(f"Categories found: {results_df['category'].value_counts()}")
```

## Migration Strategy

### Backward Compatibility
1. **Legacy Wrapper**
   ```python
   # legacy_bedrock/rnr_agent.py
   from nlp_pipeline.agents import ClassificationAgent
   from nlp_pipeline.templates import ClassificationTemplate
   import asyncio
   
   class BedrockRnRAgent:
       def __init__(self, **kwargs):
           # Wrap new ClassificationAgent
           template = ClassificationTemplate.load_rnr_template()
           self._agent = ClassificationAgent(template=template, **kwargs)
       
       def analyze_rnr_case_sync(self, dialogue, shiptrack, max_date=None):
           # Convert to new format and delegate
           return asyncio.run(self._agent.process_single({
               "dialogue": dialogue,
               "shiptrack": shiptrack,
               "max_estimated_arrival_date": max_date
           }))
   ```

2. **Migration Utilities**
   ```python
   def migrate_rnr_config_to_template(old_config: dict) -> ClassificationTemplate:
       # Convert old configuration to new template format
       pass
   
   def migrate_batch_processing(old_processor, new_processor):
       # Migrate existing batch processing workflows
       pass
   ```

## Testing Strategy

### Unit Tests
- Template loading and validation
- Prompt rendering with various inputs
- Response parsing for all model types
- Agent initialization and configuration

### Integration Tests
- End-to-end processing with Bedrock
- Batch processing with concurrent execution
- Error handling and recovery
- Performance benchmarks

### Validation Tests
- Template structure validation
- Model schema validation
- Configuration validation
- Backward compatibility validation

## Performance Considerations

### Optimization Strategies
1. **Connection Pooling**: Reuse Bedrock clients across requests
2. **Async Processing**: Use asyncio for concurrent API calls
3. **Caching**: Cache templates and parsed responses
4. **Batching**: Optimize batch sizes for throughput vs latency
5. **Resource Management**: Proper semaphore usage for rate limiting

### Monitoring
- Request latency tracking
- Error rate monitoring
- Throughput metrics
- Resource utilization
- Cost tracking per model/task type

## Security Considerations

### Access Control
- IAM roles for Bedrock access
- Secure credential management
- Environment-specific configurations

### Data Protection
- Input data sanitization
- Response data validation
- Audit logging for compliance

## Conclusion

This design provides a comprehensive framework for refactoring the bedrock scripts into a generic, modular system that supports multiple LLM tasks while maintaining backward compatibility. The architecture emphasizes modularity, reusability, and production readiness while providing a clear migration path from the existing RnR-specific implementation.

The framework will enable rapid development of new LLM-powered features while maintaining consistency in prompt engineering, response handling, and batch processing across all task types.
