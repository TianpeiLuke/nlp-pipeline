# Config

## What is the Purpose of Config?

Configs provide **centralized, structured, and validated configuration management** for step builders. They serve as the single source of truth for all step-specific settings, supporting hierarchical configuration from global to step-specific levels.

## Core Purpose

Configs provide the **configuration management layer** that:

1. **Centralized Configuration Management** - Structured, validated configuration for step builders
2. **Hierarchical Configuration Structure** - Layered configuration from global to step-specific
3. **Configuration Validation** - Ensure configuration correctness with clear error messages
4. **Environment-Specific Configuration** - Support different environments (dev, staging, prod)
5. **Configuration Templating** - Reusable configuration patterns and parameterization

## Key Features

### 1. Centralized Configuration Management

Configs provide structured, validated configuration for step builders:

```python
@dataclass
class XGBoostTrainingStepConfig(ConfigBase):
    """Configuration for XGBoost training step"""
    
    # Core training configuration
    entry_point: str = "train.py"
    framework_version: str = "1.5-1"
    instance_type: str = "ml.m5.xlarge"
    instance_count: int = 1
    
    # Hyperparameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Resource configuration
    max_runtime_in_seconds: int = 7200  # 2 hours
    volume_size_in_gb: int = 30
    
    # Optional configurations
    retry_policies: Optional[List[RetryPolicy]] = None
    cache_config: Optional[CacheConfig] = None
```

### 2. Hierarchical Configuration Structure

Support layered configuration from global to step-specific:

```python
# Global shared configuration
shared_config = {
    "role": "arn:aws:iam::123456789012:role/SageMakerRole",
    "region": "us-west-2",
    "bucket": "ml-pipeline-artifacts"
}

# Processing-specific shared configuration
processing_shared = {
    "instance_type": "ml.m5.large",
    "volume_size_in_gb": 20
}

# Step-specific configuration
xgboost_specific = {
    "framework_version": "1.5-1",
    "hyperparameters": {
        "max_depth": 6,
        "eta": 0.3,
        "objective": "binary:logistic"
    }
}

# Final configuration merges all layers
final_config = merge_configs(shared_config, processing_shared, xgboost_specific)
```

### 3. Configuration Validation

Implement validation rules to ensure correctness:

```python
def validate_configuration(self):
    """Validate configuration values"""
    errors = []
    
    # Validate instance type
    if not self.instance_type.startswith("ml."):
        errors.append("instance_type must be a valid SageMaker instance type")
    
    # Validate hyperparameters
    if "max_depth" in self.hyperparameters:
        if not isinstance(self.hyperparameters["max_depth"], int):
            errors.append("max_depth must be an integer")
        if self.hyperparameters["max_depth"] < 1:
            errors.append("max_depth must be positive")
    
    # Validate resource constraints
    if self.max_runtime_in_seconds < 300:  # 5 minutes minimum
        errors.append("max_runtime_in_seconds must be at least 300 seconds")
    
    return errors
```

### 4. Environment-Specific Configuration

Support environment-specific overrides:

```python
@dataclass
class XGBoostTrainingStepConfig(ConfigBase):
    # Base configuration
    instance_type: str = "ml.m5.xlarge"
    
    def apply_environment_overrides(self, environment: str):
        """Apply environment-specific configuration"""
        if environment == "development":
            self.instance_type = "ml.t3.medium"  # Cheaper for dev
            self.max_runtime_in_seconds = 1800   # Shorter timeout
            self.hyperparameters["n_estimators"] = 10  # Faster training
        elif environment == "staging":
            self.instance_type = "ml.m5.large"   # Medium resources
            self.max_runtime_in_seconds = 3600   # Medium timeout
        elif environment == "production":
            self.instance_type = "ml.m5.2xlarge"  # More powerful
            self.max_runtime_in_seconds = 14400   # Longer timeout
            # Use full hyperparameters for production
```

### 5. Configuration Templating

Support templating and parameterization:

```python
@dataclass
class XGBoostTrainingStepConfig(ConfigBase):
    # Template parameters
    model_type: str = "classification"  # or "regression"
    dataset_size: str = "medium"        # "small", "medium", "large"
    
    def __post_init__(self):
        """Apply configuration templates"""
        # Auto-configure based on model type
        if self.model_type == "classification":
            self.hyperparameters.setdefault("objective", "binary:logistic")
            self.hyperparameters.setdefault("eval_metric", "auc")
        elif self.model_type == "regression":
            self.hyperparameters.setdefault("objective", "reg:squarederror")
            self.hyperparameters.setdefault("eval_metric", "rmse")
        
        # Auto-configure based on dataset size
        if self.dataset_size == "large":
            self.instance_type = "ml.m5.2xlarge"
            self.instance_count = 2
            self.volume_size_in_gb = 100
        elif self.dataset_size == "small":
            self.instance_type = "ml.m5.large"
            self.volume_size_in_gb = 20
```

## Configuration Hierarchy

### 1. Global Shared Configuration

Values identical across all configurations:

```python
global_shared = {
    "role": "arn:aws:iam::123456789012:role/SageMakerRole",
    "region": "us-west-2",
    "bucket": "ml-pipeline-artifacts",
    "kms_key": "arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012"
}
```

### 2. Type Shared Configuration

Values shared within node type categories:

```python
# Processing steps shared configuration
processing_shared = {
    "instance_type": "ml.m5.large",
    "volume_size_in_gb": 20,
    "max_runtime_in_seconds": 3600
}

# Training steps shared configuration
training_shared = {
    "instance_type": "ml.m5.xlarge",
    "volume_size_in_gb": 30,
    "max_runtime_in_seconds": 7200
}
```

### 3. Step Specific Configuration

Values unique to individual step instances:

```python
# XGBoost specific configuration
xgboost_specific = {
    "framework_version": "1.5-1",
    "entry_point": "train_xgboost.py",
    "hyperparameters": {
        "max_depth": 6,
        "eta": 0.3,
        "subsample": 0.8
    }
}

# PyTorch specific configuration
pytorch_specific = {
    "framework_version": "1.12.0",
    "py_version": "py38",
    "entry_point": "train_pytorch.py",
    "hyperparameters": {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001
    }
}
```

### 4. Environment Specific Configuration

Values that vary by deployment environment:

```python
environment_configs = {
    "development": {
        "instance_type": "ml.t3.medium",
        "max_runtime_in_seconds": 1800,
        "enable_debugging": True
    },
    "staging": {
        "instance_type": "ml.m5.large",
        "max_runtime_in_seconds": 3600,
        "enable_profiling": True
    },
    "production": {
        "instance_type": "ml.m5.2xlarge",
        "max_runtime_in_seconds": 14400,
        "enable_monitoring": True
    }
}
```

## Integration with Other Components

### With Step Builders

Configs use dependency injection pattern with step builders:

```python
class XGBoostTrainingStepBuilder(BuilderStepBase):
    def __init__(self, config: XGBoostTrainingStepConfig):
        self.config = config  # Injected configuration
        super().__init__()
    
    def _create_estimator_from_config(self):
        """Create estimator based on configuration"""
        return XGBoost(
            entry_point=self.config.entry_point,
            framework_version=self.config.framework_version,
            instance_type=self.config.instance_type,
            hyperparameters=self.config.hyperparameters,
            role=self.config.role
        )
```

### With Smart Proxies

Smart Proxies use configs to create configured step builders:

```python
class XGBoostTrainingProxy:
    def __init__(self, config: XGBoostTrainingStepConfig):
        self.config = config
        self.builder = XGBoostTrainingStepBuilder(config)
    
    def with_hyperparameters(self, **hyperparams):
        """Fluent interface for hyperparameter configuration"""
        self.config.hyperparameters.update(hyperparams)
        return self
```

### Configuration-Driven Behavior

Configs enable behavior changes based on configuration:

```python
def _create_estimator_from_config(self):
    """Create estimator based on configuration"""
    if self.config.use_distributed_training:
        return self._create_distributed_estimator()
    else:
        return self._create_single_instance_estimator()
    
    if self.config.enable_spot_instances:
        # Configure spot instance settings
        pass
```

## Common Configuration Patterns

### 1. Base Configuration Class

```python
@dataclass
class ConfigBase:
    """Base class for all step configurations"""
    
    # Common fields
    role: str = ""
    region: str = "us-west-2"
    bucket: str = ""
    
    def validate_configuration(self):
        """Base validation logic"""
        errors = []
        
        if not self.role:
            errors.append("role is required")
        if not self.bucket:
            errors.append("bucket is required")
        
        return errors
    
    def merge_with(self, other_config: dict):
        """Merge with another configuration"""
        for key, value in other_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
```

### 2. Processing Configuration

```python
@dataclass
class ProcessingStepConfig(ConfigBase):
    """Configuration for processing steps"""
    
    instance_type: str = "ml.m5.large"
    instance_count: int = 1
    volume_size_in_gb: int = 20
    max_runtime_in_seconds: int = 3600
    
    # Processing-specific
    source_dir: str = ""
    dependencies: List[str] = field(default_factory=list)
```

### 3. Training Configuration

```python
@dataclass
class TrainingStepConfig(ConfigBase):
    """Configuration for training steps"""
    
    instance_type: str = "ml.m5.xlarge"
    instance_count: int = 1
    volume_size_in_gb: int = 30
    max_runtime_in_seconds: int = 7200
    
    # Training-specific
    framework: str = ""
    framework_version: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metric_definitions: List[Dict] = field(default_factory=list)
```

## Strategic Value

Configs provide:

1. **Configuration Management**: Structured, validated configuration with clear hierarchy
2. **Environment Support**: Environment-specific overrides for different deployment stages
3. **Template System**: Reusable configuration patterns and intelligent defaults
4. **Validation Framework**: Ensure configuration correctness with meaningful error messages
5. **Separation of Concerns**: Configuration logic separated from implementation logic
6. **Maintainability**: Centralized configuration makes changes easier to manage

## Example Usage

```python
# Create configuration with templates
config = XGBoostTrainingStepConfig(
    model_type="classification",
    dataset_size="large",
    hyperparameters={
        "max_depth": 8,
        "n_estimators": 100
    }
)

# Apply environment-specific overrides
config.apply_environment_overrides("production")

# Validate configuration
errors = config.validate_configuration()
if errors:
    raise ConfigurationError(f"Invalid configuration: {errors}")

# Use with step builder
builder = XGBoostTrainingStepBuilder(config)
```

Configs form the **configuration foundation** that enables flexible, environment-aware, and validated pipeline construction while maintaining clean separation between configuration and implementation concerns.
