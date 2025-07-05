# Semantic Matcher

## Overview
The Semantic Matcher provides intelligent dependency resolution through semantic similarity algorithms. It calculates similarity between dependency names and output names to enable automatic matching of compatible pipeline components, even when names don't match exactly.

## Core Functionality

### Similarity Calculation
- **Multi-Metric Scoring** - Combines string, token, semantic, and substring similarity
- **Weighted Scoring** - Balances different similarity aspects with configurable weights
- **Normalization** - Standardizes names for consistent comparison
- **Threshold Filtering** - Filters matches based on minimum similarity scores

### Semantic Intelligence
- **Synonym Recognition** - Matches conceptually similar terms
- **Abbreviation Expansion** - Expands common abbreviations for better matching
- **Stop Word Filtering** - Removes noise words that don't contribute to meaning
- **Domain-Specific Vocabulary** - Pipeline-specific synonym dictionaries

## Key Classes

### SemanticMatcher
Main class that provides semantic similarity calculation and matching.

```python
class SemanticMatcher:
    def __init__(self):
        """Initialize the semantic matcher with common patterns."""
        
    def calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate semantic similarity between two names."""
        
    def find_best_matches(self, target_name: str, candidate_names: List[str], 
                         threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Find the best matching names from a list of candidates."""
        
    def explain_similarity(self, name1: str, name2: str) -> Dict[str, float]:
        """Provide detailed explanation of similarity calculation."""
```

## Similarity Metrics

### 1. String Similarity (30% weight)
Uses sequence matching to compare character-level similarity:

```python
def _calculate_string_similarity(self, name1: str, name2: str) -> float:
    """Calculate string similarity using sequence matching."""
    return SequenceMatcher(None, name1, name2).ratio()
```

### 2. Token Similarity (25% weight)
Compares word-level overlap using Jaccard similarity:

```python
def _calculate_token_similarity(self, name1: str, name2: str) -> float:
    """Calculate similarity based on token overlap."""
    tokens1 = set(name1.split())
    tokens2 = set(name2.split())
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union) if union else 0.0
```

### 3. Semantic Similarity (25% weight)
Uses domain-specific synonyms and concept matching:

```python
def _calculate_semantic_similarity(self, name1: str, name2: str) -> float:
    """Calculate semantic similarity using synonym matching."""
    # Matches synonyms like 'model' <-> 'artifact', 'data' <-> 'dataset'
```

### 4. Substring Similarity (20% weight)
Identifies partial matches and common substrings:

```python
def _calculate_substring_similarity(self, name1: str, name2: str) -> float:
    """Calculate similarity based on substring matching."""
    # Handles cases like 'training_data' <-> 'data'
```

## Usage Examples

### Basic Similarity Calculation
```python
from src.pipeline_deps.semantic_matcher import SemanticMatcher

matcher = SemanticMatcher()

# Calculate similarity between names
score = matcher.calculate_similarity("training_data", "processed_dataset")
print(f"Similarity: {score:.3f}")
# Output: Similarity: 0.742

# Exact match
score = matcher.calculate_similarity("model_artifacts", "model_artifacts")
print(f"Exact match: {score:.3f}")
# Output: Exact match: 1.000
```

### Finding Best Matches
```python
# Find best matches from candidates
target = "model_output"
candidates = [
    "trained_model",
    "model_artifacts", 
    "preprocessing_output",
    "evaluation_results",
    "model_package"
]

matches = matcher.find_best_matches(target, candidates, threshold=0.4)
for name, score in matches:
    print(f"{name}: {score:.3f}")

# Output:
# model_artifacts: 0.825
# trained_model: 0.687
# model_package: 0.542
```

### Detailed Similarity Explanation
```python
# Get detailed breakdown of similarity calculation
explanation = matcher.explain_similarity("training_data", "processed_dataset")
print("Similarity breakdown:")
for metric, score in explanation.items():
    if isinstance(score, float):
        print(f"  {metric}: {score:.3f}")
    else:
        print(f"  {metric}: {score}")

# Output:
# Similarity breakdown:
#   overall_score: 0.742
#   normalized_names: ('training data', 'processed dataset')
#   string_similarity: 0.423
#   token_similarity: 0.500
#   semantic_similarity: 0.800
#   substring_similarity: 0.000
```

## Semantic Knowledge Base

### Synonym Dictionary
The matcher includes domain-specific synonyms for pipeline concepts:

```python
synonyms = {
    'model': ['model', 'artifact', 'trained', 'output'],
    'data': ['data', 'dataset', 'input', 'processed', 'training'],
    'config': ['config', 'configuration', 'params', 'parameters', 'hyperparameters', 'settings'],
    'payload': ['payload', 'sample', 'test', 'inference', 'example'],
    'output': ['output', 'result', 'artifact', 'generated', 'produced'],
    'training': ['training', 'train', 'fit', 'learn'],
    'preprocessing': ['preprocessing', 'preprocess', 'processed', 'clean', 'transform'],
}
```

### Abbreviation Expansion
Common abbreviations are automatically expanded:

```python
abbreviations = {
    'config': 'configuration',
    'params': 'parameters',
    'hyperparams': 'hyperparameters',
    'preprocess': 'preprocessing',
    'eval': 'evaluation',
    'reg': 'registration',
    'pkg': 'package',
    'packaged': 'package',
}
```

### Stop Words
Noise words are filtered out during matching:

```python
stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
```

## Name Normalization

### Normalization Process
Names are normalized through several steps:

1. **Case Normalization** - Convert to lowercase
2. **Separator Replacement** - Replace `_`, `-`, `.` with spaces
3. **Special Character Removal** - Remove non-alphanumeric characters
4. **Abbreviation Expansion** - Expand known abbreviations
5. **Stop Word Removal** - Filter out noise words

```python
# Example normalization
original = "Training_Data-Config.v2"
normalized = "training data configuration"
```

## Integration with Dependency Resolution

### Automatic Matching
```python
from src.pipeline_deps.dependency_resolver import DependencyResolver
from src.pipeline_deps.semantic_matcher import semantic_matcher

class DependencyResolver:
    def __init__(self):
        self.semantic_matcher = semantic_matcher
        
    def find_compatible_outputs(self, dependency_name: str, available_outputs: List[str]):
        """Find outputs that can satisfy a dependency using semantic matching."""
        matches = self.semantic_matcher.find_best_matches(
            dependency_name, 
            available_outputs, 
            threshold=0.6
        )
        return matches
```

### Smart Dependency Resolution
```python
# Example: Resolving "model_input" dependency
dependency = "model_input"
available_outputs = [
    "trained_model_artifacts",
    "preprocessing_output", 
    "model_package",
    "evaluation_results"
]

# Semantic matcher finds best matches
matches = matcher.find_best_matches(dependency, available_outputs)
# Returns: [("trained_model_artifacts", 0.73), ("model_package", 0.65)]
```

## Advanced Features

### Custom Similarity Thresholds
```python
# Different thresholds for different use cases
high_confidence_matches = matcher.find_best_matches(target, candidates, threshold=0.8)
moderate_matches = matcher.find_best_matches(target, candidates, threshold=0.6)
loose_matches = matcher.find_best_matches(target, candidates, threshold=0.4)
```

### Batch Matching
```python
# Match multiple targets against candidates
targets = ["model_output", "training_data", "config_params"]
candidates = ["trained_model", "dataset", "hyperparameters", "results"]

all_matches = {}
for target in targets:
    matches = matcher.find_best_matches(target, candidates, threshold=0.5)
    all_matches[target] = matches
```

### Similarity Matrix
```python
# Generate similarity matrix for analysis
import pandas as pd

def generate_similarity_matrix(names1: List[str], names2: List[str]) -> pd.DataFrame:
    matrix = []
    for name1 in names1:
        row = []
        for name2 in names2:
            score = matcher.calculate_similarity(name1, name2)
            row.append(score)
        matrix.append(row)
    
    return pd.DataFrame(matrix, index=names1, columns=names2)

# Usage
dependencies = ["model_input", "training_data", "config"]
outputs = ["model_artifacts", "processed_data", "hyperparameters"]
similarity_df = generate_similarity_matrix(dependencies, outputs)
```

## Performance Considerations

### Caching
For repeated calculations, consider caching similarity scores:

```python
from functools import lru_cache

class CachedSemanticMatcher(SemanticMatcher):
    @lru_cache(maxsize=1000)
    def calculate_similarity(self, name1: str, name2: str) -> float:
        return super().calculate_similarity(name1, name2)
```

### Batch Processing
For large-scale matching, use vectorized operations:

```python
def batch_similarity(self, target: str, candidates: List[str]) -> List[float]:
    """Calculate similarity for multiple candidates efficiently."""
    return [self.calculate_similarity(target, candidate) for candidate in candidates]
```

## Customization

### Adding Domain-Specific Synonyms
```python
# Extend the matcher with custom synonyms
custom_matcher = SemanticMatcher()
custom_matcher.synonyms['risk'] = ['risk', 'score', 'probability', 'likelihood']
custom_matcher.synonyms['currency'] = ['currency', 'fx', 'exchange', 'rate']
```

### Custom Abbreviations
```python
# Add domain-specific abbreviations
custom_matcher.abbreviations.update({
    'fx': 'foreign_exchange',
    'ml': 'machine_learning',
    'nlp': 'natural_language_processing'
})
```

### Adjusting Weights
```python
# Custom weight configuration for different similarity aspects
class CustomSemanticMatcher(SemanticMatcher):
    def calculate_similarity(self, name1: str, name2: str) -> float:
        # Custom weights: emphasize semantic similarity
        weights = {
            'string': 0.2,
            'token': 0.2, 
            'semantic': 0.4,  # Higher weight for semantic matching
            'substring': 0.2
        }
        # Implementation with custom weights...
```

## Integration Points

### With Dependency Resolver
```python
from src.pipeline_deps.dependency_resolver import DependencyResolver

# Semantic matcher is used by dependency resolver for intelligent matching
resolver = DependencyResolver()
resolver.semantic_matcher = semantic_matcher
```

### With Specification Registry
```python
from src.pipeline_deps.specification_registry import SpecificationRegistry

# Registry can use semantic matching for specification lookup
registry = SpecificationRegistry()
similar_specs = registry.find_similar_specifications("model_training", threshold=0.7)
```

## Related Design Documentation

For architectural context and design decisions, see:
- **[Dependency Resolver Design](../pipeline_design/dependency_resolver.md)** - How semantic matching fits into dependency resolution
- **[Specification Driven Design](../pipeline_design/specification_driven_design.md)** - Overall design philosophy
- **[Design Principles](../pipeline_design/design_principles.md)** - Core design principles
- **[Standardization Rules](../pipeline_design/standardization_rules.md)** - Naming conventions that improve matching

## Best Practices

### 1. Naming Conventions
- Use descriptive, consistent naming for better semantic matching
- Avoid overly abbreviated names that reduce matching accuracy
- Include domain-specific terms that have semantic meaning

### 2. Threshold Selection
- Use higher thresholds (0.7-0.9) for critical dependencies
- Use moderate thresholds (0.5-0.7) for general matching
- Use lower thresholds (0.3-0.5) for exploratory matching

### 3. Synonym Management
- Regularly update synonym dictionaries with domain-specific terms
- Test synonym effectiveness with real pipeline names
- Balance synonym breadth with matching precision

### 4. Performance Optimization
- Cache similarity calculations for repeated operations
- Use batch processing for large-scale matching
- Consider approximate matching for very large candidate sets

## Troubleshooting

### Low Similarity Scores
- Check if names are using consistent terminology
- Verify that relevant synonyms are included in the dictionary
- Consider if abbreviations need to be added to the expansion list

### False Positive Matches
- Increase similarity threshold to reduce false positives
- Review and refine synonym dictionaries
- Add negative examples to improve discrimination

### Performance Issues
- Implement caching for repeated calculations
- Use batch processing for multiple comparisons
- Consider pre-computing similarity matrices for static datasets
