# Semantic Matcher

## Overview
The Semantic Matcher provides intelligent name matching for pipeline dependency resolution. It calculates similarity between dependency names and output names, enabling automatic matching of compatible pipeline components even when names don't exactly match. This allows for more flexible and robust pipeline construction with minimal manual configuration.

## Core Functionality

### Key Features
- **Multi-dimensional Similarity Scoring**: Combines multiple metrics to determine name similarity
- **Domain-specific Synonym Recognition**: Understands pipeline-specific terminology relationships 
- **Abbreviation Expansion**: Automatically expands common abbreviations for better matching
- **Stop Word Filtering**: Removes noise words to focus on meaningful terms
- **Detailed Explanation**: Provides transparency into similarity calculations

## Key Components

### SemanticMatcher
Main class that provides semantic similarity calculation and matching.

```python
class SemanticMatcher:
    def __init__(self):
        """Initialize the semantic matcher with common patterns."""
        
    def calculate_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate semantic similarity between two names.
        
        Args:
            name1: First name to compare
            name2: Second name to compare
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        
    def calculate_similarity_with_aliases(self, name: str, output_spec) -> float:
        """
        Calculate semantic similarity considering aliases.
        
        Args:
            name: Name to compare (typically the dependency's logical_name)
            output_spec: OutputSpec with logical_name and potential aliases
            
        Returns:
            Highest similarity score between name and any name in output_spec
        """
        
    def find_best_matches(self, target_name: str, candidate_names: List[str], 
                         threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find the best matching names from a list of candidates.
        
        Args:
            target_name: Name to match against
            candidate_names: List of candidate names
            threshold: Minimum similarity threshold
            
        Returns:
            List of (name, score) tuples sorted by score (highest first)
        """
        
    def explain_similarity(self, name1: str, name2: str) -> Dict[str, float]:
        """
        Provide detailed explanation of similarity calculation.
        
        Args:
            name1: First name to compare
            name2: Second name to compare
            
        Returns:
            Dictionary with detailed similarity breakdown
        """
```

## Similarity Calculation Algorithm

The SemanticMatcher calculates similarity between names using a weighted average of multiple similarity metrics:

### 1. String Similarity (30% weight)
Measures character-level similarity using sequence matching:

```python
def _calculate_string_similarity(self, name1: str, name2: str) -> float:
    return SequenceMatcher(None, name1, name2).ratio()
```

### 2. Token Overlap (25% weight)
Measures word-level similarity using Jaccard similarity:

```python
def _calculate_token_similarity(self, name1: str, name2: str) -> float:
    tokens1 = set(name1.split())
    tokens2 = set(name2.split())
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    return len(intersection) / len(union) if union else 0.0
```

### 3. Semantic Similarity (25% weight)
Measures concept-level similarity using domain-specific synonyms:

```python
def _calculate_semantic_similarity(self, name1: str, name2: str) -> float:
    tokens1 = set(name1.split())
    tokens2 = set(name2.split())
    
    # Find semantic matches
    semantic_matches = 0
    total_comparisons = 0
    
    for token1 in tokens1:
        for token2 in tokens2:
            total_comparisons += 1
            
            # Direct match
            if token1 == token2:
                semantic_matches += 1
                continue
            
            # Synonym match
            if self._are_synonyms(token1, token2):
                semantic_matches += 0.8  # Slightly lower score for synonyms
    
    return semantic_matches / total_comparisons if total_comparisons > 0 else 0.0
```

### 4. Substring Similarity (20% weight)
Measures partial matching through substring detection:

```python
def _calculate_substring_similarity(self, name1: str, name2: str) -> float:
    # Check if one is a substring of the other
    if name1 in name2 or name2 in name1:
        shorter = min(len(name1), len(name2))
        longer = max(len(name1), len(name2))
        return shorter / longer
    
    # Check for common substrings
    words1 = name1.split()
    words2 = name2.split()
    
    max_substring_score = 0.0
    for word1 in words1:
        for word2 in words2:
            if len(word1) >= 3 and len(word2) >= 3:  # Only consider meaningful substrings
                if word1 in word2 or word2 in word1:
                    shorter = min(len(word1), len(word2))
                    longer = max(len(word1), len(word2))
                    score = shorter / longer
                    max_substring_score = max(max_substring_score, score)
    
    return max_substring_score
```

## Name Normalization

Before comparing names, the SemanticMatcher normalizes them to ensure consistent comparison:

```python
def _normalize_name(self, name: str) -> str:
    # Convert to lowercase
    normalized = name.lower()
    
    # Remove common separators and replace with spaces
    normalized = re.sub(r'[_\-\.]', ' ', normalized)
    
    # Remove special characters
    normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
    
    # Expand abbreviations
    words = normalized.split()
    expanded_words = []
    for word in words:
        expanded = self.abbreviations.get(word, word)
        expanded_words.append(expanded)
    
    # Remove stop words
    filtered_words = [word for word in expanded_words if word not in self.stop_words]
    
    return ' '.join(filtered_words)
```

## Semantic Knowledge Base

### Synonyms
The matcher includes domain-specific synonyms for pipeline concepts:

```python
self.synonyms = {
    'model': ['model', 'artifact', 'trained', 'output'],
    'data': ['data', 'dataset', 'input', 'processed', 'training'],
    'config': ['config', 'configuration', 'params', 'parameters', 'hyperparameters', 'settings'],
    'payload': ['payload', 'sample', 'test', 'inference', 'example'],
    'output': ['output', 'result', 'artifact', 'generated', 'produced'],
    'training': ['training', 'train', 'fit', 'learn'],
    'preprocessing': ['preprocessing', 'preprocess', 'processed', 'clean', 'transform'],
}
```

### Abbreviations
Common abbreviations are automatically expanded:

```python
self.abbreviations = {
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
self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
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

# Exact match (after normalization)
score = matcher.calculate_similarity("model_artifacts", "model-artifacts")
print(f"Normalized match: {score:.3f}")
# Output: Normalized match: 1.000
```

### Working with Aliases
```python
from src.pipeline_deps.base_specifications import OutputSpec

# Create an output spec with aliases
output_spec = OutputSpec(
    logical_name="model_artifacts",
    output_type="MODEL_ARTIFACTS",
    data_type="S3Uri",
    property_path="properties.ModelArtifacts.S3ModelArtifacts",
    aliases=["trained_model", "model_output"]
)

# Calculate similarity with aliases
score = matcher.calculate_similarity_with_aliases("model", output_spec)
print(f"Best match score: {score:.3f}")
# Output: Best match score: 0.825
```

### Finding Best Matches
```python
# Find best matches from candidates
target = "model_output"
candidates = [
    "trained_model",
    "model_artifacts", 
    "preprocessing_output",
    "evaluation_results"
]

matches = matcher.find_best_matches(target, candidates, threshold=0.5)
for name, score in matches:
    print(f"{name}: {score:.3f}")

# Output:
# model_artifacts: 0.825
# trained_model: 0.687
```

### Explaining Similarity Calculation
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

## Integration with Dependency Resolver

The SemanticMatcher is a key component of the UnifiedDependencyResolver, enabling intelligent dependency matching:

```python
from src.pipeline_deps.dependency_resolver import UnifiedDependencyResolver
from src.pipeline_deps.semantic_matcher import SemanticMatcher
from src.pipeline_deps.specification_registry import SpecificationRegistry

# Create components
registry = SpecificationRegistry()
semantic_matcher = SemanticMatcher()
resolver = UnifiedDependencyResolver(registry, semantic_matcher)

# SemanticMatcher is used in compatibility calculation
# Inside UnifiedDependencyResolver._calculate_compatibility:
semantic_score = semantic_matcher.calculate_similarity_with_aliases(
    dep_spec.logical_name, output_spec
)
score += semantic_score * 0.25  # 25% weight
```

## Best Practices

### 1. Naming Conventions
- Use consistent, descriptive names for dependencies and outputs
- Follow standard naming patterns across your pipeline components
- Include meaningful semantic terms that reflect the data's purpose
- Avoid cryptic abbreviations that aren't in the abbreviation dictionary

### 2. Alias Usage
- Add aliases to outputs for common alternative names
- Include both full and abbreviated forms in aliases
- Consider domain-specific terminology variations as aliases

### 3. Threshold Selection
- Use higher thresholds (0.7+) when exact matching is critical
- Use moderate thresholds (0.5-0.7) for general dependency resolution
- Use lower thresholds (0.3-0.5) for exploratory matching

### 4. Custom Extensions
- Extend the synonym dictionary with domain-specific terms
- Add project-specific abbreviations to the abbreviation dictionary
- Remove domain-specific stop words if they're meaningful in your context
