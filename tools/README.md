# Tools Directory

This directory contains utility scripts for maintaining the codebase.

## fix_relative_imports.py

A Python script to check and automatically fix imports within the `src` folder to use relative imports. This ensures that the `src` folder can be renamed or moved without breaking internal imports.

### Usage

```bash
# Check for import violations (dry run)
python tools/fix_relative_imports.py

# Fix violations automatically
python tools/fix_relative_imports.py --fix

# Use a different source directory
python tools/fix_relative_imports.py --src-dir myapp

# Show verbose output
python tools/fix_relative_imports.py --verbose

# Get help
python tools/fix_relative_imports.py --help
```

### Features

- **Smart Detection**: Only flags actual import statements, ignoring comments, docstrings, and string literals
- **Automatic Fixing**: Can automatically convert absolute imports to relative imports
- **Safe Operation**: Dry run by default, requires explicit `--fix` flag to make changes
- **Comprehensive**: Handles complex directory structures and calculates correct relative paths
- **Robust**: Properly handles docstrings, comments, and edge cases

### Examples

The script converts imports like:

```python
# ❌ Absolute imports (bad)
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_api.exceptions import ConfigurationError

# ✅ Relative imports (good)
from ..pipeline_steps.config_base import BasePipelineConfig
from .exceptions import ConfigurationError
```

### Exit Codes

- `0`: All imports are relative (success)
- `1`: Found violations (in check mode) or errors occurred

### Benefits

Using relative imports within the `src` folder provides several advantages:

1. **Portability**: The entire `src` folder can be renamed or moved without breaking internal imports
2. **Maintainability**: Easier to refactor and reorganize code structure
3. **Clarity**: Makes it clear which imports are internal vs external dependencies
4. **Best Practice**: Follows Python packaging best practices

## Other Tools

- `validate_contracts.py`: Validates pipeline step contracts
- `validate_step_names.py`: Validates step name consistency
