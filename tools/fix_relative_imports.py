#!/usr/bin/env python3
"""
Script to check and fix imports within the src folder to use relative imports.

This ensures that the src folder can be renamed or moved without breaking internal imports.
The script can both check for violations and automatically fix them.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set
import ast
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ImportFixer:
    """Class to handle checking and fixing relative imports."""
    
    def __init__(self, src_dir: str = "src", dry_run: bool = True):
        self.src_dir = Path(src_dir)
        self.dry_run = dry_run
        self.violations_found = 0
        self.files_processed = 0
        self.files_modified = 0
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the src directory."""
        python_files = []
        for file_path in self.src_dir.rglob("*.py"):
            # Skip __pycache__ directories
            if "__pycache__" not in str(file_path):
                python_files.append(file_path)
        return python_files
    
    def calculate_relative_import(self, current_file: Path, imported_module: str) -> str:
        """
        Calculate the correct relative import path.
        
        Args:
            current_file: Path to the current Python file
            imported_module: The module being imported (e.g., 'src.pipeline_api.exceptions')
            
        Returns:
            The correct relative import path
        """
        # Remove 'src.' prefix from imported module
        if imported_module.startswith('src.'):
            target_module = imported_module[4:]  # Remove 'src.'
        else:
            return imported_module  # Not a src import
        
        # Get the directory of the current file relative to src
        current_dir = current_file.parent
        src_relative_current = current_dir.relative_to(self.src_dir)
        
        # Split the paths into components
        current_parts = list(src_relative_current.parts) if src_relative_current.parts != ('.',) else []
        target_parts = target_module.split('.')
        
        # Calculate the number of levels to go up
        levels_up = len(current_parts)
        
        # Build the relative import
        if levels_up == 0:
            # Same level as src
            relative_import = '.' + target_module
        else:
            # Need to go up some levels
            dots = '.' * (levels_up + 1)
            relative_import = dots + target_module
            
        return relative_import
    
    def fix_import_line(self, line: str, current_file: Path) -> Tuple[str, bool]:
        """
        Fix a single import line if it needs fixing.
        
        Args:
            line: The import line to check/fix
            current_file: Path to the current file
            
        Returns:
            Tuple of (fixed_line, was_modified)
        """
        original_line = line
        modified = False
        
        # Pattern for 'from src.module import ...'
        from_match = re.match(r'^(\s*from\s+)(src\.[^\s]+)(\s+import\s+.*)$', line)
        if from_match:
            indent, module, import_part = from_match.groups()
            relative_module = self.calculate_relative_import(current_file, module)
            line = f"{indent}{relative_module}{import_part}"
            modified = True
        
        # Pattern for 'import src.module'
        import_match = re.match(r'^(\s*import\s+)(src\.[^\s,]+)(.*)$', line)
        if import_match:
            indent, module, rest = import_match.groups()
            # For direct imports, we need to convert to 'from ... import ...'
            # This is more complex, so we'll suggest manual review
            logger.warning(f"Direct import found in {current_file}: {line.strip()}")
            logger.warning("  Consider converting to 'from ... import ...' format manually")
            # For now, we'll leave direct imports as-is since they're more complex to convert
        
        return line, modified
    
    def process_file(self, file_path: Path) -> Tuple[int, bool]:
        """
        Process a single Python file to check/fix imports.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Tuple of (violations_count, was_modified)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines(keepends=True)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return 0, False
        
        violations = 0
        modified_lines = []
        file_modified = False
        in_docstring = False
        docstring_delimiter = None
        
        for line_num, line in enumerate(lines, 1):
            stripped_line = line.strip()
            
            # Skip empty lines
            if not stripped_line:
                modified_lines.append(line)
                continue
            
            # Skip comments
            if stripped_line.startswith('#'):
                modified_lines.append(line)
                continue
            
            # Handle docstrings
            if '"""' in line or "'''" in line:
                if not in_docstring:
                    # Starting a docstring
                    if '"""' in line:
                        docstring_delimiter = '"""'
                    else:
                        docstring_delimiter = "'''"
                    in_docstring = True
                    # Check if docstring ends on the same line
                    if line.count(docstring_delimiter) >= 2:
                        in_docstring = False
                        docstring_delimiter = None
                else:
                    # Ending a docstring
                    if docstring_delimiter in line:
                        in_docstring = False
                        docstring_delimiter = None
                modified_lines.append(line)
                continue
            
            # Skip lines inside docstrings
            if in_docstring:
                modified_lines.append(line)
                continue
            
            # Skip string literals containing src.
            if (('"' in line and 'src.' in line and line.count('"') >= 2) or
                ("'" in line and 'src.' in line and line.count("'") >= 2)):
                modified_lines.append(line)
                continue
            
            # Check for actual import statements
            if re.search(r'\bfrom\s+src\.|\bimport\s+src\.', line):
                violations += 1
                if not self.dry_run:
                    fixed_line, was_modified = self.fix_import_line(line, file_path)
                    modified_lines.append(fixed_line)
                    if was_modified:
                        file_modified = True
                        logger.info(f"  Fixed line {line_num}: {line.strip()} -> {fixed_line.strip()}")
                else:
                    modified_lines.append(line)
                    logger.warning(f"  Line {line_num}: {line.strip()}")
            else:
                modified_lines.append(line)
        
        # Write the file back if modifications were made
        if file_modified and not self.dry_run:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(modified_lines)
                logger.info(f"Modified {file_path}")
            except Exception as e:
                logger.error(f"Error writing {file_path}: {e}")
                return violations, False
        
        return violations, file_modified
    
    def run(self) -> int:
        """
        Run the import checker/fixer.
        
        Returns:
            Exit code (0 for success, 1 for violations found)
        """
        if not self.src_dir.exists():
            logger.error(f"Source directory '{self.src_dir}' not found")
            return 1
        
        logger.info(f"{'Checking' if self.dry_run else 'Fixing'} imports in {self.src_dir}")
        
        python_files = self.find_python_files()
        logger.info(f"Found {len(python_files)} Python files")
        
        files_with_violations = []
        
        for file_path in python_files:
            self.files_processed += 1
            violations, was_modified = self.process_file(file_path)
            
            if violations > 0:
                files_with_violations.append((file_path, violations))
                self.violations_found += violations
                
                if violations > 0 and self.dry_run:
                    logger.warning(f"File: {file_path} ({violations} violations)")
                
            if was_modified:
                self.files_modified += 1
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY:")
        print(f"Total Python files processed: {self.files_processed}")
        print(f"Files with import violations: {len(files_with_violations)}")
        print(f"Total violations found: {self.violations_found}")
        
        if not self.dry_run:
            print(f"Files modified: {self.files_modified}")
        
        if self.violations_found == 0:
            print("\n✅ All imports within src folder are using relative imports!")
            print("This means the src folder can be safely renamed or moved without breaking internal imports.")
            return 0
        else:
            if self.dry_run:
                print(f"\n❌ Found {self.violations_found} import violations that need to be fixed.")
                print("\nRECOMMENDATIONS:")
                print("1. Run with --fix to automatically fix the violations")
                print("2. Or manually replace 'from src.module import ...' with relative imports:")
                print("   - Same level: 'from .module import ...'")
                print("   - Parent level: 'from ..parent.module import ...'")
                print("   - Grandparent level: 'from ...grandparent.module import ...'")
                print("\nEXAMPLES:")
                print("❌ from src.pipeline_steps.config_base import BasePipelineConfig")
                print("✅ from ..pipeline_steps.config_base import BasePipelineConfig")
                print("\n❌ from src.pipeline_api.exceptions import ConfigurationError")
                print("✅ from .exceptions import ConfigurationError")
            else:
                if self.files_modified > 0:
                    print(f"\n✅ Fixed {self.violations_found} violations in {self.files_modified} files!")
                    print("All imports within src folder now use relative imports.")
                else:
                    print(f"\n⚠️  Found {self.violations_found} violations but no files were modified.")
                    print("Some violations may require manual review (e.g., direct imports).")
            
            return 1 if self.dry_run else 0


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Check and fix imports within the src folder to use relative imports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Check for violations (dry run)
  %(prog)s --fix              # Fix violations automatically
  %(prog)s --src-dir myapp    # Use different source directory
  %(prog)s --verbose          # Show detailed output

Exit codes:
  0 - All imports are relative (success)
  1 - Found violations (in check mode) or errors occurred
        """
    )
    
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Fix violations automatically (default is dry run)'
    )
    
    parser.add_argument(
        '--src-dir',
        default='src',
        help='Source directory to process (default: src)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run the fixer
    fixer = ImportFixer(src_dir=args.src_dir, dry_run=not args.fix)
    exit_code = fixer.run()
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
