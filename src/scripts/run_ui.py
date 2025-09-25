#!/usr/bin/env python3
"""
Enhanced Launcher script for the Data Viewer UI with integrated parsing.
This script allows users to provide input data, parses it, and launches the UI.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
from parquet_parser import process_parquet_file, analyze_parquet_file

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Launch Data Viewer UI with integrated parquet parsing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ui.py                                    # Launch UI with default file
  python run_ui.py --file data.parquet               # Launch UI with specific file
  python run_ui.py --file data.parquet --parse       # Parse data first, then launch UI
  python run_ui.py --analyze-only data.parquet       # Only analyze data, don't launch UI
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        default='.data/rnr_llm_deep_dive_result_patch_3_2025_v3.parquet',
        help='Path to the parquet file to load (default: .data/rnr_llm_deep_dive_result_patch_3_2025_v3.parquet)'
    )
    
    parser.add_argument(
        '--parse', '-p',
        action='store_true',
        help='Pre-parse the data before launching UI (recommended for large files)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output path for parsed data (only used with --parse)'
    )
    
    parser.add_argument(
        '--analyze-only', '-a',
        type=str,
        metavar='FILE',
        help='Only analyze the specified file and exit (don\'t launch UI)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port for Streamlit server (default: 8501)'
    )
    
    return parser.parse_args()

def validate_file(file_path: str) -> Path:
    """Validate that the file exists and is accessible."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ Error: File '{file_path}' does not exist")
        
        # Try to find the file in common locations
        possible_paths = [
            Path.cwd() / file_path,
            Path.cwd() / '.data' / Path(file_path).name,
            Path(__file__).parent.parent.parent / '.data' / Path(file_path).name
        ]
        
        print("🔍 Searching for file in common locations...")
        for possible_path in possible_paths:
            if possible_path.exists():
                print(f"✅ Found file at: {possible_path}")
                return possible_path
        
        print("💡 Suggestions:")
        print("  - Check the file path is correct")
        print("  - Ensure the file is in the .data directory")
        print("  - Use absolute path if needed")
        sys.exit(1)
    
    return path

def main():
    """Main function with integrated parsing and UI launch."""
    args = parse_arguments()
    
    print("🎯 Data Viewer UI Launcher with Integrated Parsing")
    print("=" * 60)
    
    # Handle analyze-only mode
    if args.analyze_only:
        print(f"📊 Analyzing file: {args.analyze_only}")
        file_path = validate_file(args.analyze_only)
        try:
            analyze_parquet_file(str(file_path))
            print("✅ Analysis complete!")
        except Exception as e:
            print(f"❌ Error analyzing file: {e}")
            sys.exit(1)
        return
    
    # Validate input file
    input_file = validate_file(args.file)
    print(f"📁 Input file: {input_file}")
    
    # Pre-parse data if requested
    if args.parse:
        print("🔄 Pre-parsing data...")
        try:
            output_path = args.output or str(input_file.parent / f"parsed_{input_file.name}")
            df = process_parquet_file(str(input_file), output_path)
            print(f"✅ Data parsed and saved to: {output_path}")
            
            # Update input file to use parsed version
            input_file = Path(output_path)
            
        except Exception as e:
            print(f"❌ Error parsing data: {e}")
            print("⚠️  Continuing with original file...")
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    ui_script = script_dir / "data_viewer_ui.py"
    
    if not ui_script.exists():
        print(f"❌ Error: UI script not found at {ui_script}")
        sys.exit(1)
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Set environment variable for the UI to use the specified file
    os.environ['PARQUET_FILE_PATH'] = str(input_file)
    
    print("\n🚀 Launching Data Viewer UI...")
    print(f"📁 Working directory: {script_dir}")
    print(f"📊 Data file: {input_file}")
    print(f"🌐 UI will open at: http://localhost:{args.port}")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Launch Streamlit with the specified file
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(ui_script),
            "--server.port", str(args.port),
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--",  # Pass remaining args to streamlit app
            "--file", str(input_file)
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down Data Viewer UI...")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
