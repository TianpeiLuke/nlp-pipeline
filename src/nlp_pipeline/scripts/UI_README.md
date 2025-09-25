# Data Viewer UI Documentation

A Streamlit-based web application for viewing and analyzing parsed parquet data with dialogue and shiptrack information.

## Quick Start

### 🚀 Enhanced Launcher with Integrated Parsing

The `run_ui.py` script now combines parquet parsing with UI launching for seamless data analysis:

```bash
cd src/scripts

# Launch UI with default file
python run_ui.py

# Launch UI with specific file
python run_ui.py --file your_data.parquet

# Pre-parse data first, then launch UI (recommended for large files)
python run_ui.py --file your_data.parquet --parse

# Only analyze data without launching UI
python run_ui.py --analyze-only your_data.parquet

# Launch on different port
python run_ui.py --port 8502
```

### Traditional Launch Method
```bash
cd src/scripts
streamlit run data_viewer_ui.py
```

The UI will open in your browser at `http://localhost:8501` (or specified port)

## Features

### Three-Panel Layout

#### Left Panel: Dialogue & Shiptrack Data
- **💬 Dialogue Messages**: Displays parsed dialogue messages in expandable sections
- **🚚 Shiptrack Events**: Shows shipment sequences organized by Shipment ID
- Each shipment sequence displays all related tracking events

#### Middle Panel: LLM Analysis
- **🤖 Category**: Classification result
- **📊 Confidence Score**: Model confidence level
- **📝 Evidence Fields**: 
  - Message Evidence
  - Shipping Evidence
  - Timeline Evidence
- **🔍 Analysis Factors**:
  - Primary Factors
  - Supporting Evidence
  - Contradicting Evidence

#### Right Panel: Flags & Metadata
- **🏷️ Main Flags**:
  - Concession Type
  - Reversal Flag (✅/❌ indicators)
  - Reclassification Flag (✅/❌ indicators)
- **📊 Additional Metadata**:
  - Order ID
  - Thread ID
  - Message Count
  - Total Ship Track Events

## Navigation

- **Row Selector**: Dropdown menu to jump to any specific row
- **⬅️ Previous / ➡️ Next Buttons**: Navigate sequentially through records
- **Record Display**: Shows "Record X of Y" with Order ID for context

## Data Loading

1. **File Upload**: Use the sidebar to upload a parquet file directly
2. **File Path**: Enter the path to your parquet file (default: `.data/rnr_llm_deep_dive_result_patch_3_2025_v3.parquet`)
3. Click **"🔄 Load Data"** to process the file

## UI Features

- **Expandable Sections**: Click to expand/collapse dialogue messages and shiptrack events
- **Formatted Text Areas**: All text content is displayed in scrollable text areas
- **Color-Coded Flags**: Green checkmarks for true flags, gray X for false flags
- **Metrics Display**: Key statistics shown as metric cards
- **Raw Data View**: Expandable section at bottom showing complete row data in JSON format
- **Caching**: Data is cached for faster navigation between rows

## File Structure
```
src/scripts/
├── data_viewer_ui.py      # Main Streamlit application
├── parquet_parser.py      # Data parsing functions
├── run_ui.py             # Launcher script
├── requirements_ui.txt    # Dependencies
└── UI_README.md          # This documentation
```

## 🎯 Integrated Parsing Features

### Command Line Options
```bash
python run_ui.py --help
```

**Available Options:**
- `--file FILE, -f FILE`: Specify parquet file path
- `--parse, -p`: Pre-parse data before launching UI (recommended for large files)
- `--output OUTPUT, -o OUTPUT`: Output path for parsed data (used with --parse)
- `--analyze-only FILE, -a FILE`: Only analyze file, don't launch UI
- `--port PORT`: Custom port for Streamlit server (default: 8501)

### Smart File Detection
The launcher automatically searches for files in common locations:
- Current directory
- `.data/` directory
- Project root `.data/` directory

### Auto-Loading
When launched with `--file`, the UI automatically loads the specified file without manual intervention.

## Dependencies

All required dependencies are already installed:
- ✅ streamlit (1.37.1)
- ✅ pandas (2.3.1)
- ✅ pyarrow (16.1.0)

## Verified Functionality

✅ All imports successful  
✅ Parser functions available  
✅ Streamlit available  
✅ Dialogue parsing works (39 messages in sample)  
✅ Shiptrack parsing works (1 sequence with 23 events in sample)  
✅ Integrated launcher with parsing  
✅ Command line argument handling  
✅ Auto-file detection and loading  
🎉 Complete integrated solution ready!

## Usage Example

1. Launch the UI: `python run_ui.py`
2. Load your parquet file using the sidebar
3. Navigate through records using the dropdown or Previous/Next buttons
4. View parsed dialogue and shiptrack data in the left panel
5. Review LLM analysis in the middle panel
6. Check flags and metadata in the right panel
7. Expand "View Raw Data" to see complete record information

The UI automatically parses the data and displays it in an organized, human-readable format across the three panels as requested.
