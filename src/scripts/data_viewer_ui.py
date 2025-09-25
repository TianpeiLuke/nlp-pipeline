#!/usr/bin/env python3
"""
Data Viewer UI for Parsed Parquet Data

A Streamlit web application that displays parsed dialogue and shiptrack data
in a three-panel layout for easy human review.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import re
from typing import List, Dict, Any
from parquet_parser import parse_column_data, parse_dialogue_messages, parse_shiptrack_events

# Configure page
st.set_page_config(
    page_title="Data Viewer - Parsed Parquet Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_parse_data(file_path: str) -> pd.DataFrame:
    """Load and parse the parquet file with caching."""
    try:
        df = pd.read_parquet(file_path)
        
        # Parse dialogue_x if it exists
        if 'dialogue_x' in df.columns:
            df['dialogue_parsed'] = parse_column_data(df, 'dialogue_x', 'dialogue')
        
        # Parse shiptrack column if it exists
        if 'shiptrack_event_history_by_order' in df.columns:
            df['shiptrack_parsed'] = parse_column_data(df, 'shiptrack_event_history_by_order', 'shiptrack')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def clean_html_content(text: str) -> str:
    """Extract clean text from HTML content."""
    if not text or not isinstance(text, str):
        return text
    
    # Check if the text contains HTML tags
    if '<html>' in text.lower() or '<div>' in text.lower() or '<span>' in text.lower() or '<table>' in text.lower():
        # For complex HTML emails, try to extract meaningful content
        
        # First, replace common HTML elements with appropriate text
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</?p[^>]*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</?div[^>]*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</tr>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</td>', ' | ', text, flags=re.IGNORECASE)
        
        # Remove script and style content completely
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove all remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up extra whitespace and newlines
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)       # Multiple spaces to single
        text = re.sub(r' \| \| ', ' | ', text)    # Clean up table separators
        text = re.sub(r'^\s*\|\s*', '', text, flags=re.MULTILINE)  # Remove leading pipes
        text = re.sub(r'\s*\|\s*$', '', text, flags=re.MULTILINE)  # Remove trailing pipes
        
        # Replace HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        text = text.replace('&mdash;', '—')
        text = text.replace('&ndash;', '–')
        
        # Remove very long URLs (likely tracking links)
        text = re.sub(r'https?://[^\s]{50,}', '[URL]', text)
        
        # Clean up and format
        text = text.strip()
        
        # If the result is still very long and messy, try to extract key information
        if len(text) > 1000 and 'Status do seu pedido' in text:
            # This looks like a shipping status email, extract key info
            lines = text.split('\n')
            key_lines = []
            for line in lines:
                line = line.strip()
                if line and any(keyword in line.lower() for keyword in [
                    'status', 'pedido', 'entrega', 'transportadora', 'criado', 
                    'despachado', 'trânsito', 'entregue', 'data estimada'
                ]):
                    key_lines.append(line)
            
            if key_lines:
                text = '\n'.join(key_lines[:10])  # Limit to first 10 relevant lines
    
    return text

def display_dialogue_messages(messages: List[str], current_row: int):
    """Display dialogue messages in a formatted way."""
    st.subheader("💬 Dialogue Messages")
    
    if not messages:
        st.info("No dialogue messages found")
        return
    
    # Clean up messages by removing [Arrival Time]: prefix and HTML content
    cleaned_messages = []
    for message in messages:
        # Remove [Arrival Time]: prefix if present
        if message.startswith("[Arrival Time]:"):
            cleaned_message = message.replace("[Arrival Time]:", "").strip()
        else:
            cleaned_message = message
        
        # Clean HTML content from the message
        cleaned_message = clean_html_content(cleaned_message)
        cleaned_messages.append(cleaned_message)
    
    # Join all messages with double newlines for better visual separation
    combined_messages = "\n\n".join(cleaned_messages)
    st.text_area(
        "All Dialogue Messages",
        value=combined_messages,
        height=300,  # Increased height to accommodate multiple messages
        key=f"dialogue_combined_{current_row}",
        label_visibility="collapsed"
    )

def display_shiptrack_events(shiptrack_sequences: List[dict], current_row: int):
    """Display shiptrack events organized by shipment sequences."""
    st.subheader("🚚 Shiptrack Events")
    
    if not shiptrack_sequences:
        st.info("No shiptrack events found")
        return
    
    for seq_idx, shipment_data in enumerate(shiptrack_sequences, 1):
        if not shipment_data or not isinstance(shipment_data, dict):
            continue
            
        shipment_id = shipment_data.get("shipment_id", "Unknown")
        event_sequence = shipment_data.get("shipment_event_sequence", [])
        
        if not event_sequence:
            continue
        
        with st.expander(f"Shipment {seq_idx}: {shipment_id}", expanded=seq_idx == 1):
            # Join all events with double newlines for better visual separation
            combined_events = "\n\n".join(event_sequence)
            st.text_area(
                f"Events for Shipment {shipment_id}",
                value=combined_events,
                height=200,  # Increased height to accommodate multiple events
                key=f"shiptrack_seq_{seq_idx}_combined_{current_row}",
                label_visibility="collapsed"
            )

def display_llm_output(row: pd.Series, current_row: int):
    """Display LLM analysis output."""
    st.subheader("🤖 LLM Analysis")
    
    # Fields that contain lists with [sep] separators
    evidence_fields = [
        'message_evidence', 'shipping_evidence', 'timeline_evidence', 
        'primary_factors', 'supporting_evidence', 'contradicting_evidence'
    ]
    
    # Simple fields that don't need special parsing
    simple_fields = ['category', 'confidence_score']
    
    # Display simple fields first
    for field in simple_fields:
        if field in row.index:
            value = row[field]
            if pd.notna(value):
                st.text_area(
                    field.replace('_', ' ').title(),
                    value=str(value),
                    height=100,
                    key=f"llm_{field}_{current_row}"
                )
            else:
                st.info(f"{field.replace('_', ' ').title()}: No data")
    
    # Display evidence fields with special parsing
    for field in evidence_fields:
        if field in row.index:
            value = row[field]
            # Parse the evidence field
            parsed_items = parse_evidence_field(value)
            if parsed_items:
                # Join items with double newlines for better separation
                combined_items = "\n\n".join(parsed_items)
                st.text_area(
                    field.replace('_', ' ').title(),
                    value=combined_items,
                    height=150,  # Increased height for multiple items
                    key=f"llm_{field}_{current_row}"
                )
            else:
                st.info(f"{field.replace('_', ' ').title()}: No data")

def parse_evidence_field(value) -> List[str]:
    """Parse evidence fields that contain lists of strings with [sep] prefixes."""
    if value is None or str(value).strip() == '' or str(value) == 'nan':
        return []
    
    # Handle numpy arrays
    import numpy as np
    if isinstance(value, np.ndarray):
        items = value.tolist()  # Convert numpy array to Python list
    elif isinstance(value, list):
        items = value
    else:
        # Try to evaluate as a Python list string representation
        try:
            import ast
            items = ast.literal_eval(str(value))
            if not isinstance(items, list):
                items = [str(value)]
        except:
            # If parsing fails, treat as single item
            items = [str(value)]
    
    cleaned_items = []
    for item in items:
        item_str = str(item).strip()
        # Remove [sep] prefix if present
        if item_str.startswith('[sep]'):
            cleaned_item = item_str[5:].strip()  # Remove '[sep]' (5 characters)
            # Only skip very specific fallback patterns
            if (cleaned_item and 
                cleaned_item != 'None' and 
                cleaned_item != 'Unable to parse detailed shipping evidence' and
                cleaned_item != 'Unable to parse detailed message evidence' and
                cleaned_item != 'Unable to parse detailed timeline evidence' and
                cleaned_item != 'Fallback analysis based on keywords' and
                cleaned_item != 'Fallback classification due to parsing error'):
                cleaned_items.append(cleaned_item)
        elif (item_str and 
              item_str != 'None'):  # Add all non-empty items except 'None'
            cleaned_items.append(item_str)
    
    return cleaned_items

def display_flags_and_metadata(row: pd.Series):
    """Display flags and metadata."""
    st.subheader("🏷️ Flags & Metadata")
    
    # Display Order ID prominently at the top
    if 'order_id' in row.index and pd.notna(row['order_id']):
        st.info(f"📋 **Order ID**: {row['order_id']}")
    else:
        st.info("📋 **Order ID**: No data")
    
    # Display Org below Order ID
    if 'org' in row.index and pd.notna(row['org']):
        st.info(f"🏢 **Org**: {row['org']}")
    else:
        st.info("🏢 **Org**: No data")
    
    st.markdown("---")
    
    # Main flags
    flag_fields = ['concession_type', 'reversal_flag', 'reclassification_flag']
    
    for field in flag_fields:
        if field in row.index:
            value = row[field]
            if pd.notna(value):
                if field == 'reversal_flag':
                    # Special color coding for reversal_flag
                    if value == "Reversal":
                        st.success(f"✅ {field.replace('_', ' ').title()}: {value}")
                    elif value == "Others":
                        st.info(f"⚪ {field.replace('_', ' ').title()}: {value}")
                    else:
                        st.error(f"❌ {field.replace('_', ' ').title()}: {value}")
                elif field == 'reclassification_flag':
                    # Special color coding for reclassification_flag
                    if value == "Reclassification":
                        st.success(f"✅ {field.replace('_', ' ').title()}: {value}")
                    elif value == "Others":
                        st.error(f"❌ {field.replace('_', ' ').title()}: {value}")
                    else:
                        st.info(f"⚪ {field.replace('_', ' ').title()}: {value}")
                elif field == 'concession_type':
                    # Special color coding for concession_type
                    if value == "NOTR":
                        st.error(f"❌ {field.replace('_', ' ').title()}: {value}")
                    elif value == "PDA":
                        st.warning(f"⚠️ {field.replace('_', ' ').title()}: {value}")
                    else:
                        st.success(f"✅ {field.replace('_', ' ').title()}: {value}")
                elif field.endswith('_flag'):
                    # Display other flags as colored badges
                    if value:
                        st.success(f"✅ {field.replace('_', ' ').title()}: {value}")
                    else:
                        st.info(f"❌ {field.replace('_', ' ').title()}: {value}")
                else:
                    st.info(f"📋 {field.replace('_', ' ').title()}: {value}")
            else:
                st.info(f"{field.replace('_', ' ').title()}: No data")
    
    # Additional metadata
    st.subheader("📊 Additional Info")
    metadata_fields = ['THREAD_ID', 'unique_message_count', 'total_ship_track_events_by_order']
    
    for field in metadata_fields:
        if field in row.index and pd.notna(row[field]):
            st.metric(field.replace('_', ' ').title(), row[field])

def main():
    """Main application function."""
    st.title("📊 Data Viewer - Parsed Parquet Data")
    st.markdown("---")
    
    # Check for file path from environment variable or command line
    env_file_path = os.environ.get('PARQUET_FILE_PATH')
    cmd_line_args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # Parse command line arguments if available
    default_file_path = ".data/rnr_llm_deep_dive_result_patch_3_2025_v3.parquet"
    if env_file_path:
        default_file_path = env_file_path
    elif '--file' in cmd_line_args:
        try:
            file_idx = cmd_line_args.index('--file')
            if file_idx + 1 < len(cmd_line_args):
                default_file_path = cmd_line_args[file_idx + 1]
        except (ValueError, IndexError):
            pass
    
    # Auto-load file if provided via environment or command line
    if env_file_path and 'data_path' not in st.session_state:
        st.session_state.data_path = env_file_path
        st.info(f"🎯 Auto-loaded file: {env_file_path}")
    
    # Sidebar for file selection and navigation
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Show current file if auto-loaded
        if 'data_path' in st.session_state:
            st.success(f"📁 Current file: {st.session_state.data_path}")
        
        # File upload or path input
        uploaded_file = st.file_uploader(
            "Upload Parquet File",
            type=['parquet'],
            help="Upload a parquet file to analyze"
        )
        
        file_path = st.text_input(
            "Or enter file path:",
            value=default_file_path,
            help="Enter the path to your parquet file"
        )
        
        # Load data button
        if st.button("🔄 Load Data", type="primary"):
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with open("temp_uploaded.parquet", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.data_path = "temp_uploaded.parquet"
            elif file_path:
                st.session_state.data_path = file_path
            else:
                st.error("Please provide a file path or upload a file")
                return
            st.rerun()
    
    # Load data if path is available
    if 'data_path' not in st.session_state:
        st.info("👆 Please load a parquet file using the sidebar controls")
        return
    
    # Load and cache data
    df = load_and_parse_data(st.session_state.data_path)
    
    if df.empty:
        st.error("Failed to load data or data is empty")
        return
    
    # Display data info
    st.success(f"✅ Loaded data: {len(df)} rows, {len(df.columns)} columns")
    
    # Initialize current_row if not exists
    if 'current_row' not in st.session_state:
        st.session_state.current_row = 0
    
    # Navigation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Handle button clicks first
    with col1:
        if st.button("⬅️ Previous", disabled=st.session_state.get('current_row', 0) <= 0):
            st.session_state.current_row = max(0, st.session_state.get('current_row', 0) - 1)
            st.rerun()
    
    with col3:
        if st.button("➡️ Next", disabled=st.session_state.get('current_row', 0) >= len(df) - 1):
            st.session_state.current_row = min(len(df) - 1, st.session_state.get('current_row', 0) + 1)
            st.rerun()
    
    with col2:
        # Row selector - use current_row from session state
        current_row = st.session_state.get('current_row', 0)
        
        # Force selectbox to update by using a dynamic key
        selected_row = st.selectbox(
            "Select Row:",
            options=range(len(df)),
            index=current_row,
            format_func=lambda x: f"Row {x+1} (ID: {df.iloc[x].get('order_id', 'N/A')})",
            key=f"row_selector_{current_row}"  # Dynamic key forces refresh
        )
        
        # Update session state if selection changed
        if selected_row != current_row:
            st.session_state.current_row = selected_row
            st.rerun()
    
    # Get current row data
    row = df.iloc[current_row]
    
    st.markdown("---")
    st.subheader(f"📄 Record {current_row + 1} of {len(df)}")
    
    # Three-panel layout
    left_col, middle_col, right_col = st.columns([1, 1, 1])
    
    # Left Panel: Dialogue and Shiptrack
    with left_col:
        st.markdown("### 📋 Left Panel")
        
        # Display dialogue messages
        if 'dialogue_parsed' in df.columns and row['dialogue_parsed'] is not None and len(row['dialogue_parsed']) > 0:
            display_dialogue_messages(row['dialogue_parsed'], current_row)
        else:
            st.info("No parsed dialogue data available")
        
        st.markdown("---")
        
        # Display shiptrack events
        if 'shiptrack_parsed' in df.columns and row['shiptrack_parsed'] is not None and len(row['shiptrack_parsed']) > 0:
            display_shiptrack_events(row['shiptrack_parsed'], current_row)
        else:
            st.info("No parsed shiptrack data available")
    
    # Middle Panel: LLM Output
    with middle_col:
        st.markdown("### 🤖 Middle Panel")
        display_llm_output(row, current_row)
    
    # Right Panel: Flags and Metadata
    with right_col:
        st.markdown("### 🏷️ Right Panel")
        display_flags_and_metadata(row)
    
    # Footer with raw data option
    st.markdown("---")
    with st.expander("🔍 View Raw Data for Current Row"):
        st.json(row.to_dict(), expanded=False)

if __name__ == "__main__":
    main()
