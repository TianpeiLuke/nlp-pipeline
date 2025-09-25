#!/usr/bin/env python3
"""
Parquet Parser for Dialogue and Shiptrack Data

This script reads a parquet file and parses dialogue_x and shiptrack_tracking_id_lists_by_order
columns that contain concatenated messages/events with [bom] and [eom] markers.

For dialogue_x: Simple message parsing with [bom] and [eom] markers
For shiptrack_tracking_id_lists_by_order: Complex parsing with shipment sequences
"""

import pandas as pd
import re
from typing import List, Union, Optional
import argparse
import sys
from pathlib import Path


def parse_dialogue_messages(text: str) -> List[str]:
    """
    Parse dialogue text containing messages separated by [bom] and [eom] markers.
    
    Args:
        text (str): Raw text containing dialogue messages
        
    Returns:
        List[str]: List of individual messages
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    # Pattern to match content between [bom] and [eom]
    pattern = r'\[bom\](.*?)\[eom\]'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Clean up the messages by stripping whitespace
    messages = [match.strip() for match in matches if match.strip()]
    
    return messages


def parse_shiptrack_events(text: str) -> List[dict]:
    """
    Parse shiptrack text containing sequences of events for different shipments.
    Each sequence starts with [bom] [Shipment ID]: xxx [eom] and ends with [bom] End of Ship Track Event for xxxx [eom]
    
    Args:
        text (str): Raw text containing shiptrack events
        
    Returns:
        List[dict]: List of shipment dictionaries with 'shipment_id' and 'shipment_event_sequence' keys
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    shipment_sequences = []
    
    # Pattern to find shipment ID markers
    shipment_id_pattern = r'\[bom\]\s*\[Shipment ID\]:\s*([^\[]+?)\s*\[eom\]'
    end_pattern = r'\[bom\]\s*End of Ship Track Event for\s+([^\[]+)\[eom\]'
    
    # Find all shipment ID positions
    shipment_matches = list(re.finditer(shipment_id_pattern, text))
    end_matches = list(re.finditer(end_pattern, text))
    
    if not shipment_matches:
        # If no shipment ID pattern found, treat as single sequence without ID
        messages = parse_dialogue_messages(text)
        if messages:
            return [{
                "shipment_id": "Unknown",
                "shipment_event_sequence": messages
            }]
        return []
    
    # Process each shipment sequence
    for i, shipment_match in enumerate(shipment_matches):
        # Extract shipment ID
        shipment_id = shipment_match.group(1).strip()
        
        start_pos = shipment_match.end()  # Start after the shipment ID marker
        
        # Find the end position for this sequence
        end_pos = len(text)
        
        # Look for the next shipment ID or end of ship track event marker
        if i + 1 < len(shipment_matches):
            end_pos = shipment_matches[i + 1].start()
        else:
            # This is the last shipment, look for "End of Ship Track Event"
            for end_match in end_matches:
                if end_match.start() > start_pos:
                    end_pos = end_match.start()  # End before the end marker
                    break
        
        # Extract the sequence text (excluding shipment ID and end markers)
        sequence_text = text[start_pos:end_pos]
        
        # Parse messages in this sequence
        all_messages = parse_dialogue_messages(sequence_text)
        
        # Filter out any remaining shipment ID or end markers
        event_messages = []
        for msg in all_messages:
            # Skip shipment ID markers and end markers
            if not (msg.startswith('[Shipment ID]:') or 
                   msg.startswith('End of Ship Track Event')):
                event_messages.append(msg)
        
        if event_messages:
            shipment_sequences.append({
                "shipment_id": shipment_id,
                "shipment_event_sequence": event_messages
            })
    
    return shipment_sequences


def parse_column_data(df: pd.DataFrame, column_name: str, parse_type: str = 'dialogue') -> pd.Series:
    """
    General function to parse a column containing concatenated messages/events.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column_name (str): Name of the column to parse
        parse_type (str): Type of parsing - 'dialogue' or 'shiptrack'
        
    Returns:
        pd.Series: Series with parsed data (lists or lists of lists)
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe")
    
    if parse_type == 'dialogue':
        return df[column_name].apply(parse_dialogue_messages)
    elif parse_type == 'shiptrack':
        return df[column_name].apply(parse_shiptrack_events)
    else:
        raise ValueError(f"Unknown parse_type: {parse_type}. Use 'dialogue' or 'shiptrack'")


def analyze_parquet_file(file_path: str) -> None:
    """
    Analyze the parquet file and show sample data from dialogue_x and shiptrack columns.
    
    Args:
        file_path (str): Path to the parquet file
    """
    print(f"Loading parquet file: {file_path}")
    
    try:
        # Load the parquet file
        df = pd.read_parquet(file_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print()
        
        # Check if target columns exist
        dialogue_col = 'dialogue_x'
        shiptrack_col = 'shiptrack_event_history_by_order'
        
        if dialogue_col in df.columns:
            print(f"=== Analysis of {dialogue_col} ===")
            print(f"Non-null values: {df[dialogue_col].notna().sum()}")
            
            # Show sample data
            sample_dialogue = df[dialogue_col].dropna().iloc[0] if df[dialogue_col].notna().any() else None
            if sample_dialogue:
                print(f"Sample raw data (first 500 chars):")
                print(sample_dialogue[:500])
                print("...")
                print()
                
                # Parse and show result
                parsed_messages = parse_dialogue_messages(sample_dialogue)
                print(f"Parsed into {len(parsed_messages)} messages:")
                for i, msg in enumerate(parsed_messages[:3]):  # Show first 3 messages
                    print(f"  Message {i+1}: {msg[:100]}...")
                if len(parsed_messages) > 3:
                    print(f"  ... and {len(parsed_messages) - 3} more messages")
                print()
        else:
            print(f"Column '{dialogue_col}' not found in dataset")
        
        if shiptrack_col in df.columns:
            print(f"=== Analysis of {shiptrack_col} ===")
            print(f"Non-null values: {df[shiptrack_col].notna().sum()}")
            
            # Show sample data
            sample_shiptrack = df[shiptrack_col].dropna().iloc[0] if df[shiptrack_col].notna().any() else None
            if sample_shiptrack:
                print(f"Sample raw data (first 500 chars):")
                print(sample_shiptrack[:500])
                print("...")
                print()
                
                # Parse and show result
                parsed_sequences = parse_shiptrack_events(sample_shiptrack)
                print(f"Parsed into {len(parsed_sequences)} shipment sequences:")
                for i, sequence in enumerate(parsed_sequences[:2]):  # Show first 2 sequences
                    print(f"  Sequence {i+1}: {len(sequence)} events")
                    for j, event in enumerate(sequence[:2]):  # Show first 2 events per sequence
                        print(f"    Event {j+1}: {event[:100]}...")
                    if len(sequence) > 2:
                        print(f"    ... and {len(sequence) - 2} more events")
                if len(parsed_sequences) > 2:
                    print(f"  ... and {len(parsed_sequences) - 2} more sequences")
                print()
        else:
            print(f"Column '{shiptrack_col}' not found in dataset")
            
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        sys.exit(1)


def process_parquet_file(file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Process the parquet file and add parsed columns.
    
    Args:
        file_path (str): Path to input parquet file
        output_path (str, optional): Path to save processed data
        
    Returns:
        pd.DataFrame: Processed dataframe with parsed columns
    """
    print(f"Processing parquet file: {file_path}")
    
    # Load the data
    df = pd.read_parquet(file_path)
    
    # Parse dialogue_x if it exists
    if 'dialogue_x' in df.columns:
        print("Parsing dialogue_x column...")
        df['dialogue_x_parsed'] = parse_column_data(df, 'dialogue_x', 'dialogue')
        print(f"Created dialogue_x_parsed with {df['dialogue_x_parsed'].apply(len).sum()} total messages")
    
    # Parse shiptrack column if it exists
    if 'shiptrack_event_history_by_order' in df.columns:
        print("Parsing shiptrack_event_history_by_order column...")
        df['shiptrack_parsed'] = parse_column_data(df, 'shiptrack_event_history_by_order', 'shiptrack')
        total_sequences = df['shiptrack_parsed'].apply(len).sum()
        total_events = df['shiptrack_parsed'].apply(lambda x: sum(len(seq) for seq in x)).sum()
        print(f"Created shiptrack_parsed with {total_sequences} sequences and {total_events} total events")
    
    # Save if output path provided
    if output_path:
        print(f"Saving processed data to: {output_path}")
        df.to_parquet(output_path, index=False)
    
    return df


def main():
    """Main function to handle command line arguments and execute parsing."""
    parser = argparse.ArgumentParser(description='Parse parquet file with dialogue and shiptrack data')
    parser.add_argument('input_file', help='Path to input parquet file')
    parser.add_argument('--analyze', action='store_true', help='Analyze the file and show sample data')
    parser.add_argument('--process', action='store_true', help='Process the file and add parsed columns')
    parser.add_argument('--output', '-o', help='Output path for processed parquet file')
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    if args.analyze:
        analyze_parquet_file(args.input_file)
    
    if args.process:
        df = process_parquet_file(args.input_file, args.output)
        print(f"Processing complete. Final dataset shape: {df.shape}")


if __name__ == "__main__":
    main()
