"""
Bedrock Batch Inference Module

This module provides functionality to process data in batches using AWS Bedrock,
with parallel execution, progress tracking, and checkpoint management.
"""

import pandas as pd
import boto3
from tqdm.auto import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import logging
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from pathlib import Path

# Import custom modules
from .invoke_bedrock import analyze_dialogue_with_claude, get_bedrock_client
from .prompt_rnr_parse import BSMAnalysis, parse_claude_response


# Set up logging
logger = logging.getLogger(__name__)

# Configure file handler if not already configured
if not logger.handlers:
    log_dir = os.path.join(os.path.expanduser('~'), 'SageMaker', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'processing_log.txt')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def save_checkpoint(checkpoint_file: Union[str, Path], processed_rows: int, results: List[Dict]) -> None:
    """
    Save processing checkpoint to a file.
    
    Args:
        checkpoint_file (Union[str, Path]): Path to save the checkpoint
        processed_rows (int): Number of rows processed so far
        results (List[Dict]): List of processing results
    """
    try:
        checkpoint_path = Path(checkpoint_file)
        
        # Create directory if it doesn't exist
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'processed_rows': processed_rows,
            'results': [
                {
                    'index': r['index'],
                    'analysis': r['analysis'].model_dump()
                } for r in results
            ]
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
            
        logger.info(f"Checkpoint saved to {checkpoint_path} ({processed_rows} rows)")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")

        
def load_checkpoint(checkpoint_file: Union[str, Path]) -> Tuple[int, List[Dict]]:
    """
    Load processing checkpoint from a file.
    
    Args:
        checkpoint_file (Union[str, Path]): Path to the checkpoint file
        
    Returns:
        Tuple[int, List[Dict]]: Tuple containing (processed_rows, results)
    """
    checkpoint_path = Path(checkpoint_file)
    
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                
            results = [
                {
                    'index': r['index'],
                    'analysis': BSMAnalysis(**r['analysis'])
                } for r in checkpoint['results']
            ]
            
            processed_rows = checkpoint['processed_rows']
            logger.info(f"Loaded checkpoint from {checkpoint_path} ({processed_rows} rows)")
            return processed_rows, results
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            
    logger.info(f"No valid checkpoint found at {checkpoint_path}")
    return 0, []


def create_result_dataframe(results: List[Dict], original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a result DataFrame by combining original data with analysis results.
    
    Args:
        results (List[Dict]): List of analysis results
        original_df (pd.DataFrame): Original input DataFrame
        
    Returns:
        pd.DataFrame: Combined DataFrame with original data and analysis results
    """
    # Save original index
    original_df = original_df.copy()
    original_df['index'] = original_df.index
    
    # Create records from analysis results
    result_records = []
    for result in results:
        analysis = result['analysis']
        result_records.append({
            'index': result['index'],
            **analysis.model_dump(exclude={'raw_response'})
        })
    
    # Create DataFrame from records
    result_df = pd.DataFrame(result_records)
    
    # Merge with original DataFrame
    final_df = original_df.merge(result_df, on='index', how='left')
    
    return final_df


def process_single_row(
    dialogue: str,
    shiptrack_event_history: Optional[str] = None,
    max_estimated_arrival_date: Optional[str] = None,
    max_retries: int = 5,
    bedrock_client: Optional[boto3.client] = None
) -> BSMAnalysis:
    """
    Process a single row of data using AWS Bedrock.
    
    Args:
        dialogue (str): The dialogue text to analyze
        shiptrack_event_history (Optional[str]): Shipping tracking information
        max_estimated_arrival_date (Optional[str]): Maximum estimated arrival date
        max_retries (int): Maximum number of retry attempts
        bedrock_client (Optional[boto3.client]): Pre-configured Bedrock client
        
    Returns:
        BSMAnalysis: Analysis result
    """
    try:
        # Call Bedrock to analyze the dialogue
        result, latency = analyze_dialogue_with_claude(
            dialogue=dialogue,
            shiptrack=shiptrack_event_history,
            max_estimated_arrival_date=max_estimated_arrival_date,
            max_retries=max_retries,
            bedrock_client=bedrock_client
        )
        
        # Handle error responses
        if isinstance(result, str) and result.startswith("Error"):
            logger.warning(f"Error from Bedrock: {result}")
            return BSMAnalysis(
                category="Error",
                confidence_score=0.0,
                raw_response=result,
                error=result,
                latency=latency
            )
        
        # Parse the response
        analysis = parse_claude_response(result)
        analysis.latency = latency
        return analysis
        
    except Exception as e:
        logger.error(f"Error processing row: {str(e)}")
        return BSMAnalysis(
            category="Error",
            confidence_score=0.0,
            raw_response="",
            error=f"Processing error: {str(e)}"
        )


def process_batch(
    batch_df: pd.DataFrame,
    max_workers: int,
    max_retries: int,
    bedrock_client: boto3.client,
    progress_callback: Optional[Callable[[int, int, int], None]] = None
) -> List[Dict]:
    """
    Process a batch of rows in parallel.
    
    Args:
        batch_df (pd.DataFrame): DataFrame containing the batch to process
        max_workers (int): Maximum number of concurrent threads
        max_retries (int): Maximum number of retry attempts
        bedrock_client (boto3.client): Bedrock client
        progress_callback (Optional[Callable]): Callback function for progress updates
        
    Returns:
        List[Dict]: List of processing results
    """
    batch_results = []
    batch_size = len(batch_df)
    processed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_row = {}
        for idx, row in batch_df.iterrows():
            # Extract required fields with fallbacks for missing columns
            dialogue = row.get('dialogue', '')
            shiptrack = row.get('shiptrack_event_history', None)
            max_date = row.get('max_estimated_arrival_date', None)
            
            future = executor.submit(
                process_single_row,
                dialogue=dialogue,
                shiptrack_event_history=shiptrack,
                max_estimated_arrival_date=max_date,
                max_retries=max_retries,
                bedrock_client=bedrock_client
            )
            future_to_row[future] = idx
        
        # Process futures as they complete
        for future in as_completed(future_to_row):
            idx = future_to_row[future]
            try:
                result = future.result()
                batch_results.append({
                    'index': idx,
                    'analysis': result
                })
            except Exception as e:
                error_msg = f"Error processing row {idx}: {str(e)}"
                logger.error(error_msg)
                batch_results.append({
                    'index': idx,
                    'analysis': BSMAnalysis(
                        category="Error",
                        confidence_score=0.0,
                        raw_response="",
                        error=error_msg
                    )
                })
            finally:
                processed += 1
                if progress_callback:
                    # Call the progress callback with (processed, batch_size, errors)
                    errors = sum(1 for r in batch_results if r['analysis'].error is not None)
                    progress_callback(processed, batch_size, errors)
    
    return batch_results


def batch_process_dataframe(
    df: pd.DataFrame,
    batch_size: int = 10,
    max_workers: int = 5,
    max_retries: int = 5,
    checkpoint_file: Optional[Union[str, Path]] = None,
    bedrock_client: Optional[boto3.client] = None
) -> pd.DataFrame:
    """
    Process DataFrame in batches with parallel execution and progress tracking.
    
    Args:
        df (pd.DataFrame): Input DataFrame with required columns
        batch_size (int): Number of rows to process in each batch
        max_workers (int): Maximum number of concurrent threads
        max_retries (int): Maximum number of retry attempts for each API call
        checkpoint_file (Optional[Union[str, Path]]): Path to save/load checkpoints
        bedrock_client (Optional[boto3.client]): Pre-configured Bedrock client
    
    Returns:
        pd.DataFrame: DataFrame with analysis results
    """
    # Get or create Bedrock client
    if bedrock_client is None:
        bedrock_client = get_bedrock_client()
    
    # Initialize variables
    results = []
    total_rows = len(df)
    processed_rows = 0
    start_batch = 0
    
    # Load checkpoint if provided
    if checkpoint_file:
        processed_rows, results = load_checkpoint(checkpoint_file)
        start_batch = processed_rows // batch_size
        
        # Skip already processed rows
        if processed_rows > 0:
            logger.info(f"Resuming from checkpoint: {processed_rows}/{total_rows} rows already processed")
    
    # Create progress bars
    main_pbar = tqdm(
        total=total_rows,
        initial=processed_rows,
        desc="Overall progress",
        position=0,
        leave=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )
    
    num_batches = (total_rows + batch_size - 1) // batch_size
    batch_pbar = tqdm(
        total=num_batches,
        initial=start_batch,
        desc="Batch progress",
        position=1,
        leave=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}]'
    )
    
    # Define progress callback
    def update_progress(processed: int, batch_size: int, errors: int) -> None:
        main_pbar.update(1)
        main_pbar.set_postfix({
            'batch': f"{processed}/{batch_size}",
            'errors': errors
        })
    
    try:
        # Process in batches
        for batch_start in range(start_batch * batch_size, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_df = df.iloc[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{num_batches} ({len(batch_df)} rows)")
            
            # Process the batch
            batch_results = process_batch(
                batch_df=batch_df,
                max_workers=max_workers,
                max_retries=max_retries,
                bedrock_client=bedrock_client,
                progress_callback=update_progress
            )
            
            # Add results
            results.extend(batch_results)
            batch_pbar.update(1)
            
            # Save checkpoint if provided
            if checkpoint_file:
                processed_rows = batch_end
                save_checkpoint(checkpoint_file, processed_rows, results)
            
            # Add delay between batches to avoid rate limiting
            if batch_end < total_rows:
                time.sleep(1)
        
        # Create result DataFrame
        result_records = []
        for result in results:
            analysis = result['analysis']
            result_records.append({
                'index': result['index'],
                **analysis.model_dump(exclude={'raw_response'})
            })
        
        result_df = pd.DataFrame(result_records)
        
        # Set index if not empty
        if not result_df.empty:
            result_df.set_index('index', inplace=True)
        
        # Calculate final statistics
        total_errors = sum(1 for r in results if r['analysis'].error is not None)
        success_rate = ((total_rows - total_errors) / total_rows) * 100 if total_rows > 0 else 0
        
        # Log final statistics
        logger.info(f"Processing completed: {total_rows} rows, {total_errors} errors, {success_rate:.2f}% success rate")
        
        # Print final statistics
        print(f"\nProcessing completed:")
        print(f"Total rows processed: {total_rows}")
        print(f"Successful: {total_rows - total_errors}")
        print(f"Errors: {total_errors}")
        print(f"Success rate: {success_rate:.2f}%")
        
        # Merge with original DataFrame
        if result_df.empty:
            return df
        else:
            # Ensure indices are unique before concatenation
            if not df.index.is_unique:
                # Create a copy with a unique index
                df_unique = df.copy()
                df_unique.index = range(len(df_unique))
                # Merge results with the unique index DataFrame
                return pd.concat([df_unique, result_df], axis=1)
            else:
                return pd.concat([df, result_df], axis=1)
        
    except Exception as e:
        logger.error(f"Critical error in batch processing: {str(e)}")
        # Save checkpoint on error if provided
        if checkpoint_file:
            save_checkpoint(checkpoint_file, processed_rows, results)
        raise
        
    finally:
        # Close progress bars
        main_pbar.close()
        batch_pbar.close()
