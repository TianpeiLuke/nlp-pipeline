import pandas as pd
import boto3
from tqdm.auto import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import logging


from .invoke_bedrock import analyze_dialogue_with_claude
from .prompt_rnr_parse import BSMAnalysis, parse_claude_response


# Set up logging
log_dir = os.path.join(os.path.expanduser('~'), 'SageMaker', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'processing_log.txt')

logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def save_checkpoint(checkpoint_file, processed_rows, results):
    """Save checkpoint to a file"""
    checkpoint = {
        'processed_rows': processed_rows,
        'results': [
            {
                'index': r['index'],
                'analysis': r['analysis'].model_dump()
            } for r in results
        ]
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f)

        
def load_checkpoint(checkpoint_file):
    """Load checkpoint from a file"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        results = [
            {
                'index': r['index'],
                'analysis': BSMAnalysis(**r['analysis'])
            } for r in checkpoint['results']
        ]
        return checkpoint['processed_rows'], results
    return 0, []


def create_result_dataframe(results: List[dict], original_df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to create result DataFrame"""
    # Save original index
    original_df = original_df.copy()
    original_df['index'] = original_df.index
    
    result_records = []
    for result in results:
        analysis = result['analysis']
        result_records.append({
            'index': result['index'],
            **analysis.model_dump(exclude={'raw_response'})
        })
    
    result_df = pd.DataFrame(result_records)
    
    # Concatenate
    final_df = original_df.merge(result_df, on='index', how='left')

    
    # Optionally, set back to original index if needed
    # final_df.set_index('original_index', inplace=True)
    
    return final_df


def process_single_row(
    bedrock_client: boto3.client,
    dialogue: str,
    shiptrack: str,
    max_estimated_arrival_date: str,
    max_retries: int = 5
) -> BSMAnalysis:
    """Process a single row of data"""
    result, latency = analyze_dialogue_with_claude(
        dialogue, shiptrack, max_estimated_arrival_date, max_retries
    )
    
    if isinstance(result, str) and result.startswith("Error"):
        return BSMAnalysis(
            category="Error",
            confidence_score=0.0,
            raw_response=result,
            error=result,
            latency=latency
        )
    
    analysis = parse_claude_response(result)
    analysis.latency = latency
    return analysis


def batch_process_dataframe(
    df: pd.DataFrame,
    batch_size: int = 10,
    max_workers: int = 5,
    max_retries: int = 5
) -> pd.DataFrame:
    """
    Process DataFrame in batches with parallel execution and progress tracking
    
    Args:
        df: Input DataFrame with required columns
        batch_size: Number of rows to process in each batch
        max_workers: Maximum number of concurrent threads
        max_retries: Maximum number of retry attempts for each API call
    
    Returns:
        DataFrame with analysis results
    """
    bedrock_client = boto3.client(service_name='bedrock-runtime')
    results = []
    total_rows = len(df)
    processed_rows = 0
    
    # Create main progress bar for overall progress
    main_pbar = tqdm(
        total=total_rows,
        desc="Overall progress",
        position=0,
        leave=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )
    
    # Create batch progress bar
    num_batches = (total_rows + batch_size - 1) // batch_size
    batch_pbar = tqdm(
        total=num_batches,
        desc="Batch progress",
        position=1,
        leave=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}]'
    )
    
    try:
        # Process in batches
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_df = df.iloc[batch_start:batch_end]
            batch_results = []
            batch_size_current = len(batch_df)
            
            # Create progress bar for current batch
            batch_processed = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_row = {
                    executor.submit(
                        process_single_row,
                        bedrock_client,
                        row['dialogue'],
                        row['shiptrack_event_history'],
                        row['max_estimated_arrival_date'],
                        max_retries
                    ): idx for idx, row in batch_df.iterrows()
                }
                
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
                        print(f"\n{error_msg}")
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
                        # Update progress bars
                        batch_processed += 1
                        processed_rows += 1
                        main_pbar.update(1)
                        
                        # Update postfix with current stats
                        main_pbar.set_postfix({
                            'batch': f"{batch_processed}/{batch_size_current}",
                            'errors': sum(1 for r in batch_results if r['analysis'].error is not None)
                        })
            
            results.extend(batch_results)
            batch_pbar.update(1)
            
            # Add delay between batches
            if batch_end < total_rows:
                time.sleep(1)
        
        # Create result DataFrame
        result_records = []
        for result in results:
            analysis = result['analysis']
            result_records.append({
                'index': result['index'],
                **analysis.model_dump(exclude={'raw_response'})  # Updated to use model_dump instead of dict
            })
        
        result_df = pd.DataFrame(result_records)
        result_df.set_index('index', inplace=True)
        
        # Calculate final statistics
        total_errors = sum(1 for r in results if r['analysis'].error is not None)
        success_rate = ((total_rows - total_errors) / total_rows) * 100
        
        # Print final statistics
        print(f"\nProcessing completed:")
        print(f"Total rows processed: {total_rows}")
        print(f"Successful: {total_rows - total_errors}")
        print(f"Errors: {total_errors}")
        print(f"Success rate: {success_rate:.2f}%")
        
        # Merge with original DataFrame
        return pd.concat([df, result_df], axis=1)
        
    except Exception as e:
        print(f"\nCritical error in batch processing: {str(e)}")
        raise
        
    finally:
        # Close progress bars
        main_pbar.close()
        batch_pbar.close()