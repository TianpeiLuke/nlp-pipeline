"""
Bedrock Processing CLI

This module provides a command-line interface for processing data with AWS Bedrock,
with support for batch processing, checkpointing, and S3 integration.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Optional, Dict, Any, List, Union

# Import custom modules
from .bedrock_batch_inference import batch_process_dataframe
from .bedrock_batch_process_merge import process_and_merge_results
from .upload_s3 import upload_to_s3


# Set up logging
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV or Parquet file.
    
    Args:
        file_path (str): Path to the input file
        
    Returns:
        pd.DataFrame: Loaded DataFrame
        
    Raises:
        ValueError: If the file format is not supported
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext in ['.parquet', '.pq']:
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def save_summary(df: pd.DataFrame, output_dir: str, s3_bucket: Optional[str] = None) -> str:
    """
    Save a summary of the processing results.
    
    Args:
        df (pd.DataFrame): DataFrame with processing results
        output_dir (str): Directory to save the summary
        s3_bucket (Optional[str]): S3 bucket to upload the summary
        
    Returns:
        str: Path to the saved summary file
    """
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'category': df['category'].value_counts().index,
        'count': df['category'].value_counts().values,
        'avg_confidence': [
            df[df['category'] == cat]['confidence_score'].mean()
            for cat in df['category'].value_counts().index
        ]
    })
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save summary
    summary_path = os.path.join(summary_dir, f"results_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")
    
    # Upload to S3 if bucket is provided
    if s3_bucket:
        s3_prefix = f"llm_processed_data/{datetime.now().strftime('%Y%m%d')}/summary"
        upload_success = upload_to_s3(
            summary_path,
            s3_bucket,
            s3_prefix
        )
        if upload_success:
            logger.info(f"Summary uploaded to s3://{s3_bucket}/{s3_prefix}")
        else:
            logger.warning(f"Failed to upload summary to S3")
    
    return summary_path


def process_with_checkpoint(
    df: pd.DataFrame,
    output_dir: str,
    batch_size: int = 10,
    max_workers: int = 5,
    checkpoint_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Process data with checkpoint support.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        output_dir (str): Directory to save output
        batch_size (int): Number of rows to process in each batch
        max_workers (int): Maximum number of concurrent threads
        checkpoint_file (Optional[str]): Path to checkpoint file
        
    Returns:
        pd.DataFrame: DataFrame with processing results
    """
    # Process data
    result_df = batch_process_dataframe(
        df=df,
        batch_size=batch_size,
        max_workers=max_workers,
        checkpoint_file=checkpoint_file
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'results.csv')
    result_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
    return result_df


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Process data with AWS Bedrock')
    
    # Required arguments
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to input CSV or Parquet file')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory to save output files')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of rows to process in each batch')
    parser.add_argument('--max-workers', type=int, default=5,
                        help='Maximum number of concurrent threads')
    parser.add_argument('--s3-bucket', type=str, default='',
                        help='S3 bucket to upload results (if provided)')
    parser.add_argument('--checkpoint-file', type=str, default='',
                        help='Path to save/load checkpoint file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--partition-cols', type=str, default='',
                        help='Comma-separated list of columns to partition by')
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the Bedrock batch processing CLI.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        # Parse arguments
        args = parse_args()
        
        # Set up logging
        setup_logging(args.log_level)
        
        # Load input data
        df = load_dataframe(args.input_file)
        logger.info(f"Loaded {len(df)} rows from {args.input_file}")
        
        # Parse partition columns
        partition_cols = None
        if args.partition_cols:
            partition_cols = [col.strip() for col in args.partition_cols.split(',')]
            logger.info(f"Using partition columns: {partition_cols}")
        
        # Process data
        if args.checkpoint_file:
            # Use batch processing with checkpoint
            result_df = process_with_checkpoint(
                df=df,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
                checkpoint_file=args.checkpoint_file
            )
        else:
            # Use process_and_merge_results for full workflow
            result_df = process_and_merge_results(
                df=df,
                base_dir=args.output_dir,
                s3_bucket=args.s3_bucket,
                batch_size=args.batch_size,
                partition_cols=partition_cols
            )
        
        # Save summary
        save_summary(result_df, args.output_dir, args.s3_bucket)
        
        # Print summary
        print("\nProcessing Summary:")
        print("-" * 50)
        print(f"Total rows processed: {len(result_df)}")
        print("\nCategory Distribution:")
        print(result_df['category'].value_counts())
        
        logger.info("Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
