from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from typing import List
from tqdm import tqdm
import os


from .upload_s3 import upload_to_s3
from .bedrock_batch_inference import batch_process_dataframe


def save_batch_results(
    df: pd.DataFrame,
    base_dir: str,
    batch_id: str,
    partition_cols: List[str] = None,
    compression: str = 'snappy'
) -> str:
    """
    Save batch results to parquet with proper handling of complex types
    
    Args:
        df: DataFrame to save
        base_dir: Base directory for saving results
        batch_id: Identifier for this batch
        partition_cols: Columns to partition by
        compression: Compression codec to use
    """
    try:
        save_path = Path(base_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"batch_{batch_id}_{timestamp}.parquet"
        final_path = save_path / filename
        
        # Convert list columns to strings
        df = df.copy()
        for col in df.select_dtypes(include=['object']).columns:
            if isinstance(df[col].iloc[0], list):
                df[col] = df[col].apply(lambda x: ';'.join(str(item) for item in x) if isinstance(x, list) else x)
        
        # Save to parquet
        if partition_cols:
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(
                table,
                root_path=str(final_path).replace('.parquet', ''),
                partition_cols=partition_cols,
                compression=compression
            )
        else:
            df.to_parquet(
                final_path,
                compression=compression,
                index=True
            )
        
        return str(final_path)
    
    except Exception as e:
        print(f"Error saving batch {batch_id}: {str(e)}")
        raise
        

        
def merge_batch_results(
    base_dir: str,
    output_dir: str,
    pattern: str = "batch_*.parquet",
    exclude_patterns: List[str] = None,
    partition_cols: List[str] = None,
    compression: str = 'snappy'
) -> pd.DataFrame:
    """
    Merge all batch results into a single parquet file
    """
    try:
        input_path = Path(base_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get list of batch files - search recursively and handle both file and directory cases
        files = []
        # Look for direct parquet files
        files.extend(list(input_path.glob(pattern)))
        # Look for partitioned directories
        files.extend(list(input_path.glob("batch_*")))
        
        if not files:
            print(f"Searching in: {input_path}")
            print(f"Current files in directory: {list(input_path.iterdir())}")
            raise ValueError(f"No files found matching pattern '{pattern}' in {input_path}")
            
        if exclude_patterns:
            for pattern in exclude_patterns:
                files = [f for f in files if pattern not in f.name]
        
        total_files = len(files)
        print(f"Found {total_files} batch files to merge")
        
        # Initialize progress bars
        pbar = tqdm(total=total_files, desc="Merging batches")
        
        # Process files
        dfs = []
        total_rows = 0
        errors = []
        
        for file in files:
            try:
                if file.is_file():
                    df = pd.read_parquet(file)
                else:  # Handle partitioned directories
                    df = pd.read_parquet(file)
                dfs.append(df)
                total_rows += len(df)
                pbar.update(1)
                print(f"Successfully read {file} with {len(df)} rows")
                
            except Exception as e:
                error_msg = f"Error reading {file}: {str(e)}"
                print(f"\nWarning: {error_msg}")
                errors.append(error_msg)
        
        if not dfs:
            raise ValueError("No data frames were successfully loaded")
            
        # Merge all DataFrames
        print("\nConcatenating results...")
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Save merged results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_path = output_path / f"merged_results_{timestamp}.parquet"
        
        print("Saving merged results...")
        if partition_cols:
            table = pa.Table.from_pandas(merged_df)
            pq.write_to_dataset(
                table,
                root_path=str(final_path).replace('.parquet', ''),
                partition_cols=partition_cols,
                compression=compression
            )
        else:
            merged_df.to_parquet(
                final_path,
                compression=compression,
                index=True
            )
        
        # Rest of the function remains the same...
        return merged_df
        
    except Exception as e:
        print(f"Critical error during merge: {str(e)}")
        raise
    finally:
        pbar.close()


def process_and_merge_results(
    df: pd.DataFrame,
    base_dir: str,
    s3_bucket: str,
    batch_size: int = 10,
    partition_cols: List[str] = None
) -> pd.DataFrame:
    """Complete workflow with improved error handling"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        current_date = datetime.now().strftime('%Y%m%d')
        
        
        # CHANGE 1: Normalize base directory path
        base_dir = os.path.abspath(os.path.expanduser(base_dir))
        
        # Local directories
        batch_dir = os.path.join(base_dir, "batches", timestamp)
        merged_dir = os.path.join(base_dir, "merged", timestamp)
        
        # CHANGE 2: Add stats directory
        stats_dir = os.path.join(base_dir, "stats", timestamp)        
        
        # S3 prefixes
        s3_batch_prefix = f"llm_processed_data/{current_date}/batches/{timestamp}"
        s3_merged_prefix = f"llm_processed_data/{current_date}/merged/{timestamp}"
        # CHANGE 3: Add stats prefix
        s3_stats_prefix = f"llm_processed_data/{current_date}/stats/{timestamp}"
        
        
        # CHANGE 4: Create all directories at once
        for dir_path in [batch_dir, merged_dir, stats_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # CHANGE 5: Add informative print statements
        print(f"\nProcessing {len(df)} rows in batches of {batch_size}")
        print(f"Using directories:")
        print(f"Batch dir: {batch_dir}")
        print(f"Merged dir: {merged_dir}")
            
        # Process batches
        processed_df = batch_process_dataframe(
            df=df,
            batch_size=batch_size,
            max_workers=5,
            max_retries=5
        )
        
        # Save and upload batch results
        batch_path = save_batch_results(
            df=processed_df,
            base_dir=batch_dir,
            batch_id=current_date,
            partition_cols=partition_cols
        )
        
        # CHANGE 6: Add file existence check and error handling
        if os.path.exists(batch_path):
            print(f"Batch file created successfully at: {batch_path}")
            
            if s3_bucket:
                print("\nUploading batch results to S3...")
                upload_success = upload_to_s3(
                    batch_path,
                    s3_bucket,
                    s3_batch_prefix
                )
                if upload_success:
                    print("Batch upload completed successfully")
                else:
                    print("Warning: Batch upload failed")
        else:
            print(f"Warning: Expected batch file not found at {batch_path}")
        
        # CHANGE 7: Add progress message
        print("\nMerging results...")
        merged_df = merge_batch_results(
            base_dir=batch_dir,
            output_dir=merged_dir,
            pattern="batch_*.parquet",  # CHANGE 8: Add explicit pattern
            partition_cols=partition_cols
        )
        
        # CHANGE 9: Improved S3 upload handling
        if s3_bucket:
            print("\nUploading merged results to S3...")
            merged_files = list(Path(merged_dir).glob("*.parquet"))
            print(f"Found {len(merged_files)} files to upload")
            
            for file in merged_files:
                if os.path.exists(file):
                    upload_to_s3(
                        file,
                        s3_bucket,
                        s3_merged_prefix
                    )
                else:
                    print(f"Warning: Merged file not found: {file}")
        
        # CHANGE 10: Move summary creation here and add error handling
        summary_df = pd.DataFrame({
            'category': merged_df['category'].value_counts().index,
            'count': merged_df['category'].value_counts().values,
            'avg_confidence': [
                merged_df[merged_df['category'] == cat]['confidence_score'].mean()
                for cat in merged_df['category'].value_counts().index
            ]
        })
        
        # CHANGE 11: Save summary to stats directory
        summary_path = os.path.join(stats_dir, f"results_summary_{timestamp}.csv")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        
        
        # CHANGE 12: Upload summary with better error handling
        if s3_bucket and os.path.exists(summary_path):
            upload_to_s3(
                summary_path,
                s3_bucket,
                f"llm_processed_data/{current_date}/summary"
            )
        
        # CHANGE 13: Enhanced summary printing
        print("\nProcessing Summary:")
        print("-" * 50)
        print(f"Total rows processed: {len(merged_df)}")
        print("\nCategory Distribution:")
        print(merged_df['category'].value_counts())
        print(f"\nResults saved to:")
        print(f"- Local: {merged_dir}")
        print(f"- S3: s3://{s3_bucket}/{s3_merged_prefix}")
        
        return merged_df
        
    except Exception as e:
        print(f"Error in processing workflow: {str(e)}")
        raise