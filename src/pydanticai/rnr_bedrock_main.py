"""
Main script for RnR Reason Code classification using Pydantic AI and AWS Bedrock.
Compatible with SageMaker environments.
"""

import os
import json
import argparse
import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from .bedrock_rnr_agent import BedrockRnRAgent
from .rnr_reason_code_models import RnRReasonCodeAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RnRBedrockProcessor:
    """
    Main processor for RnR Reason Code classification using Bedrock.
    """
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name: str = "us-west-2",
        output_dir: str = "/opt/ml/output/data",  # SageMaker default output
        **bedrock_kwargs
    ):
        """
        Initialize the RnR Bedrock Processor.
        
        Args:
            model_id: Bedrock model ID to use
            region_name: AWS region name
            output_dir: Directory to save output files
            **bedrock_kwargs: Additional arguments for Bedrock client
        """
        self.model_id = model_id
        self.region_name = region_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the Bedrock agent
        self.agent = BedrockRnRAgent(
            model_id=model_id,
            region_name=region_name,
            **bedrock_kwargs
        )
        
        logger.info(f"Initialized RnR Bedrock Processor with model: {model_id}")
    
    def load_data_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            DataFrame with the loaded data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path)
        elif file_path.suffix.lower() == '.jsonl':
            df = pd.read_json(file_path, lines=True)
        elif file_path.suffix.lower() in ['.parquet', '.pq']:
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
    
    def validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare input data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated DataFrame
        """
        required_columns = ['dialogue', 'shiptrack']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Fill missing values
        df['dialogue'] = df['dialogue'].fillna('')
        df['shiptrack'] = df['shiptrack'].fillna('')
        df['max_estimated_arrival_date'] = df.get('max_estimated_arrival_date', '').fillna('')
        
        # Convert empty strings to None for max_estimated_arrival_date
        df['max_estimated_arrival_date'] = df['max_estimated_arrival_date'].replace('', None)
        
        logger.info(f"Validated input data with {len(df)} records")
        return df
    
    def process_single_case(
        self,
        dialogue: str,
        shiptrack: str,
        max_estimated_arrival_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single RnR case.
        
        Args:
            dialogue: Buyer-seller conversation
            shiptrack: Ship track events
            max_estimated_arrival_date: Estimated delivery date
            
        Returns:
            Dictionary with analysis results
        """
        try:
            result = self.agent.analyze_rnr_case_sync(
                dialogue=dialogue,
                shiptrack=shiptrack,
                max_estimated_arrival_date=max_estimated_arrival_date
            )
            
            return {
                'category': result.category.value,
                'confidence_score': result.confidence_score,
                'message_evidence': result.key_evidence.message_evidence,
                'shipping_evidence': result.key_evidence.shipping_evidence,
                'timeline_evidence': result.key_evidence.timeline_evidence,
                'primary_factors': result.reasoning.primary_factors,
                'supporting_evidence': result.reasoning.supporting_evidence,
                'contradicting_evidence': result.reasoning.contradicting_evidence,
                'formatted_output': result.to_formatted_output(),
                'processing_status': 'success',
                'error_message': None
            }
            
        except Exception as e:
            logger.error(f"Error processing case: {str(e)}")
            return {
                'category': 'Insufficient_Information',
                'confidence_score': 0.0,
                'message_evidence': ['[sep] Error processing case'],
                'shipping_evidence': ['[sep] Error processing case'],
                'timeline_evidence': ['[sep] Error processing case'],
                'primary_factors': ['[sep] Processing error occurred'],
                'supporting_evidence': ['[sep] Unable to analyze due to technical error'],
                'contradicting_evidence': ['[sep] None'],
                'formatted_output': f"Error: {str(e)}",
                'processing_status': 'error',
                'error_message': str(e)
            }
    
    def process_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 10,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Process a batch of RnR cases.
        
        Args:
            df: Input DataFrame
            batch_size: Number of cases to process in each batch
            save_intermediate: Whether to save intermediate results
            
        Returns:
            DataFrame with analysis results
        """
        results = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size].copy()
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_df)} records)")
            
            batch_results = []
            for idx, row in batch_df.iterrows():
                result = self.process_single_case(
                    dialogue=row['dialogue'],
                    shiptrack=row['shiptrack'],
                    max_estimated_arrival_date=row.get('max_estimated_arrival_date')
                )
                
                # Add original row data
                result.update({
                    'original_index': idx,
                    'dialogue': row['dialogue'],
                    'shiptrack': row['shiptrack'],
                    'max_estimated_arrival_date': row.get('max_estimated_arrival_date')
                })
                
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Save intermediate results
            if save_intermediate:
                intermediate_df = pd.DataFrame(batch_results)
                intermediate_file = self.output_dir / f"batch_{batch_num:04d}_results.parquet"
                intermediate_df.to_parquet(intermediate_file, index=False)
                logger.info(f"Saved intermediate results to {intermediate_file}")
        
        results_df = pd.DataFrame(results)
        logger.info(f"Completed processing {len(results_df)} records")
        
        return results_df
    
    def save_results(
        self,
        results_df: pd.DataFrame,
        output_filename: str = "rnr_analysis_results"
    ) -> Dict[str, Path]:
        """
        Save results in multiple formats.
        
        Args:
            results_df: DataFrame with analysis results
            output_filename: Base filename for output files
            
        Returns:
            Dictionary mapping format to file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{output_filename}_{timestamp}"
        
        output_files = {}
        
        # Save as Parquet (efficient for large datasets)
        parquet_file = self.output_dir / f"{base_filename}.parquet"
        results_df.to_parquet(parquet_file, index=False)
        output_files['parquet'] = parquet_file
        
        # Save as CSV (human-readable)
        csv_file = self.output_dir / f"{base_filename}.csv"
        results_df.to_csv(csv_file, index=False)
        output_files['csv'] = csv_file
        
        # Save as JSON Lines (for streaming processing)
        jsonl_file = self.output_dir / f"{base_filename}.jsonl"
        results_df.to_json(jsonl_file, orient='records', lines=True)
        output_files['jsonl'] = jsonl_file
        
        # Save summary statistics
        summary_stats = {
            'total_records': len(results_df),
            'successful_analyses': len(results_df[results_df['processing_status'] == 'success']),
            'failed_analyses': len(results_df[results_df['processing_status'] == 'error']),
            'category_distribution': results_df['category'].value_counts().to_dict(),
            'average_confidence': results_df[results_df['processing_status'] == 'success']['confidence_score'].mean(),
            'processing_timestamp': timestamp
        }
        
        summary_file = self.output_dir / f"{base_filename}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        output_files['summary'] = summary_file
        
        logger.info(f"Saved results to {len(output_files)} files in {self.output_dir}")
        return output_files


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="RnR Reason Code Classification using Pydantic AI and AWS Bedrock"
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input file (CSV, JSON, JSONL, or Parquet)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/ml/output/data",
        help="Output directory for results (default: SageMaker output dir)"
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        help="Bedrock model ID to use"
    )
    
    parser.add_argument(
        "--region-name",
        type=str,
        default="us-west-2",
        help="AWS region name"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "--output-filename",
        type=str,
        default="rnr_analysis_results",
        help="Base filename for output files"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = RnRBedrockProcessor(
            model_id=args.model_id,
            region_name=args.region_name,
            output_dir=args.output_dir
        )
        
        # Load and validate data
        df = processor.load_data_from_file(args.input_file)
        df = processor.validate_input_data(df)
        
        # Process data
        results_df = processor.process_batch(
            df=df,
            batch_size=args.batch_size,
            save_intermediate=True
        )
        
        # Save results
        output_files = processor.save_results(
            results_df=results_df,
            output_filename=args.output_filename
        )
        
        # Print summary
        print(f"\nProcessing completed successfully!")
        print(f"Total records processed: {len(results_df)}")
        print(f"Successful analyses: {len(results_df[results_df['processing_status'] == 'success'])}")
        print(f"Failed analyses: {len(results_df[results_df['processing_status'] == 'error'])}")
        print(f"\nOutput files:")
        for format_name, file_path in output_files.items():
            print(f"  {format_name}: {file_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
