"""
Main script for RnR Reason Code classification using PydanticAI and AWS Bedrock.
Compatible with SageMaker environments and inference profiles.
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

from .bedrock_rnr_agent import BedrockRnRPydanticAgent
from .rnr_reason_code_models import RnRReasonCodeAnalysis, RnRCaseData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RnRPydanticBedrockProcessor:
    """
    Main processor for RnR Reason Code classification using PydanticAI and Bedrock.
    Supports both standard models and inference profiles for Claude 4.
    Enhanced with PydanticAI's structured output capabilities.
    """
    
    # Known inference profile configurations
    INFERENCE_PROFILE_CONFIGS = {
        'claude-4-global': {
            'profile_id': 'global.anthropic.claude-sonnet-4-20250514-v1:0',
            'model_id': 'anthropic.claude-sonnet-4-20250514-v1:0',
            'region': 'us-east-1'
        },
        'claude-3-sonnet': {
            'profile_arn': 'arn:aws:bedrock:us-east-1:178936618742:inference-profile/us.anthropic.claude-3-sonnet-20240229-v1:0',
            'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0',
            'region': 'us-east-1'
        }
    }
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name: str = "us-west-2",
        output_dir: str = "/opt/ml/output/data",  # SageMaker default output
        inference_profile_arn: Optional[str] = None,
        use_inference_profile: bool = True,
        **bedrock_kwargs
    ):
        """
        Initialize the RnR PydanticAI Bedrock Processor.
        
        Args:
            model_id: Bedrock model ID to use (supports inference profile IDs like global.anthropic.claude-sonnet-4-20250514-v1:0)
            region_name: AWS region name
            output_dir: Directory to save output files
            inference_profile_arn: Optional inference profile ARN to use
            use_inference_profile: Whether to automatically use inference profiles for Claude 4 models
            **bedrock_kwargs: Additional arguments for Bedrock client
        """
        self.model_id = model_id
        self.region_name = region_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_inference_profile = use_inference_profile
        
        # Configure inference profile
        self._configure_inference_profile(inference_profile_arn)
        
        # Initialize the PydanticAI Bedrock agent
        self.agent = BedrockRnRPydanticAgent(
            model_id=self.effective_model_id,
            region_name=region_name,
            inference_profile_arn=inference_profile_arn,
            use_inference_profile=use_inference_profile,
            **bedrock_kwargs
        )
        
        logger.info(f"Initialized RnR PydanticAI Bedrock Processor with model: {self.effective_model_id} in region: {region_name}")
        if hasattr(self, 'inference_profile_info'):
            logger.info(f"Inference profile configuration: {self.inference_profile_info}")
    
    def _configure_inference_profile(self, inference_profile_arn: Optional[str] = None):
        """
        Configure inference profile settings based on model and environment.
        
        Args:
            inference_profile_arn: Optional inference profile ARN to use
        """
        self.inference_profile_info = {}
        self.effective_model_id = self.model_id
        
        # Check environment variable first
        env_profile_arn = os.environ.get('BEDROCK_INFERENCE_PROFILE_ARN')
        if env_profile_arn:
            inference_profile_arn = env_profile_arn
            logger.info(f"Using inference profile ARN from environment: {inference_profile_arn}")
        
        # Set inference profile ARN if provided
        if inference_profile_arn:
            os.environ['BEDROCK_INFERENCE_PROFILE_ARN'] = inference_profile_arn
            self.inference_profile_info['arn'] = inference_profile_arn
            self.inference_profile_info['method'] = 'arn'
            return
        
        # Auto-configure for known models if use_inference_profile is True
        if self.use_inference_profile:
            if self.model_id == "anthropic.claude-sonnet-4-20250514-v1:0":
                # Use global profile ID for Claude 4
                self.effective_model_id = "global.anthropic.claude-sonnet-4-20250514-v1:0"
                self.inference_profile_info = {
                    'profile_id': 'global.anthropic.claude-sonnet-4-20250514-v1:0',
                    'original_model_id': self.model_id,
                    'method': 'profile_id'
                }
                logger.info(f"Auto-configured to use inference profile ID: {self.effective_model_id}")
                return
            
            elif 'claude-4' in self.model_id or 'claude-sonnet-4' in self.model_id:
                logger.warning(f"Model {self.model_id} may require an inference profile. Consider setting BEDROCK_INFERENCE_PROFILE_ARN or using profile ID.")
        
        # If model already starts with 'global.', it's already a profile ID
        if self.model_id.startswith('global.'):
            self.inference_profile_info = {
                'profile_id': self.model_id,
                'method': 'profile_id'
            }
            logger.info(f"Using provided inference profile ID: {self.model_id}")
    
    def get_inference_profile_config(self, config_name: str) -> Dict[str, str]:
        """
        Get a predefined inference profile configuration.
        
        Args:
            config_name: Name of the configuration ('claude-4-global', 'claude-3-sonnet')
            
        Returns:
            Dictionary with profile configuration
        """
        if config_name not in self.INFERENCE_PROFILE_CONFIGS:
            raise ValueError(f"Unknown configuration: {config_name}. Available: {list(self.INFERENCE_PROFILE_CONFIGS.keys())}")
        
        return self.INFERENCE_PROFILE_CONFIGS[config_name].copy()
    
    def set_inference_profile_arn(self, arn: str):
        """
        Set the inference profile ARN and reconfigure the agent.
        
        Args:
            arn: Inference profile ARN
        """
        os.environ['BEDROCK_INFERENCE_PROFILE_ARN'] = arn
        self.inference_profile_info = {
            'arn': arn,
            'method': 'arn'
        }
        
        # Reinitialize agent with new configuration
        self.agent = BedrockRnRPydanticAgent(
            model_id=self.model_id,
            region_name=self.region_name,
            inference_profile_arn=arn,
            use_inference_profile=True
        )
        
        logger.info(f"Updated inference profile ARN: {arn}")
    
    def set_inference_profile_id(self, profile_id: str):
        """
        Set the inference profile ID and reconfigure the agent.
        
        Args:
            profile_id: Inference profile ID (e.g., 'global.anthropic.claude-sonnet-4-20250514-v1:0')
        """
        # Clear any existing ARN
        if 'BEDROCK_INFERENCE_PROFILE_ARN' in os.environ:
            del os.environ['BEDROCK_INFERENCE_PROFILE_ARN']
        
        self.effective_model_id = profile_id
        self.inference_profile_info = {
            'profile_id': profile_id,
            'original_model_id': self.model_id,
            'method': 'profile_id'
        }
        
        # Reinitialize agent with profile ID
        self.agent = BedrockRnRPydanticAgent(
            model_id=profile_id,
            region_name=self.region_name,
            use_inference_profile=True
        )
        
        logger.info(f"Updated to use inference profile ID: {profile_id}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary with model and inference profile information
        """
        info = {
            'original_model_id': self.model_id,
            'effective_model_id': self.effective_model_id,
            'region_name': self.region_name,
            'use_inference_profile': self.use_inference_profile,
            'inference_profile_info': self.inference_profile_info,
            'framework': 'PydanticAI'
        }
        
        # Add environment variable info
        env_arn = os.environ.get('BEDROCK_INFERENCE_PROFILE_ARN')
        if env_arn:
            info['environment_profile_arn'] = env_arn
        
        return info
    
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
        Process a single RnR case using PydanticAI.
        
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
                'error_message': None,
                'framework': 'PydanticAI'
            }
            
        except Exception as e:
            logger.error(f"Error processing case with PydanticAI: {str(e)}")
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
                'error_message': str(e),
                'framework': 'PydanticAI'
            }
    
    async def process_single_case_async(
        self,
        dialogue: str,
        shiptrack: str,
        max_estimated_arrival_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single RnR case asynchronously using PydanticAI.
        
        Args:
            dialogue: Buyer-seller conversation
            shiptrack: Ship track events
            max_estimated_arrival_date: Estimated delivery date
            
        Returns:
            Dictionary with analysis results
        """
        try:
            result = await self.agent.analyze_rnr_case_async(
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
                'error_message': None,
                'framework': 'PydanticAI'
            }
            
        except Exception as e:
            logger.error(f"Error processing case with PydanticAI async: {str(e)}")
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
                'error_message': str(e),
                'framework': 'PydanticAI'
            }
    
    def process_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 10,
        save_intermediate: bool = True,
        use_async: bool = True
    ) -> pd.DataFrame:
        """
        Process a batch of RnR cases using PydanticAI.
        
        Args:
            df: Input DataFrame
            batch_size: Number of cases to process in each batch
            save_intermediate: Whether to save intermediate results
            use_async: Whether to use async processing for better performance
            
        Returns:
            DataFrame with analysis results
        """
        if use_async:
            return asyncio.run(self._process_batch_async(df, batch_size, save_intermediate))
        else:
            return self._process_batch_sync(df, batch_size, save_intermediate)
    
    def _process_batch_sync(
        self,
        df: pd.DataFrame,
        batch_size: int = 10,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Process a batch of RnR cases synchronously.
        
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
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_df)} records) - Sync")
            
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
                intermediate_file = self.output_dir / f"batch_{batch_num:04d}_results_pydantic.parquet"
                intermediate_df.to_parquet(intermediate_file, index=False)
                logger.info(f"Saved intermediate results to {intermediate_file}")
        
        results_df = pd.DataFrame(results)
        logger.info(f"Completed sync processing {len(results_df)} records with PydanticAI")
        
        return results_df
    
    async def _process_batch_async(
        self,
        df: pd.DataFrame,
        batch_size: int = 10,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Process a batch of RnR cases asynchronously for better performance.
        
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
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_df)} records) - Async")
            
            # Create async tasks for the batch
            tasks = []
            for idx, row in batch_df.iterrows():
                task = self.process_single_case_async(
                    dialogue=row['dialogue'],
                    shiptrack=row['shiptrack'],
                    max_estimated_arrival_date=row.get('max_estimated_arrival_date')
                )
                tasks.append((idx, row, task))
            
            # Execute tasks concurrently
            batch_results = []
            completed_tasks = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
            
            for (idx, row, _), result in zip(tasks, completed_tasks):
                if isinstance(result, Exception):
                    logger.error(f"Error processing case {idx}: {str(result)}")
                    result = {
                        'category': 'Insufficient_Information',
                        'confidence_score': 0.0,
                        'message_evidence': ['[sep] Error processing case'],
                        'shipping_evidence': ['[sep] Error processing case'],
                        'timeline_evidence': ['[sep] Error processing case'],
                        'primary_factors': ['[sep] Processing error occurred'],
                        'supporting_evidence': ['[sep] Unable to analyze due to technical error'],
                        'contradicting_evidence': ['[sep] None'],
                        'formatted_output': f"Error: {str(result)}",
                        'processing_status': 'error',
                        'error_message': str(result),
                        'framework': 'PydanticAI'
                    }
                
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
                intermediate_file = self.output_dir / f"batch_{batch_num:04d}_results_pydantic_async.parquet"
                intermediate_df.to_parquet(intermediate_file, index=False)
                logger.info(f"Saved intermediate results to {intermediate_file}")
        
        results_df = pd.DataFrame(results)
        logger.info(f"Completed async processing {len(results_df)} records with PydanticAI")
        
        return results_df
    
    def save_results(
        self,
        results_df: pd.DataFrame,
        output_filename: str = "rnr_analysis_results_pydantic"
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
            'processing_timestamp': timestamp,
            'framework': 'PydanticAI',
            'model_info': self.get_model_info()
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
        description="RnR Reason Code Classification using PydanticAI and AWS Bedrock"
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
        help="Bedrock model ID to use (supports inference profile IDs like global.anthropic.claude-sonnet-4-20250514-v1:0)"
    )
    
    parser.add_argument(
        "--region-name",
        type=str,
        default="us-west-2",
        help="AWS region name (use us-east-1 for inference profiles)"
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
        default="rnr_analysis_results_pydantic",
        help="Base filename for output files"
    )
    
    parser.add_argument(
        "--use-async",
        action="store_true",
        default=True,
        help="Use async processing for better performance"
    )
    
    parser.add_argument(
        "--inference-profile-arn",
        type=str,
        help="Inference profile ARN to use"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = RnRPydanticBedrockProcessor(
            model_id=args.model_id,
            region_name=args.region_name,
            output_dir=args.output_dir,
            inference_profile_arn=args.inference_profile_arn
        )
        
        # Load and validate data
        df = processor.load_data_from_file(args.input_file)
        df = processor.validate_input_data(df)
        
        # Process data
        results_df = processor.process_batch(
            df=df,
            batch_size=args.batch_size,
            save_intermediate=True,
            use_async=args.use_async
        )
        
        # Save results
        output_files = processor.save_results(
            results_df=results_df,
            output_filename=args.output_filename
        )
        
        # Print summary
        print(f"\nProcessing completed successfully with PydanticAI!")
        print(f"Total records processed: {len(results_df)}")
        print(f"Successful analyses: {len(results_df[results_df['processing_status'] == 'success'])}")
        print(f"Failed analyses: {len(results_df[results_df['processing_status'] == 'error'])}")
        print(f"Framework: PydanticAI")
        print(f"Model info: {processor.get_model_info()}")
        print(f"\nOutput files:")
        for format_name, file_path in output_files.items():
            print(f"  {format_name}: {file_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
