import sys
import logging
from pathlib import Path
import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession

from src.pipeline_builder.template_pipeline_xgboost_train_evaluate_e2e import XGBoostTrainEvaluateE2ETemplateBuilder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to demonstrate the template-based XGBoost Train-Evaluate E2E pipeline.
    
    This example shows how to use the template-based implementation to create the same
    pipeline as the original builder_pipeline_xgboost_train_evaluate_e2e.py, but using
    the PipelineBuilderTemplate to handle the connections between steps.
    """
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python builder_pipeline_xgboost_train_evaluate_e2e_template.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    logger.info(f"Using config path: {config_path}")
    
    # Setup SageMaker session
    boto_session = boto3.Session()
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    pipeline_session = PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_session.sagemaker_client,
        default_bucket=sagemaker_session.default_bucket(),
    )
    
    # Get SageMaker execution role
    role = sagemaker.get_execution_role()
    
    # Create the pipeline builder
    builder = XGBoostTrainEvaluateE2ETemplateBuilder(
        config_path=config_path,
        sagemaker_session=pipeline_session,
        role=role,
        notebook_root=Path.cwd(),
    )
    
    # Generate the pipeline
    pipeline = builder.generate_pipeline()
    
    # Print pipeline definition
    logger.info(f"Pipeline definition: {pipeline.definition()}")
    
    # Optionally, create or update and execute the pipeline
    pipeline.upsert(role_arn=role)
    logger.info(f"Pipeline '{pipeline.name}' created/updated successfully")
    
    # Uncomment to execute the pipeline
    # execution = pipeline.start()
    # logger.info(f"Pipeline execution started with ARN: {execution.arn}")

if __name__ == "__main__":
    main()
