"""
Example script demonstrating how to use AWS Bedrock Inference Profiles with PydanticAI.
This script shows both ARN-based and Profile ID-based approaches.
"""

import os
import logging
from typing import Dict, Any
from .bedrock_rnr_agent import BedrockRnRAgent
from .rnr_bedrock_main import RnRBedrockProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_with_inference_profile_arn():
    """
    Example using inference profile ARN.
    This is the recommended approach for production use.
    """
    print("=" * 60)
    print("Example 1: Using Inference Profile ARN")
    print("=" * 60)
    
    # Set the inference profile ARN (replace with your actual ARN)
    inference_profile_arn = "arn:aws:bedrock:us-east-1:178936618742:inference-profile/us.anthropic.claude-3-sonnet-20240229-v1:0"
    os.environ['BEDROCK_INFERENCE_PROFILE_ARN'] = inference_profile_arn
    
    # Initialize the agent with Claude 4 model
    agent = BedrockRnRAgent(
        model_id="anthropic.claude-sonnet-4-20250514-v1:0",
        region_name="us-east-1"
    )
    
    # Test case data
    test_case = {
        'dialogue': 'Buyer: My order is late. Seller: Sorry, there was a shipping delay.',
        'shiptrack': 'Package shipped on 2024-01-01, expected delivery 2024-01-05',
        'max_estimated_arrival_date': '2024-01-05'
    }
    
    try:
        result = agent.analyze_rnr_case_sync(
            dialogue=test_case['dialogue'],
            shiptrack=test_case['shiptrack'],
            max_estimated_arrival_date=test_case['max_estimated_arrival_date']
        )
        
        print(f"✅ Analysis successful!")
        print(f"Category: {result.category.value}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Primary factors: {result.reasoning.primary_factors}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your inference profile ARN is correct and you have proper permissions")


def example_with_profile_id():
    """
    Example using inference profile ID directly as model ID.
    This approach uses the global profile ID.
    """
    print("\n" + "=" * 60)
    print("Example 2: Using Inference Profile ID as Model ID")
    print("=" * 60)
    
    # Clear any existing inference profile ARN to test profile ID approach
    if 'BEDROCK_INFERENCE_PROFILE_ARN' in os.environ:
        del os.environ['BEDROCK_INFERENCE_PROFILE_ARN']
    
    # Initialize the agent with the global profile ID
    agent = BedrockRnRAgent(
        model_id="global.anthropic.claude-sonnet-4-20250514-v1:0",  # Use profile ID directly
        region_name="us-east-1"
    )
    
    # Test case data
    test_case = {
        'dialogue': 'Buyer: Item not as described. Seller: We can offer a refund.',
        'shiptrack': 'Package delivered on 2024-01-03',
        'max_estimated_arrival_date': '2024-01-05'
    }
    
    try:
        result = agent.analyze_rnr_case_sync(
            dialogue=test_case['dialogue'],
            shiptrack=test_case['shiptrack'],
            max_estimated_arrival_date=test_case['max_estimated_arrival_date']
        )
        
        print(f"✅ Analysis successful!")
        print(f"Category: {result.category.value}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Primary factors: {result.reasoning.primary_factors}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the profile ID is available in your region")


def example_batch_processing():
    """
    Example of batch processing with inference profiles.
    """
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing with Inference Profiles")
    print("=" * 60)
    
    # Set up inference profile
    inference_profile_arn = "arn:aws:bedrock:us-east-1:178936618742:inference-profile/us.anthropic.claude-3-sonnet-20240229-v1:0"
    os.environ['BEDROCK_INFERENCE_PROFILE_ARN'] = inference_profile_arn
    
    # Initialize processor
    processor = RnRBedrockProcessor(
        model_id="anthropic.claude-sonnet-4-20250514-v1:0",
        region_name="us-east-1",
        output_dir="./output"
    )
    
    # Create sample data
    import pandas as pd
    
    sample_data = [
        {
            'dialogue': 'Buyer: Order never arrived. Seller: Tracking shows delivered.',
            'shiptrack': 'Package marked as delivered on 2024-01-01',
            'max_estimated_arrival_date': '2024-01-01'
        },
        {
            'dialogue': 'Buyer: Wrong item sent. Seller: Please return for exchange.',
            'shiptrack': 'Package delivered on 2024-01-02',
            'max_estimated_arrival_date': '2024-01-02'
        }
    ]
    
    df = pd.DataFrame(sample_data)
    
    try:
        # Process the batch
        results_df = processor.process_batch(
            df=df,
            batch_size=2,
            save_intermediate=False
        )
        
        print(f"✅ Batch processing successful!")
        print(f"Processed {len(results_df)} records")
        print(f"Success rate: {len(results_df[results_df['processing_status'] == 'success']) / len(results_df) * 100:.1f}%")
        
        # Show results
        for idx, row in results_df.iterrows():
            print(f"\nRecord {idx + 1}:")
            print(f"  Category: {row['category']}")
            print(f"  Confidence: {row['confidence_score']:.2f}")
            print(f"  Status: {row['processing_status']}")
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")


def check_environment_setup():
    """
    Check if the environment is properly set up for inference profiles.
    """
    print("=" * 60)
    print("Environment Setup Check")
    print("=" * 60)
    
    # Check AWS credentials
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials:
            print("✅ AWS credentials found")
        else:
            print("❌ AWS credentials not found")
    except Exception as e:
        print(f"❌ AWS credentials check failed: {e}")
    
    # Check inference profile ARN
    inference_profile_arn = os.environ.get('BEDROCK_INFERENCE_PROFILE_ARN')
    if inference_profile_arn:
        print(f"✅ Inference profile ARN set: {inference_profile_arn}")
    else:
        print("⚠️  Inference profile ARN not set (will use profile ID method)")
    
    # Check Bedrock access
    try:
        import boto3
        bedrock = boto3.client('bedrock', region_name='us-east-1')
        models = bedrock.list_foundation_models()
        print(f"✅ Bedrock access confirmed ({len(models['modelSummaries'])} models available)")
    except Exception as e:
        print(f"❌ Bedrock access check failed: {e}")
    
    # Check inference profiles
    try:
        import boto3
        bedrock = boto3.client('bedrock', region_name='us-east-1')
        profiles = bedrock.list_inference_profiles()
        print(f"✅ Inference profiles available: {len(profiles['inferenceProfileSummaries'])}")
        
        # Show available profiles
        for profile in profiles['inferenceProfileSummaries'][:3]:  # Show first 3
            print(f"   - {profile['inferenceProfileName']}: {profile['inferenceProfileArn']}")
        
    except Exception as e:
        print(f"❌ Inference profiles check failed: {e}")


def main():
    """
    Run all examples.
    """
    print("AWS Bedrock Inference Profile Examples")
    print("=====================================")
    
    # Check environment first
    check_environment_setup()
    
    # Run examples
    example_with_inference_profile_arn()
    example_with_profile_id()
    example_batch_processing()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Set your actual inference profile ARN in BEDROCK_INFERENCE_PROFILE_ARN")
    print("2. Ensure you have proper IAM permissions for Bedrock")
    print("3. Test with your own data using the RnRBedrockProcessor class")
    print("\nFor more information, see:")
    print("- docs/setup_inference_profile_external.md")
    print("- docs/setup_inference_profile.md (for Amazon internal users)")


if __name__ == "__main__":
    main()
