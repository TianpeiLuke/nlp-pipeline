import os
import pandas as pd
from datetime import datetime


from .bedrock_batch_process_merge import process_and_merge_results
from .upload_s3 import upload_to_s3


if __name__ == 'main':
    try:
        # Set up parameters
        base_dir = f"/home/ec2-user/SageMaker/data/llm_results/"
        s3_bucket = "buyer-seller-messaging-reversal"
        partition_cols = None  #['category', 'org']
    
        # Process and upload
        merged_results = process_and_merge_results(
            df= correct_undelivered_upper_df, #correct_delivered_lower_df.sample(frac=0.1), #tn_mid_risk_df, #correct_not_delivered_lower_df.sample(15000), #correct_upper_df, #delivered_fp_df, #fp_df,#fn_df,
            base_dir=base_dir,
            s3_bucket=s3_bucket,
            batch_size=10,
            partition_cols=partition_cols
        )
    
        # Print summary
        print("\nProcessing Summary:")
        print("-" * 50)
        print(f"Total rows processed: {len(merged_results)}")
        print("\nCategory Distribution:")
        print(merged_results['category'].value_counts())
    
        # Optional: Save local summary
        summary_df = pd.DataFrame({
            'category': merged_results['category'].value_counts().index,
            'count': merged_results['category'].value_counts().values,
            'avg_confidence': [
                merged_results[merged_results['category'] == cat]['confidence_score'].mean()
                for cat in merged_results['category'].value_counts().index
            ]
        })
    
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = os.path.join(base_dir, "summary", f"results_summary_{timestamp}.csv")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
    
        # Upload summary to S3
        upload_to_s3(
            summary_path,
            s3_bucket,
            f"llm_processed_data/{datetime.now().strftime('%Y%m%d')}/summary"
        )

    except Exception as e:
        print(f"Processing failed: {str(e)}")