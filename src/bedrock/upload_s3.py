import boto3
from typing import Union, List
from pathlib import Path


def upload_to_s3(
    file_path: Union[str, Path],
    bucket: str,
    s3_prefix: str,
    content_type: str = 'application/octet-stream'
) -> bool:
    """
    Upload a file to S3
    
    Args:
        file_path: Local file path
        bucket: S3 bucket name
        s3_prefix: S3 object prefix (folder structure)
        content_type: File content type
    """
    try:
        s3_client = boto3.client('s3')
        file_path = Path(file_path)
        
        # Create S3 key (path)
        s3_key = f"{s3_prefix}/{file_path.name}"
        
        # Upload file
        s3_client.upload_file(
            str(file_path),
            bucket,
            s3_key,
            ExtraArgs={'ContentType': content_type}
        )
        
        print(f"Successfully uploaded {file_path.name} to s3://{bucket}/{s3_key}")
        return True
        
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        return False
    
    
def upload_directory_to_s3(
    local_dir: Union[str, Path],
    bucket: str,
    s3_prefix: str
) -> List[str]:
    """
    Upload all files in a directory to S3
    
    Args:
        local_dir: Local directory path
        bucket: S3 bucket name
        s3_prefix: S3 object prefix (folder structure)
    """
    uploaded_files = []
    local_dir = Path(local_dir)
    
    try:
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                # Create relative path for S3
                relative_path = file_path.relative_to(local_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                
                # Upload file
                if upload_to_s3(file_path, bucket, s3_prefix):
                    uploaded_files.append(s3_key)
                    
        return uploaded_files
        
    except Exception as e:
        print(f"Error uploading directory to S3: {str(e)}")
        return uploaded_files