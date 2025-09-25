"""
S3 Upload Module

This module provides functions to upload files and directories to Amazon S3.
It handles path normalization, error handling, and provides status reporting.
"""

import boto3
import logging
from typing import Union, List, Optional, Dict, Any
from pathlib import Path
from botocore.exceptions import ClientError


# Set up logging
logger = logging.getLogger(__name__)


def get_s3_client() -> boto3.client:
    """
    Create and return a boto3 client for AWS S3.
    
    Returns:
        boto3.client: Configured S3 client.
    """
    return boto3.client('s3')


def upload_to_s3(
    file_path: Union[str, Path],
    bucket: str,
    s3_prefix: str,
    content_type: str = 'application/octet-stream',
    extra_args: Optional[Dict[str, Any]] = None,
    s3_client: Optional[boto3.client] = None
) -> bool:
    """
    Upload a file to S3 with improved error handling.
    
    Args:
        file_path (Union[str, Path]): Local file path
        bucket (str): S3 bucket name
        s3_prefix (str): S3 object prefix (folder structure)
        content_type (str, optional): File content type. Defaults to 'application/octet-stream'.
        extra_args (Optional[Dict[str, Any]], optional): Additional arguments for S3 upload.
        s3_client (Optional[boto3.client], optional): Pre-configured S3 client (for testing).
        
    Returns:
        bool: True if upload was successful, False otherwise.
    """
    if s3_client is None:
        s3_client = get_s3_client()
        
    try:
        # Convert to Path object if it's a string
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
            
        if not file_path.is_file():
            logger.error(f"Not a file: {file_path}")
            return False
        
        # Create S3 key (path)
        s3_key = f"{s3_prefix.rstrip('/')}/{file_path.name}"
        
        # Prepare upload arguments
        upload_args = {'ContentType': content_type}
        if extra_args:
            upload_args.update(extra_args)
        
        # Upload file
        s3_client.upload_file(
            str(file_path),
            bucket,
            s3_key,
            ExtraArgs=upload_args
        )
        
        logger.info(f"Successfully uploaded {file_path.name} to s3://{bucket}/{s3_key}")
        return True
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"S3 client error ({error_code}) uploading {file_path.name}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error uploading {file_path.name} to S3: {str(e)}")
        return False
    
    
def upload_directory_to_s3(
    local_dir: Union[str, Path],
    bucket: str,
    s3_prefix: str,
    content_type_mapping: Optional[Dict[str, str]] = None,
    s3_client: Optional[boto3.client] = None
) -> List[str]:
    """
    Upload all files in a directory to S3 with improved path handling.
    
    Args:
        local_dir (Union[str, Path]): Local directory path
        bucket (str): S3 bucket name
        s3_prefix (str): S3 object prefix (folder structure)
        content_type_mapping (Optional[Dict[str, str]], optional): 
            Mapping of file extensions to content types.
        s3_client (Optional[boto3.client], optional): Pre-configured S3 client (for testing).
        
    Returns:
        List[str]: List of successfully uploaded S3 keys.
    """
    uploaded_files = []
    
    if s3_client is None:
        s3_client = get_s3_client()
    
    try:
        local_dir = Path(local_dir)
        
        if not local_dir.exists():
            logger.error(f"Directory not found: {local_dir}")
            return uploaded_files
            
        if not local_dir.is_dir():
            logger.error(f"Not a directory: {local_dir}")
            return uploaded_files
        
        # Default content type mapping if none provided
        if content_type_mapping is None:
            content_type_mapping = {
                '.html': 'text/html',
                '.css': 'text/css',
                '.js': 'application/javascript',
                '.json': 'application/json',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.pdf': 'application/pdf',
                '.txt': 'text/plain',
                '.csv': 'text/csv',
                '.parquet': 'application/octet-stream'
            }
        
        # Normalize s3_prefix to ensure it doesn't have trailing slash
        s3_prefix = s3_prefix.rstrip('/')
        
        # Count total files for progress reporting
        total_files = sum(1 for _ in local_dir.rglob('*') if _.is_file())
        logger.info(f"Found {total_files} files to upload in {local_dir}")
        
        processed = 0
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                processed += 1
                
                # Determine content type based on file extension
                content_type = content_type_mapping.get(
                    file_path.suffix.lower(), 
                    'application/octet-stream'
                )
                
                # Create relative path for S3
                relative_path = file_path.relative_to(local_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                
                # Log progress periodically
                if processed % 10 == 0 or processed == total_files:
                    logger.info(f"Uploading file {processed}/{total_files}: {relative_path}")
                
                # Upload file with appropriate content type
                if upload_to_s3(
                    file_path=file_path,
                    bucket=bucket,
                    s3_prefix=f"{s3_prefix}/{relative_path.parent}",
                    content_type=content_type,
                    s3_client=s3_client
                ):
                    uploaded_files.append(s3_key)
        
        logger.info(f"Successfully uploaded {len(uploaded_files)}/{total_files} files to S3")
        return uploaded_files
        
    except Exception as e:
        logger.error(f"Error uploading directory to S3: {str(e)}")
        return uploaded_files
