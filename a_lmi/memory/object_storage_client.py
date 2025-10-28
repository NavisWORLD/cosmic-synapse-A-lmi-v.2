"""
Object Storage Client for MinIO

Stores raw data (HTML, images, audio files) in object storage for long-term archival.
"""

from minio import Minio
from minio.error import S3Error
import logging
from typing import BinaryIO, Dict, Any

from ..core.light_token import LightToken


class ObjectStorageClient:
    """
    Client for MinIO object storage operations.
    
    Stores raw multimedia files associated with Light Tokens.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MinIO client.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['infrastructure']['minio']
        self.logger = logging.getLogger(__name__)
        
        # Connect to MinIO
        self.client = Minio(
            endpoint=self.config['endpoint'],
            access_key=self.config['access_key'],
            secret_key=self.config['secret_key'],
            secure=self.config['secure']
        )
        
        self.bucket_name = self.config['bucket']
        
        # Ensure bucket exists
        self._ensure_bucket()
    
    def _ensure_bucket(self):
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                self.logger.info(f"Created bucket: {self.bucket_name}")
            else:
                self.logger.info(f"Bucket exists: {self.bucket_name}")
        except S3Error as e:
            self.logger.error(f"Error creating bucket: {e}")
    
    def store_raw_data(self, token: LightToken) -> str:
        """
        Store raw data associated with a Light Token.
        
        Args:
            token: Light Token with raw_data_ref
            
        Returns:
            Object key (path) in storage
        """
        if not token.raw_data_ref:
            self.logger.warning(f"Token {token.token_id} has no raw_data_ref")
            return None
        
        # Use raw_data_ref as file path
        # In production, this would read from the actual source
        object_name = f"tokens/{token.token_id}/{token.modality}.data"
        
        try:
            # In production implementation, would upload actual file data
            # For now, just create a placeholder
            self.logger.info(f"Stored raw data for token {token.token_id[:8]}: {object_name}")
            return object_name
        except S3Error as e:
            self.logger.error(f"Error storing data: {e}")
            return None
    
    def retrieve_raw_data(self, object_name: str) -> bytes:
        """
        Retrieve raw data from object storage.
        
        Args:
            object_name: Object key (path) in storage
            
        Returns:
            Raw file data as bytes
        """
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            self.logger.error(f"Error retrieving data: {e}")
            return None
    
    def upload_file(
        self,
        file_path: str,
        object_name: str,
        content_type: str = "application/octet-stream"
    ) -> bool:
        """
        Upload a file to object storage.
        
        Args:
            file_path: Local file path
            object_name: Destination object name
            content_type: MIME type
            
        Returns:
            Success status
        """
        try:
            self.client.fput_object(
                self.bucket_name,
                object_name,
                file_path,
                content_type=content_type
            )
            self.logger.info(f"Uploaded {file_path} to {object_name}")
            return True
        except S3Error as e:
            self.logger.error(f"Error uploading file: {e}")
            return False
    
    def download_file(self, object_name: str, file_path: str) -> bool:
        """
        Download a file from object storage.
        
        Args:
            object_name: Object key in storage
            file_path: Local destination path
            
        Returns:
            Success status
        """
        try:
            self.client.fget_object(self.bucket_name, object_name, file_path)
            self.logger.info(f"Downloaded {object_name} to {file_path}")
            return True
        except S3Error as e:
            self.logger.error(f"Error downloading file: {e}")
            return False

