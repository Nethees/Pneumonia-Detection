import os
import sys
from Xray.exception import XRayException
from Xray.logger import logging

class S3operation:
    
    def sync_folder_to_s3(self, folder: str, bucket_name: str, bucket_folder_name: str)-> None: # Uploading
        try:
            command: str = (
                
                f" aws s3 sync {folder} s3://{bucket_name}/{bucket_folder_name}" 
            )
            print(command)
            os.system(command)
        except Exception as e:
            raise XRayException(e, sys)
        
    def sync_folder_from_s3(self, folder: str, bucket_name: str, bucket_folder_name: str)-> None: # Downloading
        try:
            logging.info("Enters sync_folder_from_S3")
            command: str = (
                
                f"aws s3 sync s3://{bucket_name}/{bucket_folder_name}/ {folder} " 
            )
            print(command)
            os.system(command)
        except Exception as e:
            raise XRayException(e, sys)