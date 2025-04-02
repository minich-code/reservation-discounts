

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from src.ReservationDiscounts.exception import CustomException
from src.ReservationDiscounts.logger import logger
from src.ReservationDiscounts.constants import DATA_INGESTION_CONFIG_FILEPATH, DATA_VALIDATION_CONFIG_FILEPATH
from src.ReservationDiscounts.utils.commons import read_yaml, create_directories
from src.ReservationDiscounts.config_entity.config_params import DataIngestionConfig, DataValidationConfig
load_dotenv()


class ConfigurationManager:
    def __init__(
            self, 
            data_ingestion_config: str = DATA_INGESTION_CONFIG_FILEPATH, 
            data_validation_config: str = DATA_VALIDATION_CONFIG_FILEPATH
            
            
        ):

        try:
            logger.info(f"Initializing ConfigurationManager")
            
            self.ingestion_config = read_yaml(data_ingestion_config)
            self.data_val_config = read_yaml(data_validation_config)
            
            
            create_directories([self.ingestion_config['artifacts_root']])
            create_directories([self.data_val_config['artifacts_root']])
            
            
            logger.info("Configuration directories created successfully.")
        
        except Exception as e:
            logger.error(f"Error initializing ConfigurationManager: {e}")
            raise CustomException(e, sys)
        
# ---------------- Data Ingestion Configuration Manager -------------------
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            data_config = self.ingestion_config['data_ingestion']
            create_directories([data_config['root_dir']])
            logger.info(f"Data ingestion configuration loaded from: {DATA_INGESTION_CONFIG_FILEPATH}")
            mongo_uri = os.environ.get('MONGO_URI')
            return DataIngestionConfig(
                root_dir=data_config['root_dir'],
                database_name=data_config['database_name'],
                collection_name=data_config['collection_name'],
                batch_size=data_config['batch_size'],
                mongo_uri=mongo_uri
            )
        except Exception as e:
            logger.error(f"Error loading data ingestion configuration: {e}")
            raise CustomException(e, sys)
        
# ------------ Data Validation Configuration Manager -------------------------
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            data_validation = self.data_val_config['data_validation']
            create_directories([data_validation['root_dir']])
            schema_path = Path(data_validation['all_schema'])
            
            try:
                all_schema = read_yaml(schema_path)
            except Exception as e:
                logger.error(f"Error reading schema file: {e}")
                raise CustomException(f"Error reading schema file: {e}", sys)

            data_validation_config = DataValidationConfig(
                root_dir=data_validation['root_dir'],
                data_dir=data_validation['data_dir'],
                val_status=data_validation['val_status'],
                all_schema=all_schema,
                validated_data=data_validation['validated_data'],
                profile_report_name=data_validation.get('profile_report_name', "data_profile.html"),
            )
            return data_validation_config
        except Exception as e:
            logger.exception(f"Error getting Data Validation config: {e}")
            raise CustomException(e, sys)

