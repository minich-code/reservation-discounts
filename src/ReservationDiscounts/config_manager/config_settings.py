

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from src.ReservationDiscounts.exception import CustomException
from src.ReservationDiscounts.logger import logger
from src.ReservationDiscounts.constants import *
from src.ReservationDiscounts.utils.commons import read_yaml, create_directories
from src.ReservationDiscounts.config_entity.config_params import (DataIngestionConfig, DataValidationConfig, DataDriftConfig,
                                                                   DataTransformationConfig)
load_dotenv()


class ConfigurationManager:
    def __init__(
            self, 
            data_ingestion_config: str = DATA_INGESTION_CONFIG_FILEPATH, 
            data_validation_config: str = DATA_VALIDATION_CONFIG_FILEPATH,
            data_drift_config: str = DATA_DRIFT_CONFIG_FILEPATH,
            data_transformation_config: str = DATA_TRANSFORMATION_CONFIG_FILEPATH 
            
            
        ):

        try:
            logger.info(f"Initializing ConfigurationManager")
            
            self.ingestion_config = read_yaml(data_ingestion_config)
            self.data_val_config = read_yaml(data_validation_config)
            self.data_drift = read_yaml(data_drift_config)
            self.preprocessing_config = read_yaml(data_transformation_config)
            
            
            create_directories([self.ingestion_config['artifacts_root']])
            create_directories([self.data_val_config['artifacts_root']])
            create_directories([self.data_drift['artifacts_root']])
            create_directories([self.preprocessing_config['artifacts_root']])
            
            
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
        
# ------------ Data Drift Configuration Manager -------------------------
    def get_data_drift_config(self) -> DataDriftConfig:
        try:
            drift_config = self.data_drift['data_drift']
            create_directories([drift_config['root_dir']])

            return DataDriftConfig(
                root_dir = drift_config['root_dir'],
                data_path = drift_config['data_path'],
                random_state = drift_config['random_state'],
                target_col = drift_config['target_col'],
                numerical_cols = drift_config['numerical_cols'],
                categorical_cols = drift_config['categorical_cols']
            )

        except Exception as e:
            logger.exception(f"Error getting the Data Drift Config")
            raise CustomException(e, sys)
        
# ------------ Data Transformation Configuration Manager------------------------
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            transformation_config = self.preprocessing_config['data_transformation']
            create_directories([transformation_config['root_dir']])

            return DataTransformationConfig(
                root_dir = Path(transformation_config['root_dir']),
                training_features = Path(transformation_config['training_features']),
                test_features = Path(transformation_config['test_features']),
                validation_features = Path(transformation_config['validation_features']),
                training_target = Path(transformation_config['training_target']),
                test_target = Path(transformation_config['test_target']),
                validation_target = Path(transformation_config['validation_target']),
                numerical_cols = transformation_config['numerical_cols'],
                categorical_cols = transformation_config['categorical_cols']
            )
        except Exception as e:
            raise CustomException(e, sys)


