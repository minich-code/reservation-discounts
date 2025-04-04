

# import os 
# import sys 
# import numpy as np 
# import pandas as pd 
# import joblib 

# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer

# from pathlib import Path 
# from dataclasses import dataclass 
# from typing import List 

# from src.ReservationDiscounts.exception import CustomException
# from src.ReservationDiscounts.logger import logger
# from src.ReservationDiscounts.utils.commons import *
# from src.ReservationDiscounts.constants import *

# @dataclass
# class DataTransformationConfig:
#     root_dir: Path
#     training_features: Path
#     test_features: Path
#     validation_features: Path
#     training_target: Path
#     test_target: Path
#     validation_target: Path
#     numerical_cols: List[str]
#     categorical_cols: List[str]

# class ConfigurationManager:
#     def __init__(self, data_transformation_config: str = DATA_TRANSFORMATION_CONFIG_FILEPATH):
#         try:
#             self.preprocessing_config = read_yaml(data_transformation_config)
#             create_directories([self.preprocessing_config['artifacts_root']])
#         except Exception as e:
#             raise CustomException(e, sys)

#     def get_data_transformation_config(self) -> DataTransformationConfig:
#         try:
#             transformation_config = self.preprocessing_config['data_transformation']
#             create_directories([transformation_config['root_dir']])

#             return DataTransformationConfig(
#                 root_dir = Path(transformation_config['root_dir']),
#                 training_features = Path(transformation_config['training_features']),
#                 test_features = Path(transformation_config['test_features']),
#                 validation_features = Path(transformation_config['validation_features']),
#                 training_target = Path(transformation_config['training_target']),
#                 test_target = Path(transformation_config['test_target']),
#                 validation_target = Path(transformation_config['validation_target']),
#                 numerical_cols = transformation_config['numerical_cols'],
#                 categorical_cols = transformation_config['categorical_cols']
#             )
#         except Exception as e:
#             raise CustomException(e, sys)

# class DataTransformation:
#     def __init__(self, config: DataTransformationConfig):
#         self.config = config

#     def get_transformer_object(self) -> ColumnTransformer:
#         logger.info("Getting transformer object")

#         try:
#             numerical_transformer = Pipeline(steps=[
#                 ('imputer', SimpleImputer(strategy='median')),
#                 ('scaler', StandardScaler())
#             ])

#             categorical_transformer = Pipeline(steps=[
#                 ('imputer', SimpleImputer(strategy='most_frequent')),
#                 ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
#             ])

#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     ('num', numerical_transformer, self.config.numerical_cols),
#                     ('cat', categorical_transformer, self.config.categorical_cols)
#                 ]
#             )

#             return preprocessor

#         except Exception as e:
#             raise CustomException(e, sys)

#     def initiate_data_transformation(self):
#         logger.info("Initiating data transformation")

#         try:
#             # Load data from Parquet files
#             logger.info("Loading datasets...")
#             X_train = pd.read_parquet(self.config.training_features)
#             X_test = pd.read_parquet(self.config.test_features)
#             X_val = pd.read_parquet(self.config.validation_features)

#             # Get the preprocessor object
#             preprocessor_obj = self.get_transformer_object()

#             # Transform the training, testing, and validation data
#             logger.info("Fitting and transforming training data...")
#             X_train_transformed = preprocessor_obj.fit_transform(X_train)

#             logger.info("Transforming test and validation data...")
#             X_test_transformed = preprocessor_obj.transform(X_test)
#             X_val_transformed = preprocessor_obj.transform(X_val)

#             # Convert transformed arrays back to DataFrames
#             X_train_transformed_df = pd.DataFrame(X_train_transformed)
#             X_test_transformed_df = pd.DataFrame(X_test_transformed)
#             X_val_transformed_df = pd.DataFrame(X_val_transformed)

#             # Save the transformed data as Parquet files
#             transformed_dir = self.config.root_dir / "transformed_data"
#             transformed_dir.mkdir(parents=True, exist_ok=True)

#             X_train_transformed_df.to_parquet(transformed_dir / "X_train_transformed.parquet")
#             X_test_transformed_df.to_parquet(transformed_dir / "X_test_transformed.parquet")
#             X_val_transformed_df.to_parquet(transformed_dir / "X_val_transformed.parquet")

#             logger.info(f"Transformed datasets saved in {transformed_dir}")

#             # Save the preprocessor object to artifacts 
#             preprocessor_path = self.config.root_dir / 'preprocessor.joblib'
#             joblib.dump(preprocessor_obj, preprocessor_path)
#             logger.info(f"Preprocessor object saved at: {preprocessor_path}")

#         except Exception as e:
#             raise CustomException(e, sys)

# if __name__== "__main__":
#     try:
#         config = ConfigurationManager()
#         data_transformation_config = config.get_data_transformation_config()
#         data_transformation = DataTransformation(config=data_transformation_config)
#         data_transformation.initiate_data_transformation()

#     except Exception as e:
#         raise CustomException(e, sys)


import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from pathlib import Path
from dataclasses import dataclass
from typing import List

from src.ReservationDiscounts.exception import CustomException
from src.ReservationDiscounts.logger import logger
from src.ReservationDiscounts.utils.commons import *
from src.ReservationDiscounts.constants import *

@dataclass
class DataTransformationConfig:
    root_dir: Path
    training_features: Path
    test_features: Path
    validation_features: Path
    training_target: Path
    test_target: Path
    validation_target: Path
    numerical_cols: List[str]
    categorical_cols: List[str]

class ConfigurationManager:
    def __init__(self, data_transformation_config: str = DATA_TRANSFORMATION_CONFIG_FILEPATH):
        try:
            self.preprocessing_config = read_yaml(data_transformation_config)
            create_directories([self.preprocessing_config['artifacts_root']])
        except Exception as e:
            raise CustomException(e, sys)

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

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_transformer_object(self) -> ColumnTransformer:
        logger.info("Getting transformer object")

        try:
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, self.config.numerical_cols),
                    ('cat', categorical_transformer, self.config.categorical_cols)
                ],
                 remainder='passthrough'  # Or 'drop' if you don't want to include the other columns
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        logger.info("Initiating data transformation")

        try:
            # Load data from Parquet files
            logger.info("Loading datasets...")
            X_train = pd.read_parquet(self.config.training_features)
            X_test = pd.read_parquet(self.config.test_features)
            X_val = pd.read_parquet(self.config.validation_features)

            # Get the preprocessor object
            preprocessor_obj = self.get_transformer_object()

            # Transform the training, testing, and validation data
            logger.info("Fitting and transforming training data...")
            X_train_transformed = preprocessor_obj.fit_transform(X_train) #Use fit transform
            logger.info("Transforming test and validation data...")
            X_test_transformed = preprocessor_obj.transform(X_test) #use transform
            X_val_transformed = preprocessor_obj.transform(X_val) #Use transform

            # Convert transformed arrays back to DataFrames
            X_train_transformed_df = pd.DataFrame(X_train_transformed)
            X_test_transformed_df = pd.DataFrame(X_test_transformed)
            X_val_transformed_df = pd.DataFrame(X_val_transformed)

            # Save the transformed data as Parquet files
            transformed_dir = self.config.root_dir / "transformed_data"
            transformed_dir.mkdir(parents=True, exist_ok=True)

            X_train_transformed_df.to_parquet(transformed_dir / "X_train_transformed.parquet")
            X_test_transformed_df.to_parquet(transformed_dir / "X_test_transformed.parquet")
            X_val_transformed_df.to_parquet(transformed_dir / "X_val_transformed.parquet")

            logger.info(f"Transformed datasets saved in {transformed_dir}")

            # Save the preprocessor object to artifacts 
            preprocessor_path = self.config.root_dir / 'preprocessor.joblib'
            joblib.dump(preprocessor_obj, preprocessor_path)
            logger.info(f"Preprocessor object saved at: {preprocessor_path}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__== "__main__":
    try:
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.initiate_data_transformation()

    except Exception as e:
        raise CustomException(e, sys)

