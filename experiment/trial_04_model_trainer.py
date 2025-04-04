

import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import wandb
from datetime import datetime

# Local Modules
from src.ReservationDiscounts.exception import CustomException
from src.ReservationDiscounts.logger import logger
from src.ReservationDiscounts.utils.commons import create_directories, read_yaml
from src.ReservationDiscounts.constants import MODEL_TRAINER_CONFIG_FILEPATH, PARAMS_CONFIG_FILEPATH


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_features: Path
    train_targets: Path
    model_name: str
    project_name: str
    random_state: int
    number_of_splits: int
    model_params: dict  


class ConfigurationManager:
    def __init__(self,
                 model_training_config: Path = MODEL_TRAINER_CONFIG_FILEPATH,
                 model_params_config: Path = PARAMS_CONFIG_FILEPATH):
        try:
            self.training_config = read_yaml(model_training_config)
            self.model_params_config = read_yaml(model_params_config)

            # Ensure artifacts_root exists in the config
            create_directories([self.training_config['artifacts_root']])
        except Exception as e:
            logger.error(f"Error loading model training config file: {str(e)}")
            raise CustomException(e, sys)

    def get_model_training_config(self) -> ModelTrainerConfig:
        logger.info("Getting model training configuration")
        try:
            trainer_config = self.training_config['model_trainer']
            model_params = self.model_params_config['RandomForest_params']

            create_directories([trainer_config['root_dir']])

            return ModelTrainerConfig(
                root_dir=Path(trainer_config['root_dir']),
                train_features=Path(trainer_config['train_features']),
                train_targets=Path(trainer_config['train_targets']),
                model_name=trainer_config['model_name'],
                model_params=model_params,
                project_name=trainer_config['project_name'],
                random_state=trainer_config['random_state'],
                number_of_splits=trainer_config['number_of_splits']
            )

        except Exception as e:
            logger.exception(f"Error loading model training configuration: {e}")
            raise CustomException(e, sys)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_data(self):
        """Loads training data from parquet files"""
        try:
            logger.info("Loading data")
            X_train = pd.read_parquet(self.config.train_features)
            y_train = pd.read_parquet(self.config.train_targets).squeeze()

            assert y_train.dtype in [int, 'int32', 'int64'], "Target labels should be integers"

            logger.info("Data loaded successfully")

            return X_train, y_train
        except Exception as e:
            logger.exception("Error loading data")
            raise CustomException(e, sys)

    def model_training(self):
        """Trains a RandomForest model and logs it using Weights & Biases"""
        try:
            # Load Data
            X_train, y_train = self.load_data()

            # Initialize Weights & Biases tracking
            run = wandb.init(
                project=self.config.project_name,
                config={**self.config.model_params, "random_state": self.config.random_state},
            )

            # Train Model
            logger.info("Initializing and Training RandomForest model")
            rf_model = RandomForestClassifier(**self.config.model_params)#, random_state=self.config.random_state)
            rf_model.fit(X_train, y_train)

            # Save Model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.config.model_name.split('.')[0]}_{timestamp}.joblib"
            model_path = self.config.root_dir / model_name
            joblib.dump(rf_model, model_path)
            logger.info(f"Model trained and saved at: {model_path}")

            # Log Model Artifact to Weights & Biases
            artifact = wandb.Artifact(f"{self.config.model_name}_{run.id}", type="model")
            artifact.add_file(str(model_path))
            run.log_artifact(artifact)

            run.finish()

            logger.info("Model Training Completed")
            return rf_model, model_path

        except Exception as e:
            logger.exception(f"Error during model training: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        model_training_config = config_manager.get_model_training_config()
        model_trainer = ModelTrainer(config=model_training_config)
        model_trainer.model_training()  # Correct method name
        logger.info("Model training process completed successfully.")

    except CustomException as e:
        logger.error(f"Error during model training: {str(e)}")
        sys.exit(1)



# import sys
# from dataclasses import dataclass
# from pathlib import Path
# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# import wandb

# # Local Modules
# from src.ReservationDiscounts.exception import CustomException
# from src.ReservationDiscounts.logger import logger
# from src.ReservationDiscounts.utils.commons import create_directories, read_yaml
# from src.ReservationDiscounts.constants import MODEL_TRAINER_CONFIG_FILEPATH, PARAMS_CONFIG_FILEPATH


# @dataclass
# class ModelTrainerConfig:
#     root_dir: Path
#     train_features: Path
#     train_targets: Path
#     model_name: str
#     project_name: str
#     random_state: int
#     number_of_splits: int
#     model_params: dict  


# class ConfigurationManager:
#     def __init__(self,
#                  model_training_config: Path = MODEL_TRAINER_CONFIG_FILEPATH,
#                  model_params_config: Path = PARAMS_CONFIG_FILEPATH):
#         try:
#             self.training_config = read_yaml(model_training_config)
#             self.model_params_config = read_yaml(model_params_config)

#             # Ensure artifacts_root exists in the config
#             create_directories([self.training_config['artifacts_root']])
#         except Exception as e:
#             logger.error(f"Error loading model training config file: {str(e)}")
#             raise CustomException(e, sys)

#     def get_model_training_config(self) -> ModelTrainerConfig:
#         logger.info("Getting model training configuration")
#         try:
#             trainer_config = self.training_config['model_trainer']
#             model_params = self.model_params_config['RandomForest_params']

#             create_directories([trainer_config['root_dir']])

#             return ModelTrainerConfig(
#                 root_dir=Path(trainer_config['root_dir']),
#                 train_features=Path(trainer_config['train_features']),
#                 train_targets=Path(trainer_config['train_targets']),
#                 model_name=trainer_config['model_name'],
#                 model_params=model_params,
#                 project_name=trainer_config['project_name'],
#                 random_state=trainer_config['random_state'],
#                 number_of_splits=trainer_config['number_of_splits']
#             )

#         except Exception as e:
#             logger.exception(f"Error loading model training configuration: {e}")
#             raise CustomException(e, sys)


# class ModelTrainer:
#     def __init__(self, config: ModelTrainerConfig):
#         self.config = config

#     def load_data(self):
#         """Loads training data from parquet files"""
#         try:
#             logger.info("Loading data")
#             X_train = pd.read_parquet(self.config.train_features)
#             y_train = pd.read_parquet(self.config.train_targets).squeeze()

#             logger.info("Data loaded successfully")

#             return X_train, y_train
#         except Exception as e:
#             logger.exception("Error loading data")
#             raise CustomException(e, sys)

#     def model_training(self):
#         """Trains a RandomForest model and logs it using Weights & Biases"""
#         try:
#             # Load Data
#             X_train, y_train = self.load_data()

#             # Initialize Weights & Biases tracking
#             run = wandb.init(
#                 project=self.config.project_name,
#                 config={**self.config.model_params, "random_state": self.config.random_state},
        
#             )

#             # Train Model
#             logger.info("Initializing and Training RandomForest model")
#             rf_model = RandomForestClassifier(**self.config.model_params, random_state=self.config.random_state)
#             rf_model.fit(X_train, y_train)

#             # Save Model
#             model_path = self.config.root_dir / self.config.model_name
#             joblib.dump(rf_model, model_path)
#             logger.info(f"Model trained and saved at: {model_path}")

#             # Log Model Artifact to Weights & Biases
#             artifact = wandb.Artifact("model", type="model")
#             artifact.add_file(str(model_path))
#             run.log_artifact(artifact)

#             run.finish()

#             logger.info("Model Training Completed")
#             return rf_model  # Ensure logger executes before returning

#         except Exception as e:
#             logger.exception(f"Error during model training: {e}")
#             raise CustomException(e, sys)


# if __name__ == "__main__":
#     try:
#         config_manager = ConfigurationManager()
#         model_training_config = config_manager.get_model_training_config()
#         model_trainer = ModelTrainer(config=model_training_config)
#         model_trainer.model_training()  # Correct method name
#         logger.info("Model training process completed successfully.")

#     except CustomException as e:
#         logger.error(f"Error during model training: {str(e)}")
#         sys.exit(1)
