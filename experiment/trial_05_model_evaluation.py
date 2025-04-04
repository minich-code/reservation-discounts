# import sys
# import os
# import time
# import joblib
# import numpy as np
# import pandas as pd
# import wandb

# from dataclasses import dataclass
# from pathlib import Path
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# from sklearn.model_selection import StratifiedKFold

# # Local Modules
# from src.ReservationDiscounts.exception import CustomException
# from src.ReservationDiscounts.logger import logger
# from src.ReservationDiscounts.utils.commons import create_directories, read_yaml
# from src.ReservationDiscounts.constants import MODEL_EVALUATION_CONFIG_FILEPATH


# @dataclass
# class ModelEvaluationConfig:
#     root_dir: Path
#     val_features: Path
#     val_targets: Path
#     model_path: Path
#     project_name: str
#     random_state: int = 42


# class ConfigurationManager:
#     def __init__(self, model_evaluation_config_path: str = MODEL_EVALUATION_CONFIG_FILEPATH):
#         try:
#             self.model_evaluation = read_yaml(model_evaluation_config_path)
#             create_directories([self.model_evaluation['artifacts_root']])
#         except Exception as e:
#             logger.error("Failed to load model evaluation configuration files")
#             raise CustomException(e, sys)

#     def get_model_evaluation_config(self) -> ModelEvaluationConfig:
#         logger.info("Getting the model evaluation configuration")

#         try:
#             model_eval = self.model_evaluation['model_evaluation']
#             create_directories([model_eval['root_dir']])

#             return ModelEvaluationConfig(
#                 root_dir=Path(model_eval['root_dir']),
#                 val_features=Path(model_eval['val_features']),
#                 val_targets=Path(model_eval['val_targets']),
#                 model_path=Path(model_eval['model_path']),
#                 project_name=model_eval['project_name'],
#                 random_state=model_eval['random_state']
#             )
#         except Exception as e:
#             logger.error(f"Failed to load the model evaluation configuration: {str(e)}")
#             raise CustomException(e, sys)


# class ModelEvaluator:
#     def __init__(self, config: ModelEvaluationConfig):
#         self.config = config

#     def load_data(self):
#         """Loads validation data from parquet files"""
#         try:
#             logger.info("Loading validation data")
#             X_val = pd.read_parquet(self.config.val_features)
#             y_val = pd.read_parquet(self.config.val_targets) # Load without squeezing initially

#             if isinstance(y_val, pd.DataFrame):
#                 if y_val.shape[1] == 1:
#                     y_val = y_val.squeeze()  # Convert to Series if single column
#                 else:
#                     raise ValueError("y_val DataFrame has more than one column.  Target must be a single column.")

#             if isinstance(y_val, pd.Series):
#                 print("y_val dtype:", y_val.dtype)  # Access dtype from Series
#             else:
#                 raise TypeError("y_val must be a pandas Series after processing.") # Check it's a Series after processing

#             logger.info("Validation data loaded successfully")
#             return X_val, y_val

#         except Exception as e:
#             logger.exception("Error loading validation data")
#             raise CustomException(e, sys)

#     def load_model(self):
#         """Loads the trained model"""
#         try:
#             model_path = self.config.model_path
#             model = joblib.load(model_path)
#             logger.info(f"Loaded the trained model from: {model_path}")
#             return model
#         except FileNotFoundError as fnf_error:
#             logger.error(f"File not found: {str(fnf_error)}")
#             raise CustomException(fnf_error, sys)
#         except Exception as e:
#             logger.error(f"Error loading model: {str(e)}")
#             raise CustomException(e, sys)

#     def evaluate(self, run_number: int):
#         try:
#             rf_model = self.load_model()
#             X_val, y_val = self.load_data()

#             # Initialize WandB
#             run_name = f"Evaluation {run_number}"
#             run = wandb.init(
#                 project=self.config.project_name,
#                 name=run_name,
#                 config={"random_state": self.config.random_state}
#             )

#             # Stratified K-Fold Cross Validation
#             skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.config.random_state)

#             f1_scores, accuracy_scores, auc_scores = [], [], []
#             train_f1_scores, train_accuracy_scores = [], []

#             logger.info("Starting Stratified K-Fold Cross Validation")

#             for train_idx, val_idx in skf.split(X_val, y_val):
#                 X_train_fold, X_val_fold = X_val.iloc[train_idx], X_val.iloc[val_idx]
#                 y_train_fold, y_val_fold = y_val.iloc[train_idx], y_val.iloc[val_idx]

#                 # Training on train fold
#                 start_time = time.time()
#                 rf_model.fit(X_train_fold, y_train_fold)
#                 train_time = time.time() - start_time

#                 # Train Predictions
#                 y_train_pred = rf_model.predict(X_train_fold)

#                 # Validation Predictions
#                 start_time = time.time()
#                 y_val_pred = rf_model.predict(X_val_fold)
#                 inference_time = time.time() - start_time

#                 # Calculate Metrics
#                 train_f1 = f1_score(y_train_fold, y_train_pred)
#                 train_acc = accuracy_score(y_train_fold, y_train_pred)

#                 f1 = f1_score(y_val_fold, y_val_pred)
#                 acc = accuracy_score(y_val_fold, y_val_pred)
#                 auc = roc_auc_score(y_val_fold, rf_model.predict_proba(X_val_fold)[:, 1])

#                 train_f1_scores.append(train_f1)
#                 train_accuracy_scores.append(train_acc)

#                 f1_scores.append(f1)
#                 accuracy_scores.append(acc)
#                 auc_scores.append(auc)

#             # Compute mean scores
#             mean_f1 = np.mean(f1_scores)
#             mean_acc = np.mean(accuracy_scores)
#             mean_auc = np.mean(auc_scores)

#             mean_train_f1 = np.mean(train_f1_scores)
#             mean_train_acc = np.mean(train_accuracy_scores)

#             # Overfitting Check
#             overfitting_f1_gap = mean_train_f1 - mean_f1
#             overfitting_acc_gap = mean_train_acc - mean_acc

#             # Model Size
#             model_size = os.path.getsize(self.config.model_path) / 1024  # in KB

#             # Log metrics to WandB
#             wandb.log({
#                 "F1 Score": mean_f1,
#                 "Accuracy": mean_acc,
#                 "AUC": mean_auc,
#                 "Train F1": mean_train_f1,
#                 "Train Accuracy": mean_train_acc,
#                 "Overfitting F1 Gap": overfitting_f1_gap,
#                 "Overfitting Accuracy Gap": overfitting_acc_gap,
#                 "Model Size (KB)": model_size,
#                 "Train Time (s)": train_time,
#                 "Inference Time (s)": inference_time
#             })

#             logger.info("Evaluation metrics logged successfully")
#             run.finish()

#         except Exception as e:
#             logger.error("Failed to evaluate the model")
#             raise CustomException(e, sys)


# def get_run_count(root_dir: str, filename: str = "eval_run_count.txt") -> int:
#     """Reads the current evaluation run count from a file"""
#     filepath = os.path.join(root_dir, filename)
#     try:
#         with open(filepath, 'r') as f:
#             return int(f.read().strip())
#     except FileNotFoundError:
#         return 0
#     except ValueError:
#         logger.warning("Failed to read the evaluation run count")
#         return 0


# def write_run_count(root_dir: str, count: int, filename: str = "eval_run_count.txt") -> None:
#     """Writes the evaluation run count to a file"""
#     filepath = os.path.join(root_dir, filename)
#     try:
#         with open(filepath, 'w') as f:
#             f.write(str(count))
#     except Exception as e:
#         logger.error("Failed to write the evaluation run count")
#         raise CustomException(e, sys)


# if __name__ == '__main__':
#     try:
#         config_manager = ConfigurationManager()
#         model_evaluation_config = config_manager.get_model_evaluation_config()
#         model_evaluator = ModelEvaluator(config=model_evaluation_config)

#         run_number = get_run_count(model_evaluation_config.root_dir) + 1
#         model_evaluator.evaluate(run_number)

#         write_run_count(model_evaluation_config.root_dir, run_number)
#         logger.info("Model Evaluation Completed Successfully")

#     except CustomException:
#         logger.error("Error in model evaluation")
#         wandb.finish()
#         sys.exit(1)


# import sys
# import os
# import time
# import joblib
# import numpy as np
# import pandas as pd
# import wandb

# from dataclasses import dataclass
# from pathlib import Path
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# from sklearn.model_selection import StratifiedKFold

# # Local Modules
# from src.ReservationDiscounts.exception import CustomException
# from src.ReservationDiscounts.logger import logger
# from src.ReservationDiscounts.utils.commons import create_directories, read_yaml
# from src.ReservationDiscounts.constants import MODEL_EVALUATION_CONFIG_FILEPATH


# @dataclass
# class ModelEvaluationConfig:
#     root_dir: Path
#     val_features: Path
#     val_targets: Path
#     model_path: Path
#     project_name: str
#     random_state: int 
#     threshold: float


# class ConfigurationManager:
#     def __init__(self, model_evaluation_config_path: str = MODEL_EVALUATION_CONFIG_FILEPATH):
#         try:
#             self.model_evaluation = read_yaml(model_evaluation_config_path)
#             create_directories([self.model_evaluation['artifacts_root']])
#         except Exception as e:
#             logger.error("Failed to load model evaluation configuration files")
#             raise CustomException(e, sys)

#     def get_model_evaluation_config(self) -> ModelEvaluationConfig:
#         logger.info("Getting the model evaluation configuration")

#         try:
#             model_eval = self.model_evaluation['model_evaluation']
#             create_directories([model_eval['root_dir']])

#             return ModelEvaluationConfig(
#                 root_dir=Path(model_eval['root_dir']),
#                 val_features=Path(model_eval['val_features']),
#                 val_targets=Path(model_eval['val_targets']),
#                 model_path=Path(model_eval['model_path']),
#                 project_name=model_eval['project_name'],
#                 random_state=model_eval['random_state'],
#                 threshold=model_eval['threshold']
#             )
#         except Exception as e:
#             logger.error(f"Failed to load the model evaluation configuration: {str(e)}")
#             raise CustomException(e, sys)


# class ModelEvaluator:
#     def __init__(self, config: ModelEvaluationConfig):
#         self.config = config

#     def load_data(self):
#         """Loads validation data from parquet files"""
#         try:
#             logger.info("Loading validation data")
#             X_val = pd.read_parquet(self.config.val_features)
#             y_val = pd.read_parquet(self.config.val_targets)  # Load without squeezing initially

#             if isinstance(y_val, pd.DataFrame):
#                 if y_val.shape[1] == 1:
#                     y_val = y_val.squeeze()  # Convert to Series if single column
#                 else:
#                     raise ValueError("y_val DataFrame has more than one column.  Target must be a single column.")

#             if isinstance(y_val, pd.Series):
#                 print("y_val dtype:", y_val.dtype)  # Access dtype from Series
#             else:
#                 raise TypeError("y_val must be a pandas Series after processing.")  # Check it's a Series after processing

#             logger.info("Validation data loaded successfully")
#             return X_val, y_val

#         except Exception as e:
#             logger.exception("Error loading validation data")
#             raise CustomException(e, sys)

#     def load_model(self):
#         """Loads the trained model"""
#         try:
#             model_path = self.config.model_path
#             model = joblib.load(model_path)
#             logger.info(f"Loaded the trained model from: {model_path}")
#             return model
#         except FileNotFoundError as fnf_error:
#             logger.error(f"File not found: {str(fnf_error)}")
#             raise CustomException(fnf_error, sys)
#         except Exception as e:
#             logger.error(f"Error loading model: {str(e)}")
#             raise CustomException(e, sys)

#     def evaluate(self, run_number: int, threshold: float = 0.65): #default 0.65 threshold
#         try:
#             rf_model = self.load_model()
#             X_val, y_val = self.load_data()

#             # Initialize WandB
#             run_name = f"Evaluation {run_number}"
#             run = wandb.init(
#                 project=self.config.project_name,
#                 name=run_name,
#                 config={"random_state": self.config.random_state, "threshold": threshold} #Logs 0.65 threshold
#             )

#             # Make predictions on the validation set (using probabilities)
#             y_pred_proba = rf_model.predict_proba(X_val)[:, 1] #Probabilties from X_val instead of cross validation
#             y_pred_adjusted = (y_pred_proba >= threshold).astype(int) # Apply threshold

#             # Calculate Metrics
#             accuracy = accuracy_score(y_val, y_pred_adjusted)
#             f1 = f1_score(y_val, y_pred_adjusted)
#             auc = roc_auc_score(y_val, y_pred_proba)

#             # Log metrics to WandB
#             wandb.log({
#                 "F1 Score": f1,
#                 "Accuracy": accuracy,
#                 "AUC": auc,
#                 "Threshold": threshold #Logs threshold
#             })

#             logger.info("Evaluation metrics logged successfully")
#             run.finish()

#         except Exception as e:
#             logger.error(f"Failed to evaluate the model: {e}")
#             raise CustomException(e, sys)


# def get_run_count(root_dir: str, filename: str = "eval_run_count.txt") -> int:
#     """Reads the current evaluation run count from a file"""
#     filepath = os.path.join(root_dir, filename)
#     try:
#         with open(filepath, 'r') as f:
#             return int(f.read().strip())
#     except FileNotFoundError:
#         return 0
#     except ValueError:
#         logger.warning("Failed to read the evaluation run count")
#         return 0

# def write_run_count(root_dir: str, count: int, filename: str = "eval_run_count.txt") -> None:
#     """Writes the evaluation run count to a file"""
#     filepath = os.path.join(root_dir, filename)
#     try:
#         with open(filepath, 'w') as f:
#             f.write(str(count))
#     except Exception as e:
#         logger.error("Failed to write the evaluation run count")
#         raise CustomException(e, sys)


# if __name__ == '__main__':
#     try:
#         config_manager = ConfigurationManager()
#         model_evaluation_config = config_manager.get_model_evaluation_config()
#         model_evaluator = ModelEvaluator(config=model_evaluation_config)

#         run_number = get_run_count(model_evaluation_config.root_dir) + 1
#         model_evaluator.evaluate(run_number, threshold = 0.65)  # Call function with threshold

#         write_run_count(model_evaluation_config.root_dir, run_number)
#         logger.info("Model Evaluation Completed Successfully")

#     except CustomException as e:
#         logger.error(f"Error in model evaluation: {e}")
#         wandb.finish()
#         sys.exit(1)


import sys
import os
import time
import joblib
import numpy as np
import pandas as pd
import wandb

from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Local Modules
from src.ReservationDiscounts.exception import CustomException
from src.ReservationDiscounts.logger import logger
from src.ReservationDiscounts.utils.commons import create_directories, read_yaml
from src.ReservationDiscounts.constants import MODEL_EVALUATION_CONFIG_FILEPATH


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    val_features: Path
    val_targets: Path
    model_path: Path
    project_name: str
    random_state: int
    threshold: float


class ConfigurationManager:
    def __init__(self, model_evaluation_config_path: str = MODEL_EVALUATION_CONFIG_FILEPATH):
        try:
            self.model_evaluation = read_yaml(model_evaluation_config_path)
            create_directories([self.model_evaluation['artifacts_root']])
        except Exception as e:
            logger.error("Failed to load model evaluation configuration files")
            raise CustomException(e, sys)

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        logger.info("Getting the model evaluation configuration")

        try:
            model_eval = self.model_evaluation['model_evaluation']
            create_directories([model_eval['root_dir']])

            return ModelEvaluationConfig(
                root_dir=Path(model_eval['root_dir']),
                val_features=Path(model_eval['val_features']),
                val_targets=Path(model_eval['val_targets']),
                model_path=Path(model_eval['model_path']),
                project_name=model_eval['project_name'],
                random_state=model_eval['random_state'],
                threshold=float(model_eval['threshold'])
            )
        except Exception as e:
            logger.error(f"Failed to load the model evaluation configuration: {str(e)}")
            raise CustomException(e, sys)


class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_data(self):
        """Loads validation data from parquet files"""
        try:
            logger.info("Loading validation data")
            X_val = pd.read_parquet(self.config.val_features)
            y_val = pd.read_parquet(self.config.val_targets)  # Load without squeezing initially

            if isinstance(y_val, pd.DataFrame):
                if y_val.shape[1] == 1:
                    y_val = y_val.squeeze()  # Convert to Series if single column
                else:
                    raise ValueError("y_val DataFrame has more than one column.  Target must be a single column.")

            if isinstance(y_val, pd.Series):
                print("y_val dtype:", y_val.dtype)  # Access dtype from Series
            else:
                raise TypeError("y_val must be a pandas Series after processing.")  # Check it's a Series after processing

            logger.info("Validation data loaded successfully")
            return X_val, y_val

        except Exception as e:
            logger.exception("Error loading validation data")
            raise CustomException(e, sys)

    def load_model(self):
        """Loads the trained model"""
        try:
            model_path = self.config.model_path
            model = joblib.load(model_path)
            logger.info(f"Loaded the trained model from: {model_path}")
            return model
        except FileNotFoundError as fnf_error:
            logger.error(f"File not found: {str(fnf_error)}")
            raise CustomException(fnf_error, sys)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise CustomException(e, sys)

    def evaluate(self, run_number: int): #default 0.65 threshold
        try:
            rf_model = self.load_model()
            X_val, y_val = self.load_data()

            # Initialize WandB
            run_name = f"Evaluation {run_number}"
            run = wandb.init(
                project=self.config.project_name,
                name=run_name,
                config={"random_state": self.config.random_state, "threshold": self.config.threshold} #Logs 0.65 threshold
            )

            # Make predictions on the validation set (using probabilities)
            y_pred_proba = rf_model.predict_proba(X_val)[:, 1] #Probabilties from X_val instead of cross validation
            y_pred_adjusted = (y_pred_proba >= self.config.threshold).astype(int) # Apply threshold

            # Calculate Metrics
            accuracy = accuracy_score(y_val, y_pred_adjusted)
            f1 = f1_score(y_val, y_pred_adjusted)
            auc = roc_auc_score(y_val, y_pred_proba)

            # Log metrics to WandB
            wandb.log({
                "F1 Score": f1,
                "Accuracy": accuracy,
                "AUC": auc,
                "Threshold": self.config.threshold #Logs threshold
            })

            logger.info("Evaluation metrics logged successfully")
            run.finish()

        except Exception as e:
            logger.error(f"Failed to evaluate the model: {e}")
            raise CustomException(e, sys)


def get_run_count(root_dir: str, filename: str = "eval_run_count.txt") -> int:
    """Reads the current evaluation run count from a file"""
    filepath = os.path.join(root_dir, filename)
    try:
        with open(filepath, 'r') as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0
    except ValueError:
        logger.warning("Failed to read the evaluation run count")
        return 0

def write_run_count(root_dir: str, count: int, filename: str = "eval_run_count.txt") -> None:
    """Writes the evaluation run count to a file"""
    filepath = os.path.join(root_dir, filename)
    try:
        with open(filepath, 'w') as f:
            f.write(str(count))
    except Exception as e:
        logger.error("Failed to write the evaluation run count")
        raise CustomException(e, sys)


if __name__ == '__main__':
    try:
        config_manager = ConfigurationManager()
        model_evaluation_config = config_manager.get_model_evaluation_config()
        model_evaluator = ModelEvaluator(config=model_evaluation_config)

        run_number = get_run_count(model_evaluation_config.root_dir) + 1
        model_evaluator.evaluate(run_number)  # Call function with threshold

        write_run_count(model_evaluation_config.root_dir, run_number)
        logger.info("Model Evaluation Completed Successfully")

    except CustomException as e:
        logger.error(f"Error in model evaluation: {e}")
        wandb.finish()
        sys.exit(1)