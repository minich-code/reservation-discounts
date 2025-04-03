
import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from evidently.metrics import DataDriftTable, ColumnDriftMetric
from evidently.report import Report

from src.ReservationDiscounts.exception import CustomException
from src.ReservationDiscounts.logger import logger
from src.ReservationDiscounts.utils.commons import *
from src.ReservationDiscounts.constants import *

@dataclass
class DataDriftConfig:
    root_dir: Path
    data_path: Path
    random_state: int
    target_col: str
    numerical_cols: list
    categorical_cols: list

class ConfigurationManager:
    def __init__(self, data_drift_config: str = DATA_DRIFT_CONFIG_FILEPATH):
        try:
            self.data_drift = read_yaml(data_drift_config)
            create_directories([self.data_drift['artifacts_root']])
        except Exception as e:
            logger.exception(f"Error initializing data drift Configuration Manager")
            raise CustomException(e, sys)

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

class DataDriftDetector:
    def __init__(self, config: DataDriftConfig):
        self.config = config

    def load_data(self):
        try:
            logger.info("Loading data")
            df = pd.read_parquet(self.config.data_path)
            logger.info(f"Data shape: {df.shape}")
            return df
        except Exception as e:
            logger.exception(f"Error loading data from {self.config.data_path}")
            raise CustomException(e, sys)

    def split_data(self, df):
        try:
            X = df.drop(self.config.target_col, axis=1)
            y = df[self.config.target_col]

            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=self.config.random_state
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=self.config.random_state
            )

            logger.info("Data split into Train, Validation, and Test sets")

            # Save the datasets as Parquet files
            save_dir = Path(self.config.root_dir) / "split_data"
            save_dir.mkdir(parents=True, exist_ok=True)

            X_train.to_parquet(save_dir / "X_train.parquet")
            X_val.to_parquet(save_dir / "X_val.parquet")
            X_test.to_parquet(save_dir / "X_test.parquet")
            y_train.to_frame().to_parquet(save_dir / "y_train.parquet")
            y_val.to_frame().to_parquet(save_dir / "y_val.parquet")
            y_test.to_frame().to_parquet(save_dir / "y_test.parquet")

            logger.info(f"Split data saved in {save_dir}")

            return X_train, X_val, X_test, y_train, y_val, y_test
        except Exception as e:
            logger.exception("Error splitting the data")
            raise CustomException(e, sys)

    def detect_drift(self, X_train, X_test, y_train, y_test):
        try:
            logger.info("Performing Data Drift Detection using Evidently")

            # Ensure output directory exists
            os.makedirs(self.config.root_dir, exist_ok=True)

            # Overall Data Drift (Feature Data)
            overall_drift_report = Report(
                metrics=[DataDriftTable(num_stattest='psi', cat_stattest='jensenshannon')]
            )
            overall_drift_report.run(reference_data=X_train, current_data=X_test)
            overall_drift_report.save_html(os.path.join(self.config.root_dir, "overall_data_drift.html"))
            logger.info("Overall Feature Data Drift Report saved.")

            # Create a list to store all column drift metrics
            column_metrics = []

            # Add drift metrics for numerical columns
            for col in self.config.numerical_cols:
                column_metrics.append(ColumnDriftMetric(column_name=col))
                logger.info(f"Added Numerical Drift Metric for {col}")

            # Add drift metrics for categorical columns
            for col in self.config.categorical_cols:
                column_metrics.append(ColumnDriftMetric(column_name=col))
                logger.info(f"Added Categorical Drift Metric for {col}")

            # Add target drift metric
            column_metrics.append(ColumnDriftMetric(column_name=self.config.target_col))
            logger.info(f"Added Target Drift Metric for {self.config.target_col}")

            # Create a single report containing all column drift metrics
            combined_drift_report = Report(metrics=column_metrics)

            # Prepare target data as DataFrames for target drift analysis
            y_train_df = pd.DataFrame(y_train, columns=[self.config.target_col])
            y_test_df = pd.DataFrame(y_test, columns=[self.config.target_col])

            # Run the combined report, passing in feature and target data
            combined_drift_report.run(reference_data=X_train.join(y_train_df),
                                    current_data=X_test.join(y_test_df),
                                    column_mapping=None)  # Let Evidently infer column types
            combined_drift_report.save_html(os.path.join(self.config.root_dir, "combined_data_drift.html"))
            logger.info("Combined Data Drift Report saved.")

        except Exception as e:
            logger.exception("Error during Data Drift Detection")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_drift_config = config_manager.get_data_drift_config()

        drift_detector = DataDriftDetector(data_drift_config)

        df = drift_detector.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = drift_detector.split_data(df)

        # Run Data Drift Detection
        drift_detector.detect_drift(X_train, X_test, y_train, y_test)

    except Exception as e:
        logger.error(f"Error in Data Drift Pipeline: {e}")
        sys.exit(1)




# import sys
# from dataclasses import dataclass
# from pathlib import Path
# import pandas as pd
# import os

# from sklearn.model_selection import train_test_split
# from evidently.metrics import DataDriftTable, ColumnDriftMetric
# from evidently.report import Report

# from src.ReservationDiscounts.exception import CustomException
# from src.ReservationDiscounts.logger import logger
# from src.ReservationDiscounts.utils.commons import *
# from src.ReservationDiscounts.constants import *

# @dataclass
# class DataDriftConfig:
#     root_dir: Path
#     data_path: Path
#     random_state: int
#     target_col: str
#     numerical_cols: list
#     categorical_cols: list

# class ConfigurationManager:
#     def __init__(self, data_drift_config: str = DATA_DRIFT_CONFIG_FILEPATH):
#         try:
#             self.data_drift = read_yaml(data_drift_config)
#             create_directories([self.data_drift['artifacts_root']])
#         except Exception as e:
#             logger.exception(f"Error initializing data drift Configuration Manager")
#             raise CustomException(e, sys)

#     def get_data_drift_config(self) -> DataDriftConfig:
#         try:
#             drift_config = self.data_drift['data_drift']
#             create_directories([drift_config['root_dir']])

#             return DataDriftConfig(
#                 root_dir = drift_config['root_dir'],
#                 data_path = drift_config['data_path'],
#                 random_state = drift_config['random_state'],
#                 target_col = drift_config['target_col'],
#                 numerical_cols = drift_config['numerical_cols'],
#                 categorical_cols = drift_config['categorical_cols']
#             )

#         except Exception as e:
#             logger.exception(f"Error getting the Data Drift Config")
#             raise CustomException(e, sys)

# class DataDriftDetector:
#     def __init__(self, config: DataDriftConfig):
#         self.config = config

#     def load_data(self):
#         try:
#             logger.info("Loading data")
#             df = pd.read_parquet(self.config.data_path)
#             logger.info(f"Data shape: {df.shape}")
#             return df
#         except Exception as e:
#             logger.exception(f"Error loading data from {self.config.data_path}")
#             raise CustomException(e, sys)

#     def split_data(self, df):
#         try:
#             X = df.drop(self.config.target_col, axis=1)
#             y = df[self.config.target_col]

#             X_train, X_temp, y_train, y_temp = train_test_split(
#                 X, y, test_size=0.3, stratify=y, random_state=self.config.random_state
#             )
#             X_val, X_test, y_val, y_test = train_test_split(
#                 X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=self.config.random_state
#             )

#             logger.info("Data split into Train, Validation, and Test sets")
#             return X_train, X_val, X_test, y_train, y_val, y_test
#         except Exception as e:
#             logger.exception("Error splitting the data")
#             raise CustomException(e, sys)

#     def detect_drift(self, X_train, X_test, y_train, y_test):
#         try:
#             logger.info("Performing Data Drift Detection using Evidently")

#             # Ensure output directory exists
#             os.makedirs(self.config.root_dir, exist_ok=True)

#             # Overall Data Drift (Feature Data)
#             overall_drift_report = Report(
#                 metrics=[DataDriftTable(num_stattest='ks', cat_stattest='jensenshannon')]
#             )
#             overall_drift_report.run(reference_data=X_train, current_data=X_test)
#             overall_drift_report.save_html(os.path.join(self.config.root_dir, "overall_data_drift.html"))
#             logger.info("Overall Feature Data Drift Report saved.")

#             # Create a list to store all column drift metrics
#             column_metrics = []

#             # Add drift metrics for numerical columns
#             for col in self.config.numerical_cols:
#                 column_metrics.append(ColumnDriftMetric(column_name=col))
#                 logger.info(f"Added Numerical Drift Metric for {col}")

#             # Add drift metrics for categorical columns
#             for col in self.config.categorical_cols:
#                 column_metrics.append(ColumnDriftMetric(column_name=col))
#                 logger.info(f"Added Categorical Drift Metric for {col}")

#             # Add target drift metric
#             column_metrics.append(ColumnDriftMetric(column_name=self.config.target_col))
#             logger.info(f"Added Target Drift Metric for {self.config.target_col}")

#             # Create a single report containing all column drift metrics
#             combined_drift_report = Report(metrics=column_metrics)

#             # Prepare target data as DataFrames for target drift analysis
#             y_train_df = pd.DataFrame(y_train, columns=[self.config.target_col])
#             y_test_df = pd.DataFrame(y_test, columns=[self.config.target_col])

#             # Run the combined report, passing in feature and target data
#             combined_drift_report.run(reference_data=X_train.join(y_train_df),
#                                     current_data=X_test.join(y_test_df),
#                                     column_mapping=None)  # Let Evidently infer column types
#             combined_drift_report.save_html(os.path.join(self.config.root_dir, "combined_data_drift.html"))
#             logger.info("Combined Data Drift Report saved.")

#         except Exception as e:
#             logger.exception("Error during Data Drift Detection")
#             raise CustomException(e, sys)

# if __name__ == "__main__":
#     try:
#         config_manager = ConfigurationManager()
#         data_drift_config = config_manager.get_data_drift_config()

#         drift_detector = DataDriftDetector(data_drift_config)

#         df = drift_detector.load_data()
#         X_train, X_val, X_test, y_train, y_val, y_test = drift_detector.split_data(df)

#         # Run Data Drift Detection
#         drift_detector.detect_drift(X_train, X_test, y_train, y_test)

#     except Exception as e:
#         logger.error(f"Error in Data Drift Pipeline: {e}")
#         sys.exit(1)



#-----------------------------------------------------use above--------------------------



# import sys
# from dataclasses import dataclass
# from pathlib import Path
# import pandas as pd
# import os
# from sklearn.model_selection import train_test_split
# from evidently.metrics import DataDriftTable, ColumnDriftMetric
# from evidently.report import Report

# from src.ReservationDiscounts.exception import CustomException
# from src.ReservationDiscounts.logger import logger
# from src.ReservationDiscounts.utils.commons import *
# from src.ReservationDiscounts.constants import *

# @dataclass
# class DataDriftConfig:
#     root_dir: Path
#     data_path: Path
#     random_state: int
#     target_col: str
#     numerical_cols: list
#     categorical_cols: list

# class ConfigurationManager:
#     def __init__(self, data_drift_config: str = DATA_DRIFT_CONFIG_FILEPATH):
#         try:
#             self.data_drift = read_yaml(data_drift_config)
#             create_directories([self.data_drift['artifacts_root']])
#         except Exception as e:
#             logger.exception("Error initializing data drift Configuration Manager")
#             raise CustomException(e, sys)

#     def get_data_drift_config(self) -> DataDriftConfig:
#         try:
#             drift_config = self.data_drift['data_drift']
#             create_directories([drift_config['root_dir']])

#             return DataDriftConfig(
#                 root_dir=drift_config['root_dir'],
#                 data_path=drift_config['data_path'],
#                 random_state=drift_config['random_state'],
#                 target_col=drift_config['target_col'],
#                 numerical_cols=drift_config['numerical_cols'],
#                 categorical_cols=drift_config['categorical_cols']
#             )
#         except Exception as e:
#             logger.exception("Error getting the Data Drift Config")
#             raise CustomException(e, sys)

# class DataDriftDetector:
#     def __init__(self, config: DataDriftConfig):
#         self.config = config

#     def load_data(self) -> pd.DataFrame:
#         try:
#             logger.info("Loading data")
#             df = pd.read_parquet(self.config.data_path)
#             logger.info(f"Data shape: {df.shape}")
#             return df
#         except Exception as e:
#             logger.exception("Error loading data")
#             raise CustomException(e, sys)

#     def split_data(self, df: pd.DataFrame):
#         try:
#             X = df.drop(columns=[self.config.target_col])
#             y = df[self.config.target_col]

#             X_train, X_temp, y_train, y_temp = train_test_split(
#                 X, y, test_size=0.3, stratify=y, random_state=self.config.random_state
#             )
#             X_val, X_test, y_val, y_test = train_test_split(
#                 X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=self.config.random_state
#             )

#             logger.info("Training, validation, and testing data split successfully")
#             return X_train, X_test, y_train, y_test
#         except Exception as e:
#             logger.exception("Error splitting the data")
#             raise CustomException(e, sys)

#     def detect_drift(self, X_train, X_test, y_train, y_test):
#         try:
#             logger.info("Performing Data Drift Detection using Evidently")

#             # Create a single drift report covering all numerical, categorical, and target columns
#             drift_report = Report(
#                 metrics=[
#                     DataDriftTable(num_stattest='psi', cat_stattest='jensenshannon')
#                 ] + 
#                 [ColumnDriftMetric(column_name=col) for col in self.config.numerical_cols + self.config.categorical_cols] +
#                 [ColumnDriftMetric(column_name=self.config.target_col)]
#             )
#             drift_report.run(reference_data=X_train.assign(target=y_train), current_data=X_test.assign(target=y_test))

#             # Save the full report
#             full_report_path = os.path.join(self.config.root_dir, "data_drift_report.html")
#             drift_report.save_html(full_report_path)
#             logger.info(f"Full Data Drift Report saved at: {full_report_path}")

#         except Exception as e:
#             logger.exception("Error during Data Drift Detection")
#             raise CustomException(e, sys)

# if __name__ == "__main__":
#     try:
#         config_manager = ConfigurationManager()
#         data_drift_config = config_manager.get_data_drift_config()

#         drift_detector = DataDriftDetector(data_drift_config)

#         df = drift_detector.load_data()
#         X_train, X_test, y_train, y_test = drift_detector.split_data(df)

#         # Run Data Drift Detection
#         drift_detector.detect_drift(X_train, X_test, y_train, y_test)

#     except Exception as e:
#         logger.error(f"Error in Data Drift Pipeline: {e}")
#         sys.exit(1)


# ------------INDIVIDUAL COLUMNS -----------------------
# import sys 
# from dataclasses import dataclass 
# from pathlib import Path 
# import pandas as pd 
# import os
# from sklearn.model_selection import train_test_split 
# from evidently.metrics import DataDriftTable, ColumnDriftMetric
# from evidently.report import Report

# from src.ReservationDiscounts.exception import CustomException
# from src.ReservationDiscounts.logger import logger 
# from src.ReservationDiscounts.utils.commons import *
# from src.ReservationDiscounts.constants import *

# @dataclass 
# class DataDriftConfig:
#     root_dir: Path
#     data_path: Path
#     random_state: int
#     target_col: str
#     numerical_cols: list
#     categorical_cols: list

# class ConfigurationManager:
#     def __init__(self, data_drift_config: str = DATA_DRIFT_CONFIG_FILEPATH):
#         try:
#             self.data_drift = read_yaml(data_drift_config)
#             create_directories([self.data_drift['artifacts_root']])
#         except Exception as e:
#             logger.exception("Error initializing data drift Configuration Manager")
#             raise CustomException(e, sys)

#     def get_data_drift_config(self) -> DataDriftConfig:
#         try:
#             drift_config = self.data_drift['data_drift']
#             create_directories([drift_config['root_dir']])

#             return DataDriftConfig(
#                 root_dir = drift_config['root_dir'],
#                 data_path = drift_config['data_path'],
#                 random_state = drift_config['random_state'],
#                 target_col = drift_config['target_col'],
#                 numerical_cols = drift_config['numerical_cols'],
#                 categorical_cols = drift_config['categorical_cols']
#             )
#         except Exception as e:
#             logger.exception("Error getting the Data Drift Config")
#             raise CustomException(e, sys)

# class DataDriftDetector:
#     def __init__(self, config: DataDriftConfig):
#         self.config = config 

#     def load_data(self) -> pd.DataFrame:
#         try:
#             logger.info("Loading data")
#             df = pd.read_parquet(self.config.data_path)
#             logger.info(f"Data shape: {df.shape}")
#             return df
#         except Exception as e:
#             logger.exception("Error loading data")
#             raise CustomException(e, sys)
        
#     def split_data(self, df: pd.DataFrame):
#         try:
#             # Define features and target variable 
#             X = df.drop(columns=[self.config.target_col])
#             y = df[self.config.target_col]

#             # Split data into training, testing, and validation 
#             X_train, X_temp, y_train, y_temp = train_test_split(
#                 X, y, test_size=0.3, stratify=y, random_state=self.config.random_state
#             )
#             X_val, X_test, y_val, y_test = train_test_split(
#                 X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=self.config.random_state
#             )

#             logger.info("Training, validation, and testing data split successfully")
#             return X_train, X_val, X_test, y_train, y_val, y_test
#         except Exception as e:
#             logger.exception("Error splitting the data")
#             raise CustomException(e, sys)

#     def detect_drift(self, X_train, X_test, y_train, y_test):
#         try:
#             logger.info("Performing Data Drift Detection using Evidently")

#             # Create overall data drift report
#             drift_report = Report(metrics=[DataDriftTable(num_stattest='psi', cat_stattest='jensenshannon')])
#             drift_report.run(reference_data=X_train, current_data=X_test)
#             overall_report_path = os.path.join(self.config.root_dir, "overall_data_drift.html")
#             drift_report.save_html(overall_report_path)
#             logger.info(f"Overall Data Drift Report saved at: {overall_report_path}")

#             # Detect drift for numerical columns
#             for col in self.config.numerical_cols:
#                 if col == self.config.target_col:
#                     continue  # Skip target column
#                 num_drift_report = Report(metrics=[ColumnDriftMetric(column_name=col)])
#                 num_drift_report.run(reference_data=X_train, current_data=X_test)
#                 num_report_path = os.path.join(self.config.root_dir, f"num_drift_{col}.html")
#                 num_drift_report.save_html(num_report_path)
#                 logger.info(f"Numerical Drift Report for {col} saved at: {num_report_path}")

#             # Detect drift for categorical columns (excluding target column)
#             for col in self.config.categorical_cols:
#                 if col == self.config.target_col:
#                     continue  # Skip target column
#                 cat_drift_report = Report(metrics=[ColumnDriftMetric(column_name=col)])
#                 cat_drift_report.run(reference_data=X_train, current_data=X_test)
#                 cat_report_path = os.path.join(self.config.root_dir, f"cat_drift_{col}.html")
#                 cat_drift_report.save_html(cat_report_path)
#                 logger.info(f"Categorical Drift Report for {col} saved at: {cat_report_path}")

#             # Detect drift for target column separately
#             target_drift_report = Report(metrics=[ColumnDriftMetric(column_name=self.config.target_col)])
#             target_drift_report.run(
#                 reference_data=pd.DataFrame({self.config.target_col: y_train}), 
#                 current_data=pd.DataFrame({self.config.target_col: y_test})
#             )
#             target_report_path = os.path.join(self.config.root_dir, f"target_drift_{self.config.target_col}.html")
#             target_drift_report.save_html(target_report_path)
#             logger.info(f"Target Drift Report for {self.config.target_col} saved at: {target_report_path}")

#         except Exception as e:
#             logger.exception("Error during Data Drift Detection")
#             raise CustomException(e, sys)

# if __name__ == "__main__":
#     try:
#         config_manager = ConfigurationManager()
#         data_drift_config = config_manager.get_data_drift_config()

#         drift_detector = DataDriftDetector(data_drift_config)

#         df = drift_detector.load_data()
#         X_train, X_val, X_test, y_train, y_val, y_test = drift_detector.split_data(df)

#         # Run Data Drift Detection
#         drift_detector.detect_drift(X_train, X_test, y_train, y_test)

#     except Exception as e:
#         logger.error(f"Error in Data Drift Pipeline: {e}")
#         sys.exit(1)

# import sys
# from dataclasses import dataclass
# from pathlib import Path
# import pandas as pd
# import os

# from sklearn.model_selection import train_test_split
# from evidently.metrics import DataDriftTable, ColumnDriftMetric
# from evidently.report import Report

# from src.ReservationDiscounts.exception import CustomException
# from src.ReservationDiscounts.logger import logger
# from src.ReservationDiscounts.utils.commons import *
# from src.ReservationDiscounts.constants import *

# @dataclass
# class DataDriftConfig:
#     root_dir: Path
#     data_path: Path
#     random_state: int
#     target_col: str
#     numerical_cols: list
#     categorical_cols: list

# class ConfigurationManager:
#     def __init__(self, data_drift_config: str = DATA_DRIFT_CONFIG_FILEPATH):
#         try:
#             self.data_drift = read_yaml(data_drift_config)
#             create_directories([self.data_drift['artifacts_root']])
#         except Exception as e:
#             logger.exception(f"Error initializing data drift Configuration Manager")
#             raise CustomException(e, sys)

#     def get_data_drift_config(self) -> DataDriftConfig:
#         try:
#             drift_config = self.data_drift['data_drift']
#             create_directories([drift_config['root_dir']])

#             return DataDriftConfig(
#                 root_dir = drift_config['root_dir'],
#                 data_path = drift_config['data_path'],
#                 random_state = drift_config['random_state'],
#                 target_col = drift_config['target_col'],
#                 numerical_cols = drift_config['numerical_cols'],
#                 categorical_cols = drift_config['categorical_cols']
#             )

#         except Exception as e:
#             logger.exception(f"Error getting the Data Drift Config")
#             raise CustomException(e, sys)

# class DataDriftDetector:
#     def __init__(self, config: DataDriftConfig):
#         self.config = config

#     def load_data(self):
#         try:
#             logger.info("Loading data")
#             df = pd.read_parquet(self.config.data_path)
#             logger.info(f"Data shape: {df.shape}")
#             return df
#         except Exception as e:
#             logger.exception(f"Error loading data from {self.config.data_path}")
#             raise CustomException(e, sys)

#     def split_data(self, df):
#         try:
#             X = df.drop(self.config.target_col, axis=1)
#             y = df[self.config.target_col]

#             X_train, X_temp, y_train, y_temp = train_test_split(
#                 X, y, test_size=0.3, stratify=y, random_state=self.config.random_state
#             )
#             X_val, X_test, y_val, y_test = train_test_split(
#                 X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=self.config.random_state
#             )

#             logger.info("Data split into Train, Validation, and Test sets")
#             return X_train, X_val, X_test, y_train, y_val, y_test
#         except Exception as e:
#             logger.exception("Error splitting the data")
#             raise CustomException(e, sys)

#     def detect_drift(self, X_train, X_test, y_train, y_test):  # Keep y_train and y_test for comparison within ColumnDriftMetric
#         try:
#             logger.info("Performing Data Drift Detection using Evidently")

#             # Ensure output directory exists
#             os.makedirs(self.config.root_dir, exist_ok=True)

#             # Overall Data Drift (Feature Data)
#             drift_report = Report(
#                 metrics=[DataDriftTable(num_stattest='psi', cat_stattest='jensenshannon')]
#             )
#             drift_report.run(reference_data=X_train, current_data=X_test)
#             drift_report.save_html(os.path.join(self.config.root_dir, "overall_data_drift.html"))
#             logger.info("Overall Feature Data Drift Report saved.")

#             # Detect drift for numerical columns
#             for col in self.config.numerical_cols:
#                 num_drift_report = Report(metrics=[ColumnDriftMetric(column_name=col)])
#                 num_drift_report.run(reference_data=X_train, current_data=X_test)
#                 num_drift_report.save_html(os.path.join(self.config.root_dir, f"num_drift_{col}.html"))
#                 logger.info(f"Numerical Drift Report for {col} saved.")

#             # Detect drift for categorical columns
#             for col in self.config.categorical_cols:
#                 cat_drift_report = Report(metrics=[ColumnDriftMetric(column_name=col)])
#                 cat_drift_report.run(reference_data=X_train, current_data=X_test)
#                 cat_drift_report.save_html(os.path.join(self.config.root_dir, f"cat_drift_{col}.html"))
#                 logger.info(f"Categorical Drift Report for {col} saved.")

#             # Target Drift Analysis - Using ColumnDriftMetric
#             target_drift_report = Report(metrics=[ColumnDriftMetric(column_name=self.config.target_col)])
#             target_drift_report.run(reference_data=pd.DataFrame(y_train), current_data=pd.DataFrame(y_test)) #CHANGED X TRAIN AND X TEST
#             target_drift_report.save_html(os.path.join(self.config.root_dir, "target_drift.html"))
#             logger.info("Target Drift Report saved.")

#         except Exception as e:
#             logger.exception("Error during Data Drift Detection")
#             raise CustomException(e, sys)

# if __name__ == "__main__":
#     try:
#         config_manager = ConfigurationManager()
#         data_drift_config = config_manager.get_data_drift_config()

#         drift_detector = DataDriftDetector(data_drift_config)

#         df = drift_detector.load_data()
#         X_train, X_val, X_test, y_train, y_val, y_test = drift_detector.split_data(df)

#         # Run Data Drift Detection
#         drift_detector.detect_drift(X_train, X_test, y_train, y_test)

#     except Exception as e:
#         logger.error(f"Error in Data Drift Pipeline: {e}")
#         sys.exit(1)





## -------------WITHOUT IS COLUMN CREATING ERROR------------
# import sys 
# from dataclasses import dataclass 
# from pathlib import Path 
# import pandas as pd 
# import os

# from sklearn.model_selection import train_test_split 
# from evidently.metrics import DataDriftTable, ColumnDriftMetric
# from evidently.report import Report

# from src.ReservationDiscounts.exception import CustomException
# from src.ReservationDiscounts.logger import logger 
# from src.ReservationDiscounts.utils.commons import *
# from src.ReservationDiscounts.constants import *

# @dataclass 
# class DataDriftConfig:
#     root_dir: Path
#     data_path: Path
#     random_state: int
#     target_col: str
#     numerical_cols: list
#     categorical_cols: list

# class ConfigurationManager:
#     def __init__(self, data_drift_config: str = DATA_DRIFT_CONFIG_FILEPATH):
#         try:
#             self.data_drift = read_yaml(data_drift_config)
#             create_directories([self.data_drift['artifacts_root']])
#         except Exception as e:
#             logger.exception(f"Error initializing data drift Configuration Manager")
#             raise CustomException(e, sys)

#     def get_data_drift_config(self) -> DataDriftConfig:
#         try:
#             drift_config = self.data_drift['data_drift']
#             create_directories([drift_config['root_dir']])

#             return DataDriftConfig(
#                 root_dir = drift_config['root_dir'],
#                 data_path = drift_config['data_path'],
#                 random_state = drift_config['random_state'],
#                 target_col = drift_config['target_col'],
#                 numerical_cols = drift_config['numerical_cols'],
#                 categorical_cols = drift_config['categorical_cols']
#             )

#         except Exception as e:
#             logger.exception(f"Error getting the Data Drift Config")
#             raise CustomException(e, sys)

# class DataDriftDetector:
#     def __init__(self, config: DataDriftConfig):
#         self.config = config 

#     def load_data(self):
#         try:
#             logger.info("Loading data")
#             df = pd.read_parquet(self.config.data_path)
#             logger.info(f"Data shape: {df.shape}")
#             return df
#         except Exception as e:
#             logger.exception(f"Error loading data from {self.config.data_path}")
#             raise CustomException(e, sys)

#     def split_data(self, df):
#         try:
#             X = df.drop(self.config.target_col, axis=1)
#             y = df[self.config.target_col]

#             X_train, X_temp, y_train, y_temp = train_test_split(
#                 X, y, test_size=0.3, stratify=y, random_state=self.config.random_state
#             )
#             X_val, X_test, y_val, y_test = train_test_split(
#                 X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=self.config.random_state
#             )

#             logger.info("Data split into Train, Validation, and Test sets")
#             return X_train, X_val, X_test, y_train, y_val, y_test
#         except Exception as e:
#             logger.exception("Error splitting the data")
#             raise CustomException(e, sys)

#     def detect_drift(self, X_train, X_test):
#         try:
#             logger.info("Performing Data Drift Detection using Evidently")
            
#             # Ensure output directory exists
#             os.makedirs(self.config.root_dir, exist_ok=True)

#             # Overall Data Drift
#             drift_report = Report(
#                 metrics=[DataDriftTable(num_stattest='psi', cat_stattest='jensenshannon')]
#             )
#             drift_report.run(reference_data=X_train, current_data=X_test)
#             drift_report.save_html(os.path.join(self.config.root_dir, "overall_data_drift.html"))
#             logger.info("Overall Data Drift Report saved.")

#             # Detect drift for numerical columns
#             for col in self.config.numerical_cols:
#                 num_drift_report = Report(metrics=[ColumnDriftMetric(column_name=col)])
#                 num_drift_report.run(reference_data=X_train, current_data=X_test)
#                 num_drift_report.save_html(os.path.join(self.config.root_dir, f"num_drift_{col}.html"))
#                 logger.info(f"Numerical Drift Report for {col} saved.")

#             # Detect drift for categorical columns
#             for col in self.config.categorical_cols:
#                 cat_drift_report = Report(metrics=[ColumnDriftMetric(column_name=col)])
#                 cat_drift_report.run(reference_data=X_train, current_data=X_test)
#                 cat_drift_report.save_html(os.path.join(self.config.root_dir, f"cat_drift_{col}.html"))
#                 logger.info(f"Categorical Drift Report for {col} saved.")

#         except Exception as e:
#             logger.exception("Error during Data Drift Detection")
#             raise CustomException(e, sys)
        
# if __name__ == "__main__":
#     try:
#         config_manager = ConfigurationManager()
#         data_drift_config = config_manager.get_data_drift_config()

#         drift_detector = DataDriftDetector(data_drift_config)

#         df = drift_detector.load_data()
#         X_train, X_val, X_test, y_train, y_val, y_test = drift_detector.split_data(df)

#         # Run Data Drift Detection
#         drift_detector.detect_drift(X_train, X_test)  # FIXED: Removed X_val

#     except Exception as e:
#         logger.error(f"Error in Data Drift Pipeline: {e}")
#         sys.exit(1)







# import sys 
# from dataclasses import dataclass 
# from pathlib import Path 
# import pandas as pd 
# import joblib 

# from sklearn.model_selection import train_test_split 

# from evidently.metrics import DataDriftTable, ColumnDriftMetric
# from evidently.report import Report
# import os

# from src.ReservationDiscounts.exception import CustomException
# from src.ReservationDiscounts.logger import logger 
# from src.ReservationDiscounts.utils.commons import *
# from src.ReservationDiscounts.constants import *

# @dataclass 
# class DataDriftConfig:
#     root_dir: Path
#     data_path: Path
#     random_state: int
#     target_col: str
#     numerical_cols: list
#     categorical_cols: list

# class ConfigurationManager:
#     def __init__(self, data_drift_config: str = DATA_DRIFT_CONFIG_FILEPATH):
#         try:
#             self.data_drift = read_yaml(data_drift_config)
#             create_directories([self.data_drift['artifacts_root']])

#         except Exception as e:
#             logger.exception(f"Error initializing data drift Configuration Manager")
#             raise CustomException (e, sys)


#     def get_data_drift_config(self) -> DataDriftConfig:
#         try:
#             drift_config = self.data_drift['data_drift']
#             create_directories([drift_config['root_dir']])

#             drift_detection_config = DataDriftConfig(
#                 root_dir = drift_config['root_dir'],
#                 data_path = drift_config['data_path'],
#                 random_state = drift_config['random_state'],
#                 target_col = drift_config['target_col'],
#                 numerical_cols = drift_config['numerical_cols'],
#                 categorical_cols = drift_config['categorical_cols']
#             )
#             return drift_detection_config


#         except Exception as e:
#             logger.exception(f"Error getting the Data Drift Config")
#             raise CustomException(e, sys)

# class DataDriftDetector:
#     def __init__(self, config:DataDriftConfig):
#         self.config = config 

#     def load_data(self) -> None:
#         try:
#             logger.info("Loading data")

#             # Load data 
#             data_path = self.config.data_path

#             try:
#                 with open(data_path, 'rb') as f:
#                     df = pd.read_parquet(f)
#                 logger.info(f"Data shape: {df.shape}")

#             except Exception as e:
#                 logger.exception(f"Error loading data from {data_path}")
#                 raise CustomException(e, sys)
            
#             return df

#         except Exception as e:
#             logger.exception(f"Error Loading the data")
#             raise CustomException(e, sys)
        
#     def split_data(self, df)-> None:
#         try:
#             # Define features and target variable 
#             X = df.drop(self.config.target_col, axis=1)
#             y = df[self.config.target_col]

#             # Split data into training, testing and validation 
#             X_train, X_temp, y_train, y_temp = train_test_split(
#                 X, y, test_size=0.3, stratify=y, random_state=self.config.random_state
#             )
#             # Split the temporary set into validation and testing 
#             X_val, X_test, y_val, y_test = train_test_split(
#                 X_temp, y_temp, test_size=0.5, stratify= y_temp, random_state=self.config.random_state
#             )

#             logger.info("Saving the training, validation and testing data")

#             return X_train, X_val, X_test, y_train, y_val, y_test
#         except Exception as e:
#             logger.exception(f"Error splitting the data")
#             raise CustomException(e, sys)



# class DataDriftDetector:
#     def __init__(self, config: DataDriftConfig):
#         self.config = config 

#     def detect_drift(self, X_train, X_test):
#         try:
#             logger.info("Performing Data Drift Detection using Evidently")

#             # Create Evidently report for overall drift
#             drift_report = Report(
#                 metrics=[
#                     DataDriftTable(num_stattest='psi', cat_stattest='jensenshannon'),  # Overall Data Drift
#                 ]
#             )
#             drift_report.run(reference_data=X_train, current_data=X_test)

#             # Save Overall Report
#             overall_report_path = os.path.join(self.config.root_dir, "overall_data_drift.html")
#             drift_report.save_html(overall_report_path)
#             logger.info(f"Overall Data Drift Report saved at: {overall_report_path}")

#             # Detect drift for numerical columns
#             for col in self.config.numerical_cols:
#                 num_drift_report = Report(metrics=[ColumnDriftMetric(column_name=col)])
#                 num_drift_report.run(reference_data=X_train, current_data=X_test)

#                 num_report_path = os.path.join(self.config.root_dir, f"num_drift_{col}.html")
#                 num_drift_report.save_html(num_report_path)
#                 logger.info(f"Numerical Drift Report for {col} saved at: {num_report_path}")

#             # Detect drift for categorical columns
#             for col in self.config.categorical_cols:
#                 cat_drift_report = Report(metrics=[ColumnDriftMetric(column_name=col)])
#                 cat_drift_report.run(reference_data=X_train, current_data=X_test)

#                 cat_report_path = os.path.join(self.config.root_dir, f"cat_drift_{col}.html")
#                 cat_drift_report.save_html(cat_report_path)
#                 logger.info(f"Categorical Drift Report for {col} saved at: {cat_report_path}")

#         except Exception as e:
#             logger.exception("Error during Data Drift Detection")
#             raise CustomException(e, sys)
        
# if __name__ == "__main__":
#     try:
#         config_manager = ConfigurationManager()
#         data_drift_config = config_manager.get_data_drift_config()

#         drift_detector = DataDriftDetector(data_drift_config)

#         df = drift_detector.load_data()
#         X_train, X_val, X_test, y_train, y_val, y_test = drift_detector.split_data(df)

#         # Run Data Drift Detection
#         drift_detector.detect_drift(X_train, X_val, X_test)

#     except Exception as e:
#         logger.error(f"Error in Data Drift Pipeline: {e}")
#         sys.exit(1)

