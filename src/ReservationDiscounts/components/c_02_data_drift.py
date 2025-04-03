
import sys
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
from src.ReservationDiscounts.config_entity.config_params import DataDriftConfig


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

