

from dataclasses import dataclass
from pathlib import Path


# Data Ingestion Config
@dataclass
class DataIngestionConfig:
    root_dir: str
    database_name: str
    collection_name: str
    batch_size: int
    mongo_uri: str

# Data Validation Config 
@dataclass
class DataValidationConfig:
    root_dir: str
    data_dir: str
    val_status: str
    all_schema: dict
    validated_data: str
    profile_report_name: str


# Data Drift Config
@dataclass
class DataDriftConfig:
    root_dir: Path
    data_path: Path
    random_state: int
    target_col: str
    numerical_cols: list
    categorical_cols: list
