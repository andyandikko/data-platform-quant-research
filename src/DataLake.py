import os
import json
import pickle
import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
from pandasql import sqldf


class DataLake:
    secured_access = {134: "Andy", 245: "Matt", 367: "Harry"}
    METADATA_FILE = "metadata.json"

    def __init__(self, base_path: str = "data_lake"):
        """
        Initialize the Data Lake.

        Args:
            base_path (str): Path to store raw and processed data.
        """
        self.base_path = base_path
        os.makedirs(f"{self.base_path}/raw", exist_ok=True)
        os.makedirs(f"{self.base_path}/processed", exist_ok=True)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from a file if it exists."""
        if os.path.exists(DataLake.METADATA_FILE):
            try:
                with open(DataLake.METADATA_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load metadata: {e}")
        return {}

    def _save_metadata(self):
        """Persist metadata to a file."""
        try:
            with open(DataLake.METADATA_FILE, "w") as f:
                json.dump(self.metadata, f, indent=4)
                print("Metadata saved successfully.")
        except Exception as e:
            print(f"Failed to save metadata: {e}")

    def __get_file_path(self, dataset_name: str, processed: bool) -> str:
        """Construct the file path for a dataset."""
        folder = "processed" if processed else "raw"
        return os.path.join(self.base_path, folder, f"{dataset_name}.pkl")

    @staticmethod
    def access_decorator(func):
        def wrapper(self, *args, **kwargs):
            access_key = kwargs.get("access_key")
            if access_key not in DataLake.secured_access or access_key == -1:
                print("Access Denied: Invalid access key.")
                return None
            return func(self, *args, **kwargs)
        return wrapper

    @access_decorator
    def store_data(
        self,
        dataset_name: str,
        data: Any,
        access_key: int = -1,
        processed: bool = False,
        metadata: Optional[Dict] = None,
        force: bool = False
    ):
        """Store a dataset in the Data Lake."""
        file_path = self.__get_file_path(dataset_name, processed)
        data_type = type(data).__name__
        data_structure = {}

        # Check if dataset already exists
        if dataset_name in self.metadata and not force:
            print(f"Dataset '{dataset_name}' already exists. Use force=True to overwrite.")
            return

        # Collect metadata
        modification_time = datetime.datetime.now().isoformat()
        try:
            if isinstance(data, pd.DataFrame):
                data_structure = {
                    "columns": list(data.columns),
                    "data_types": data.dtypes.astype(str).to_dict(),
                    "row_count": len(data),
                    "index": data.index.name,
                }
            elif isinstance(data, (dict, list)):
                data_structure = {"data_type": "JSON-like", "item_count": len(data)}
            elif isinstance(data, (str, bytes)):
                data_structure = {"data_type": "Text/Binary"}

            temp_metadata = metadata or {}
            temp_metadata.update({
                "Author": DataLake.secured_access.get(access_key, "Unknown"),
                "processed": processed,
                "modification_time": modification_time,
                "data_type": data_type,
                "data_structure": data_structure,
            })

        except Exception as e:
            print(f"Failed to create metadata for dataset '{dataset_name}': {e}")
            return

        # Write data and update metadata
        try:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
                print(f"Data stored at: {file_path}")

            # Update metadata and persist
            self.metadata[dataset_name] = temp_metadata
            self._save_metadata()
        except Exception as e:
            print(f"Failed to store data: {e}")

    @access_decorator
    def retrieve_data(self, dataset_name: str, access_key: int = -1, processed: bool = False) -> Optional[Any]:
        """Retrieve a dataset from the Data Lake."""
        file_path = self.__get_file_path(dataset_name, processed)
        try:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            else:
                print(f"Dataset '{dataset_name}' not found.")
        except Exception as e:
            print(f"Failed to retrieve dataset '{dataset_name}': {e}")
        return None

    @access_decorator
    def execute_sql(self, sql_query: str, access_key: int = -1) -> Optional[pd.DataFrame]:
        """Execute an SQL query on all datasets loaded in the Data Lake."""
        try:
            # Load all datasets into a namespace for querying
            namespace = {}
            for dataset_name, metadata in self.metadata.items():
                if metadata.get("data_type") == "DataFrame":
                    file_path = self.__get_file_path(dataset_name, metadata["processed"])
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as f:
                            namespace[dataset_name] = pickle.load(f)

            if not namespace:
                print("No DataFrame datasets available for SQL query.")
                return None

            result = sqldf(sql_query, namespace)
            if result.empty:
                print("SQL query returned no results.")
            return result

        except Exception as e:
            print(f"SQL query error: {e}")
            return None

    @access_decorator
    def list_datasets(self, access_key: int = -1, processed: Optional[bool] = None) -> List[str]:
        """List available datasets in the Data Lake."""
        return [
            dataset_name
            for dataset_name, metadata in self.metadata.items()
            if processed is None or metadata["processed"] == processed
        ]

    @access_decorator
    def get_metadata(self, dataset_name: str, access_key: int = -1) -> Optional[Dict]:
        """Retrieve metadata for a dataset."""
        return self.metadata.get(dataset_name)

    @access_decorator
    def delete_dataset(self, dataset_name: str, access_key: int = -1, processed: bool = False):
        """Delete a dataset from the Data Lake."""
        file_path = self.__get_file_path(dataset_name, processed)
        if os.path.exists(file_path):
            os.remove(file_path)
            self.metadata.pop(dataset_name, None)
            self._save_metadata()
            print(f"Dataset '{dataset_name}' deleted.")
        else:
            print(f"Dataset '{dataset_name}' not found.")
