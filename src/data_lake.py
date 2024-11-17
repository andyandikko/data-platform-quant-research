import numpy as np
import pandas as pd
import polars as pl 
import os
from typing import tuple, List, Dict,Optional, Any
import json
import pickle
import datetime
import sqldf


class DataLake:
    secured_access = {134:"Andy", 245: "Matheus", 367: "Harrison"}
    def __init__(self, base_path: str = "data_lake"):
        """
        Initialize the Data Lake.

        Args:
            base_path (str): Path to store raw and processed data.
        """
        self.base_path = base_path
        os.makedirs(f"{self.base_path}/raw", exist_ok=True)
        os.makedirs(f"{self.base_path}/processed", exist_ok=True)
        self.metadata = {}  # Metadata storage for datasets

    def __get_file_path(self, dataset_name: str, processed: bool) -> str:
        """
        Construct file path for a dataset.

        Args:
            dataset_name (str): Name of the dataset.
            processed (bool): Flag indicating if the data is processed.

        Returns:
            str: File path.
        """
        folder = "processed" if processed else "raw"
        return os.path.join(self.base_path, folder, f"{dataset_name}.parquet")
    
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
    ):
        """
        Store a dataset in the Data Lake using pickle, supporting structured, semi-structured, and unstructured data.

        Args:
            dataset_name (str): Name of the dataset.
            data (Any): Data to store (can be DataFrame, dict, list, str, bytes).
            access_key (int): Access key to edit the data.
            processed (bool): Flag indicating if the data is processed.
            metadata (Optional[Dict]): Additional metadata for the dataset.
        """
        file_path = None
        data_type = type(data).__name__
        file_size = 0
        data_structure = {}
        if dataset_name in self.metadata:
            print(f"Dataset '{dataset_name}' already exists. Do you want to update the data.")
            response = input("Type Yes to proceed...")
            if response.lower() != "yes":
                print("Data not stored.")
                return

        try:
            file_path = self.__get_file_path(dataset_name, "pkl", processed)
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            # Additional metadata for structured data (if it's a DataFrame)
            if isinstance(data, pd.DataFrame):
                data_structure = {
                    "columns": list(data.columns),
                    "data_types": data.dtypes.astype(str).to_dict(),
                    "row_count": len(data),
                    "index": data.index.name,
                }
            # General metadata for semi-structured data (dict, list)
            elif isinstance(data, (dict, list)):
                data_structure = {"data_type": "JSON-like", "item_count": len(data)}

            # General metadata for unstructured data
            elif isinstance(data, (str, bytes)):
                data_structure = {"data_type": "Text/Binary"}

            # Get file size
            file_size = os.path.getsize(file_path)

        except Exception as e:
            print(f"Failed to store data: {e}")
            file_path = None  # Data not stored
            print("Data not stored.")
            return

        # Collect metadata
        modification_time = datetime.now().isoformat()
        self.metadata[dataset_name] = metadata or {}
        self.metadata[dataset_name]["Author"] = DataLake.secured_access.get(access_key, "Unknown")
        self.metadata[dataset_name]["processed"] = processed
        self.metadata[dataset_name]["file_size"] = f"{file_size / 1024:.2f} KB" if file_size else "Unknown"
        self.metadata[dataset_name]["modification_time"] = modification_time
        self.metadata[dataset_name]["data_type"] = data_type
        self.metadata[dataset_name]["data_structure"] = data_structure

        # Print metadata and file path
        if file_path:
            print(f"Data stored at: {file_path}")
        print(f"Metadata updated: {self.metadata[dataset_name]}")
    @access_decorator
    def retrieve_data(self, dataset_name: str, access_key: int = -1, processed: bool = False) -> Optional[Any]:
        """
        Retrieve a dataset from the Data Lake using pickle.

        Args:
            dataset_name (str): Name of the dataset.
            access_key (int): Access key to access the data.
            processed (bool): Flag indicating if the data is processed.

        Returns:
            Optional[Any]: Retrieved data or None if not found.
        """
        file_path = self.__get_file_path(dataset_name, processed)

        try:
            # Check if the file exists
            if os.path.exists(file_path):
                print(f"Retrieving dataset '{dataset_name}' from '{file_path}'.")
                # Load the data using pickle
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            else:
                print(f"Dataset '{dataset_name}' not found.")
                return None

        except Exception as e:
            print(f"Failed to retrieve dataset '{dataset_name}': {e}")
            return None
    @access_decorator
    def execute_sql(self, sql_query: str, access_key: int = -1) -> Optional[pd.DataFrame]:
        """
        Execute an SQL query on all datasets loaded in the Data Lake.

        Args:
            sql_query (str): SQL query string.
            access_key (int): Access key to access the data.

        Returns:
            Optional[pd.DataFrame]: Result of the SQL query or None if error occurs.
        """
        try:
            # Load all datasets into a namespace for querying
            namespace = {}
            for dataset_name, metadata in self.metadata.items():
                file_path = self.__get_file_path(dataset_name, metadata["processed"])
                if os.path.exists(file_path) and metadata["data_type"] == "DataFrame":
                    namespace[dataset_name] = pickle.load(open(file_path, "rb"))
            # Execute SQL query using pandasql
            return sqldf(sql_query, namespace)
        except Exception as e:
            print(f"SQL query error: {e}")
            return None
    @access_decorator
    def list_datasets(self, access_key: int = -1, processed: Optional[bool] = None) -> List[str]:
        """
        List available datasets in the Data Lake.

        Args:
            access_key (int): Access key to access the data.
            processed (Optional[bool]): Filter datasets by type.

        Returns:
            list: List of dataset names.
        """
        datasets = []
        for dataset_name, metadata in self.metadata.items():
            if processed is None or metadata["processed"] == processed:
                datasets.append(dataset_name)
        return datasets
    @access_decorator
    def get_metadata(self, dataset_name: str, access_key: int = -1) -> Optional[Dict]:
        """
        Retrieve metadata for a dataset.

        Args:
            dataset_name (str): Name of the dataset.
            access_key (int): Access key to access the metadata.

        Returns:
            Optional[Dict]: Metadata dictionary or None if not found.
        """
        metadata = self.metadata.get(dataset_name)
        if metadata:
            return metadata
        print(f"Metadata for dataset '{dataset_name}' not found.")
        return None
    
    @access_decorator
    def delete_dataset(self, dataset_name: str, access_key: int = -1, processed: bool = False):
        """
        Delete a dataset from the Data Lake.

        Args:
            dataset_name (str): Name of the dataset.
            access_key (int): Access key to delete the dataset.
            processed (bool): Flag indicating if the data is processed.
        """
        file_path = self.__get_file_path(dataset_name, processed)
        if os.path.exists(file_path):
            os.remove(file_path)
            self.metadata.pop(dataset_name, None)
            print(f"Dataset '{dataset_name}' deleted.")
        else:
            print(f"Dataset '{dataset_name}' not found.")





