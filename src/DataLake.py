"""
# DataLake Class Explanation

## Overview
The `DataLake` class provides a comprehensive framework for managing datasets in a file-based Data Lake. It allows for storing, retrieving, managing metadata, and querying datasets using SQL. The design emphasizes security via access keys and supports both raw and processed datasets.
Note the security feature that restricts access to certain methods based on the access key provided. Tested using unit tests (can refer for example usage). 

ALternatively, the QuantAnalyst.py file showcases a comprehensive usage example and shows the entire inegration of DataLake with other classes.
---

## Key Features and Methods

### 1. Initialization (`__init__`)
- Creates the base directory (`data_lake` by default).
- Initializes folders for raw and processed data.
- Loads metadata from a `metadata.json` file.

### 2. Metadata Management
- **`_load_metadata`:** Loads dataset metadata from the JSON file.
- **`_save_metadata`:** Saves metadata back to the file after any changes.
- Metadata includes details like dataset structure, author, processing status, and timestamps.

### 3. File Path Construction
- **`__get_file_path`:** Constructs file paths for datasets based on their name and whether they are raw or processed.

### 4. Data Storage and Retrieval
- **`store_data`:** 
  - Saves datasets (DataFrame, dict, list, or text/binary) to disk.
  - Updates metadata with details like data structure and author.
  - Prevents overwriting unless `force=True` is specified.

- **`retrieve_data`:**
  - Fetches datasets from disk based on their name and type (raw or processed).

### 5. Security: Access Control
- **`access_decorator`:** 
  - Ensures that only users with valid access keys can call restricted methods like `store_data`, `retrieve_data`, or `delete_dataset`.

### 6. SQL Query Execution
- **`execute_sql`:**
  - Executes SQL queries across all loaded DataFrames in the Data Lake.
  - Uses the `pandasql` library for SQL-like querying.

### 7. Dataset Management
- **`list_datasets`:** Lists all available datasets, optionally filtering by processed or raw type.
- **`delete_dataset`:** Deletes a dataset from both disk and metadata.

### 8. Enhanced Metadata Retrieval
- **`get_metadata`:**
  - Fetches metadata for a specific dataset.
  - Reloads metadata from the file if it's not found in memory.
"""

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
        """
        Retrieve metadata for a dataset. If metadata is not in memory, reload from the metadata file.

        Args:
            dataset_name (str): Name of the dataset.
            access_key (int): Access key to access the metadata.

        Returns:
            Optional[Dict]: Metadata dictionary or None if not found.
        """
        # Check if metadata exists in memory
        metadata = self.metadata.get(dataset_name)
        if metadata:
            return metadata

        # Reload metadata from the file if not found in memory
        print(f"Metadata for dataset '{dataset_name}' not found in memory. Reloading metadata from file...")
        self.metadata = self._load_metadata()  # Reload metadata from the file

        # Check again after reloading
        metadata = self.metadata.get(dataset_name)
        if metadata:
            return metadata

        print(f"Metadata for dataset '{dataset_name}' still not found after reloading.")
        return None

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



# #old

# import numpy as np
# import pandas as pd
# import polars as pl 
# import os
# from typing import Tuple, List, Dict,Optional, Any
# import json
# import pickle
# import datetime
# from pandasql import sqldf



# class DataLake:
#     secured_access = {134:"Andy", 245: "Matt", 367: "Harry"}
#     def __init__(self, base_path: str = "data_lake"):
#         """
#         Initialize the Data Lake.

#         Args:
#             base_path (str): Path to store raw and processed data.
#         """
#         self.base_path = base_path
#         os.makedirs(f"{self.base_path}/raw", exist_ok=True)
#         os.makedirs(f"{self.base_path}/processed", exist_ok=True)
#         self.metadata = {}  # Metadata storage for datasets

#     def __get_file_path(self, dataset_name: str, processed: bool) -> str:
#         """
#         Construct file path for a dataset.

#         Args:
#             dataset_name (str): Name of the dataset.
#             processed (bool): Flag indicating if the data is processed.

#         Returns:
#             str: File path.
#         """
#         folder = "processed" if processed else "raw"
#         return os.path.join(self.base_path, folder, f"{dataset_name}.pkl")
    
#     @staticmethod
#     def access_decorator(func):
#         def wrapper(self, *args, **kwargs):
#             access_key = kwargs.get("access_key") 
#             if access_key not in DataLake.secured_access or access_key == -1:
#                 print("Access Denied: Invalid access key.")
#                 return None
#             return func(self, *args, **kwargs)
#         return wrapper
#     @access_decorator
#     def store_data(
#         self,
#         dataset_name: str,
#         data: Any,
#         access_key: int = -1,
#         processed: bool = False,
#         metadata: Optional[Dict] = None,
#         force: bool = False  # Optional parameter to bypass overwrite prompt
#     ):
#         """
#         Store a dataset in the Data Lake using pickle, supporting structured, semi-structured, and unstructured data.

#         Args:
#             dataset_name (str): Name of the dataset.
#             data (Any): Data to store (can be DataFrame, dict, list, str, bytes).
#             access_key (int): Access key to edit the data.
#             processed (bool): Flag indicating if the data is processed.
#             metadata (Optional[Dict]): Additional metadata for the dataset.
#             force (bool): If True, bypass overwrite confirmation prompt.
#         """
#         file_path = None
#         data_type = type(data).__name__
#         data_structure = {}

#         # Check if dataset already exists
#         if dataset_name in self.metadata:
#             if not force:
#                 print(f"Dataset '{dataset_name}' already exists. Do you want to update the data?")
#                 response = input("Type 'yes' to proceed: ").strip().lower()
#                 if response != "yes":
#                     print("Data not stored.")
#                     return
#             else:
#                 print(f"Overwriting existing dataset '{dataset_name}'...")

#         # Generate file path
#         try:
#             file_path = self.__get_file_path(dataset_name, processed)
#         except Exception as e:
#             print(f"Failed to get file path: {e}")
#             return

#         # Collect metadata
#         modification_time = datetime.datetime.now().isoformat()
#         try:
#             if isinstance(data, pd.DataFrame):
#                 data_structure = {
#                     "columns": list(data.columns),
#                     "data_types": data.dtypes.astype(str).to_dict(),
#                     "row_count": len(data),
#                     "index": data.index.name,
#                 }
#             elif isinstance(data, (dict, list)):
#                 data_structure = {"data_type": "JSON-like", "item_count": len(data)}
#             elif isinstance(data, (str, bytes)):
#                 data_structure = {"data_type": "Text/Binary"}

#             temp_metadata = metadata or {}
#             temp_metadata.update({
#                 "Author": DataLake.secured_access.get(access_key, "Unknown"),
#                 "processed": processed,
#                 "modification_time": modification_time,
#                 "data_type": data_type,
#                 "data_structure": data_structure,
#             })

#         except Exception as e:
#             print(f"Failed to create metadata for dataset '{dataset_name}': {e}")
#             print("Metadata not updated. Data not stored.")
#             return

#         # Write data and update metadata
#         try:
#             with open(file_path, "wb") as f:
#                 pickle.dump(data, f)
#                 print(f"Data stored at: {file_path}")

#             # Update file size metadata
#             file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
#             temp_metadata["file_size"] = f"{file_size / 1024:.2f} KB" if file_size else "Unknown"

#             # Assign metadata
#             self.metadata[dataset_name] = temp_metadata
#             print(f"Metadata updated: {self.metadata[dataset_name]}")

#         except Exception as e:
#             print(f"Failed to store data: {e}")
#             print("Data not stored.")

        
#     @access_decorator
#     def retrieve_data(self, dataset_name: str, access_key: int = -1, processed: bool = False) -> Optional[Any]:
#         """
#         Retrieve a dataset from the Data Lake using pickle.

#         Args:
#             dataset_name (str): Name of the dataset.
#             access_key (int): Access key to access the data.
#             processed (bool): Flag indicating if the data is processed.

#         Returns:
#             Optional[Any]: Retrieved data or None if not found.
#         """
#         file_path = self.__get_file_path(dataset_name, processed)

#         try:
#             # Check if the file exists
#             if os.path.exists(file_path):
#                 print(f"Retrieving dataset '{dataset_name}' from '{file_path}'.")
#                 # Load the data using pickle
#                 with open(file_path, "rb") as f:
#                     return pickle.load(f)
#             else:
#                 print(f"Dataset '{dataset_name}' not found.")
#                 return None

#         except Exception as e:
#             print(f"Failed to retrieve dataset '{dataset_name}': {e}")
#             return None
#     @access_decorator
#     def execute_sql(self, sql_query: str, access_key: int = -1) -> Optional[pd.DataFrame]:
#         """
#         Execute an SQL query on all datasets loaded in the Data Lake.

#         Args:
#             sql_query (str): SQL query string.
#             access_key (int): Access key to access the data.

#         Returns:
#             Optional[pd.DataFrame]: Result of the SQL query or None if error occurs.
#         """
#         try:
#             # Load all datasets into a namespace for querying
#             namespace = {}
#             for dataset_name, metadata in self.metadata.items():
#                 if metadata.get("data_type") == "DataFrame":  # Only load DataFrames
#                     file_path = self.__get_file_path(dataset_name, metadata["processed"])
#                     if os.path.exists(file_path):
#                         try:
#                             with open(file_path, "rb") as f:
#                                 namespace[dataset_name] = pickle.load(f)
#                         except Exception as e:
#                             print(f"Failed to load DataFrame '{dataset_name}': {e}")
#                             return None
            
#             if not namespace:
#                 print("No DataFrame datasets available for SQL query")
#                 return None

#             # Execute SQL query using pandasql
#             result = sqldf(sql_query, namespace)
            
#             if result is None or result.empty:
#                 print("SQL query returned no results")
#                 return None
                
#             return result
        
#         except Exception as e:
#             print(f"SQL query error: {e}")
#             return None
#     @access_decorator
#     def list_datasets(self, access_key: int = -1, processed: Optional[bool] = None) -> List[str]:
#         """
#         List available datasets in the Data Lake.

#         Args:
#             access_key (int): Access key to access the data.
#             processed (Optional[bool]): Filter datasets by type.

#         Returns:
#             list: List of dataset names.
#         """
#         datasets = []
#         for dataset_name, metadata in self.metadata.items():
#             if processed is None or metadata["processed"] == processed:
#                 datasets.append(dataset_name)
#         return datasets
#     @access_decorator
#     def get_metadata(self, dataset_name: str, access_key: int = -1) -> Optional[Dict]:
#         """
#         Retrieve metadata for a dataset.

#         Args:
#             dataset_name (str): Name of the dataset.
#             access_key (int): Access key to access the metadata.

#         Returns:
#             Optional[Dict]: Metadata dictionary or None if not found.
#         """
#         metadata = self.metadata.get(dataset_name)
#         if metadata:
#             return metadata
#         print(f"Metadata for dataset '{dataset_name}' not found.")
#         return None
    
#     @access_decorator
#     def delete_dataset(self, dataset_name: str, access_key: int = -1, processed: bool = False):
#         """
#         Delete a dataset from the Data Lake.

#         Args:
#             dataset_name (str): Name of the dataset.
#             access_key (int): Access key to delete the dataset.
#             processed (bool): Flag indicating if the data is processed.
#         """
#         file_path = self.__get_file_path(dataset_name, processed)
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             self.metadata.pop(dataset_name, None)
#             print(f"Dataset '{dataset_name}' deleted.")
#         else:
#             print(f"Dataset '{dataset_name}' not found.")