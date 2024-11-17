"""
Explanation of DataWorkbench Class

The DataWorkbench class extends the functionality of the DataLake class to enhance data management capabilities by adding in-memory storage, transformation functionalities, and processing logs. It integrates well with the existing DataLake for seamless data retrieval and storage. Below is a concise explanation of its components and functionalities.

---

Key Features

1. In-Memory Storage:
   - Stores datasets temporarily in memory for quick access.
   - Includes metadata about the datasets such as storage time and additional properties.

   Key Methods:
   - store_data_in_memory: Saves data and metadata in memory for efficient retrieval.
   - retrieve_data_in_memory: Fetches data from memory and validates it if necessary.
   - clean_memory_storage: Cleans in-memory data storage based on a timestamp or completely clears it.

---

2. Transformation Management:
   - Allows registering transformation functions and applying them to datasets.
   - Supports both single and chained transformations.
   - Transformations can be applied in memory or stored in the DataLake.

   Key Methods:
   - register_transformation: Registers a named transformation with optional metadata (e.g., description).
   - transform_data: Applies a single transformation (by name or function) to a dataset, with options to store the transformed data.
   - chain_transformations: Applies multiple transformations sequentially, with options to store the final output in the DataLake.

---

3. Processing Logs:
   - Tracks all transformations applied to datasets, including the timestamp and transformation details.
   - Useful for auditing and debugging transformation workflows.

   Key Methods:
   - _log_processing: Internally logs each transformation step applied to a dataset.
   - get_processing_history: Retrieves the processing history for a specific dataset or all datasets.

---

How It Works

1. Data Storage:
   - Data can be stored in memory or in the DataLake for persistent storage.
   - Metadata about the datasets is maintained alongside the data.

2. Transformations:
   - Users can define custom transformations (e.g., lambda df: df.assign(new_col=df['existing_col'] * 2)).
   - These transformations can be applied either individually or chained together for complex workflows.
   - Transformed data can be saved back to the DataLake or retained in memory for subsequent use.

3. Integration with DataLake:
   - The class builds on the capabilities of the DataLake for persistent storage and metadata handling.
   - It leverages DataLake methods to store and retrieve datasets while adding in-memory options and processing logs.

---
"""

from typing import Any, Dict, List, Optional, Callable, Union
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from functools import wraps
from src.DataLake import DataLake


class DataWorkbench(DataLake):
    """Enhanced DataWorkbench with better integration features."""
    
    def __init__(self, base_path: str = "data_lake"):
        super().__init__(base_path)
        self.data_storage = {}  # In-memory storage
        self.transformations = {}  # Registered transformations
        self.processing_logs = []  # Track processing history

    def register_transformation(
        self,
        name: str,
        func: Callable,
        description: str = None
    ) -> None:
        """Register a named transformation function."""
        self.transformations[name] = {
            "func": func,
            "description": description,
            "registered_at": datetime.now().isoformat()
        }

    def _log_processing(self, dataset_name: str, transformation: str) -> None:
        """Log processing steps."""
        self.processing_logs.append({
            "dataset": dataset_name,
            "transformation": transformation,
            "timestamp": datetime.now().isoformat()
        })

    def store_data_in_memory(
        self,
        dataset_name: str,
        data: Any,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Enhanced in-memory storage with metadata."""
        if data is None:
            print("Invalid data provided.")
            return False
        try:
            self.data_storage[dataset_name] = {
                "data": data,
                "metadata": metadata or {},
                "cached_at": datetime.now().isoformat()
            }
            return True
        except Exception as e:
            print(f"Failed to store in memory: {e}")
            return False

    def retrieve_data_in_memory(
        self, 
        dataset_name: str,
        validate: bool = True
    ) -> Optional[Any]:
        """Enhanced memory retrieval with validation."""
        stored = self.data_storage.get(dataset_name)
        if stored is None:
            return None
            
        if validate and isinstance(stored["data"], pd.DataFrame):
            if stored["data"].empty:
                return None
                
        return stored["data"]

    def transform_data(
        self,
        dataset_name: str,
        transformation_func: Union[Callable, str],
        access_key: int = -1,
        update_in_datalake: bool = False,
        store_in_memory: bool = True,
        metadata: Optional[Dict] = None
    ) -> Optional[Any]:
        """Enhanced transformation with multiple storage options."""
        # Get data (try memory first, then DataLake)
        data = self.retrieve_data_in_memory(dataset_name)
        if data is None:
            data = self.retrieve_data(dataset_name, access_key=access_key)
            if data is None:
                return None

        try:
            # Get transformation function
            if isinstance(transformation_func, str):
                if transformation_func not in self.transformations:
                    raise ValueError(f"Unknown transformation: {transformation_func}")
                func = self.transformations[transformation_func]["func"]
            elif callable(transformation_func):
                func = transformation_func
            else:
                raise ValueError(f"Invalid transformation function: {transformation_func}")

            # Apply transformation
            transformed_data = func(data)
            self._log_processing(dataset_name, str(func.__name__))

            # Store results based on flags
            if store_in_memory:
                self.store_data_in_memory(
                    f"{dataset_name}_transformed",
                    transformed_data,
                    metadata
                )

            if update_in_datalake:
                self.store_data(
                    dataset_name=f"{dataset_name}_transformed",
                    data=transformed_data,
                    access_key=access_key,
                    processed=True,
                    metadata=metadata,
                    force=True
                )

            return transformed_data

        except Exception as e:
            print(f"Transformation failed: {e}")
            return None

    def chain_transformations(
        self,
        dataset_name: str,
        transformations: List[Union[Callable, str]],
        access_key: int = -1,
        update_in_datalake: bool = False
    ) -> Optional[Any]:
        """Apply multiple transformations in sequence."""
        data = self.retrieve_data_in_memory(dataset_name)
        if data is None:
            data = self.retrieve_data(dataset_name, access_key=access_key)
            if data is None:
                return None

        try:
            for transform in transformations:
                data = self.transform_data(
                    dataset_name=dataset_name,
                    transformation_func=transform,
                    access_key=access_key,
                    update_in_datalake=False,  # Only update at the end
                    store_in_memory=True
                )
                if data is None:
                    return None

            if update_in_datalake:
                self.store_data(
                    dataset_name=f"{dataset_name}_chain_transformed",
                    data=data,
                    access_key=access_key,
                    processed=True,
                    force=True
                )

            return data

        except Exception as e:
            print(f"Chain transformation failed: {e}")
            return None

    def get_processing_history(
        self, 
        dataset_name: Optional[str] = None
    ) -> List[Dict]:
        """Get processing history for a dataset or all datasets."""
        if dataset_name:
            return [log for log in self.processing_logs 
                   if log["dataset"] == dataset_name]
        return self.processing_logs

    def clean_memory_storage(
        self, 
        older_than: Optional[str] = None
    ) -> None:
        """Clean in-memory storage."""
        if older_than:
            cutoff = pd.to_datetime(older_than)
            self.data_storage = {
                k: v for k, v in self.data_storage.items()
                if pd.to_datetime(v["cached_at"]) > cutoff
            }
        else:
            self.data_storage.clear()
            
            
            