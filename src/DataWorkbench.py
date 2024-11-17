from typing import Callable
from src.DataLake import DataLake

class DataWorkbench(DataLake):
    def __init__(self, base_path: str = "data_lake"):
        """
        Initialize the Data Workbench, inheriting from DataLake.

        Args:
            base_path (str): Path to store raw and processed data.
        """
        super().__init__(base_path)
        self.data_storage = {}

    def store_data_in_memory(self, dataset_name: str, data: Any) -> None:
        """
        Store a dataset in memory for immediate access.

        Args:
            dataset_name (str): Name of the dataset.
            data (Any): Data to store (e.g., DataFrame, dict, list).
        """
        self.data_storage[dataset_name] = data
        print(f"Dataset '{dataset_name}' stored in memory successfully.")

    def retrieve_data_in_memory(self, dataset_name: str) -> Optional[Any]:
        """
        Retrieve a dataset from memory.

        Args:
            dataset_name (str): Name of the dataset.

        Returns:
            Optional[Any]: Retrieved data or a message if not found.
        """
        data = self.data_storage.get(dataset_name)
        if data is None:
            print(f"Dataset '{dataset_name}' not found in memory.")
            return None
        return data

    def transform_data(
        self, 
        dataset_name: str, 
        transformation_func: Callable, 
        access_key: int = -1, 
        update_in_datalake: bool = False
    ) -> Optional[Any]:
        """
        Apply a transformation function to a dataset.

        Args:
            dataset_name (str): Name of the dataset.
            transformation_func (Callable): Function to transform the dataset.
            access_key (int): Access key for secure access.
            update_in_datalake (bool): Whether to update the dataset in the Data Lake with the transformed data.

        Returns:
            Optional[Any]: Transformed data or a message if not found.
        """
        data = self.retrieve_data(dataset_name, access_key=access_key)
        if data is None:
            print(f"Dataset '{dataset_name}' not found in Data Lake. Transformation not applied.")
            return None
        
        try:
            transformed_data = transformation_func(data)
            print(f"Transformation applied successfully to dataset '{dataset_name}'.")
            
            if update_in_datalake:
                self.store_data(
                    dataset_name, 
                    transformed_data, 
                    access_key=access_key, 
                    force=True
                )
                print(f"Dataset '{dataset_name}' updated in the Data Lake with transformed data.")
            
            return transformed_data
        except Exception as e:
            print(f"Error during transformation: {e}")
            return None

    def clean_data(self, dataset_name: str, cleaning_func: Callable, access_key: int = -1) -> None:
        """
        Apply a cleaning function to a dataset and update it in the Data Lake.

        Args:
            dataset_name (str): Name of the dataset.
            cleaning_func (Callable): Function to clean the dataset.
            access_key (int): Access key for secure access.

        Returns:
            None
        """
        data = self.retrieve_data(dataset_name, access_key=access_key)
        if data is None:
            print(f"Dataset '{dataset_name}' not found in Data Lake. Cleaning not applied.")
            return
        try:
            cleaned_data = cleaning_func(data)
            self.store_data(dataset_name, cleaned_data, access_key=access_key, force=True)
            print(f"Dataset '{dataset_name}' cleaned and updated successfully.")
        except Exception as e:
            print(f"Error during cleaning: {e}")
