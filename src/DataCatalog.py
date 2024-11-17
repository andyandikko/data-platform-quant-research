from typing import Any, Dict, List, Optional
import json
from src.DataLake import DataLake
import os


from functools import wraps

class DataCatalog(DataLake):
    CATALOG_FILE = "categories.json"
    secured_access = {134: "Andy", 245: "Matt", 367: "Harry"}

    @staticmethod
    def catalog_access_decorator(func):
        """Decorator for catalog access control."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            access_key = kwargs.get("access_key", -1)
            if access_key not in DataCatalog.secured_access:
                print("Catalog Access Denied: Invalid access key.")
                return None
            return func(self, *args, **kwargs)
        return wrapper

    def __init__(self, base_path: str = "data_lake") -> None:
        super().__init__(base_path)
        self.categories: Dict[str, List[Dict[str, Any]]] = self._load_categories()

    def _load_categories(self) -> Dict[str, List[Dict[str, Any]]]:
        if os.path.exists(DataCatalog.CATALOG_FILE):
            try:
                with open(DataCatalog.CATALOG_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load categories: {e}")
        return {}

    def _save_categories(self) -> bool:
        try:
            with open(DataCatalog.CATALOG_FILE, "w") as f:
                json.dump(self.categories, f, indent=4)
                print("Categories saved successfully.")
            return True
        except Exception as e:
            print(f"Failed to save categories: {e}")
            return False

    @catalog_access_decorator
    def add_category(self, name: str, access_key: int = -1) -> bool:
        """
        Add a new category to the catalog.
        
        Args:
            name (str): Name of the category.
            access_key (int): Catalog access key to verify permission.
            
        Returns:
            bool: Success status of the operation.
        """
        try:
            if name not in self.categories:
                self.categories[name] = []
                if self._save_categories():
                    print(f"Category '{name}' added to the catalog.")
                    return True
            else:
                print(f"Category '{name}' already exists.")
            return False
        except Exception as e:
            print(f"Failed to add category: {e}")
            return False

    @catalog_access_decorator
    def add_dataset_to_category(
        self,
        category_name: str,
        dataset_name: str,
        access_key: int = -1,
        metadata: Optional[Dict] = None,
        datalake_key: Optional[int] = None
    ) -> bool:
        """
        Add a dataset to a specific category.
        
        Args:
            category_name (str): The category to add the dataset to.
            dataset_name (str): The name of the dataset.
            access_key (int): Catalog access key to verify permission.
            metadata (Optional[Dict]): Metadata associated with the dataset.
            datalake_key (Optional[int]): Separate key for DataLake access if needed.
            
        Returns:
            bool: Success status of the operation.
        """
        try:
            if category_name not in self.categories:
                print(f"Category '{category_name}' does not exist. Add the category first.")
                return False

            # Use separate DataLake key if provided
            dl_key = datalake_key if datalake_key is not None else access_key
            existing_metadata = super().get_metadata(dataset_name, access_key=dl_key)
            if not existing_metadata:
                print(f"Dataset '{dataset_name}' not found in DataLake.")
                return False

            dataset_metadata = metadata if metadata else existing_metadata

            if any(dataset["name"] == dataset_name for dataset in self.categories[category_name]):
                print(f"Dataset '{dataset_name}' already exists in category '{category_name}'.")
                return False

            self.categories[category_name].append({
                "name": dataset_name,
                "metadata": dataset_metadata,
                "catalog_owner": DataCatalog.secured_access[access_key]
            })
            
            if self._save_categories():
                print(f"Dataset '{dataset_name}' added to category '{category_name}'.")
                return True
            return False

        except Exception as e:
            print(f"Failed to add dataset to category: {e}")
            return False

    @catalog_access_decorator
    def list_datasets_in_category(
        self,
        category_name: str,
        access_key: int = -1
    ) -> List[str]:
        """
        List all datasets in a given category.
        
        Args:
            category_name (str): The category to list datasets from.
            access_key (int): Catalog access key to verify permission.
            
        Returns:
            List[str]: List of dataset names.
        """
        try:
            if category_name in self.categories:
                return [dataset["name"] for dataset in self.categories[category_name]]
            print(f"Category '{category_name}' not found.")
            return []
        except Exception as e:
            print(f"Failed to list datasets: {e}")
            return []

    @catalog_access_decorator
    def search_datasets(
        self,
        keyword: str,
        access_key: int = -1
    ) -> List[Dict[str, Any]]:
        """
        Search for datasets by keyword across all categories.
        
        Args:
            keyword (str): Keyword to search for.
            access_key (int): Catalog access key to verify permission.
            
        Returns:
            List[Dict[str, Any]]: List of matching datasets with their categories.
        """
        try:
            results = []
            keyword = keyword.lower()
            
            for category, datasets in self.categories.items():
                for dataset in datasets:
                    if keyword in dataset["name"].lower() or keyword in str(dataset["metadata"]).lower():
                        results.append({
                            "name": dataset["name"],
                            "category": category,
                            "metadata": dataset["metadata"],
                            "catalog_owner": dataset.get("catalog_owner", "Unknown")
                        })
            return results
            
        except Exception as e:
            print(f"Search failed: {e}")
            return []

    @catalog_access_decorator
    def remove_from_category(
        self,
        category_name: str,
        dataset_name: str,
        access_key: int = -1
    ) -> bool:
        """
        Remove a dataset from a category.
        
        Args:
            category_name (str): The category to remove the dataset from.
            dataset_name (str): The name of the dataset to remove.
            access_key (int): Catalog access key to verify permission.
            
        Returns:
            bool: Success status of the operation.
        """
        try:
            if category_name not in self.categories:
                print(f"Category '{category_name}' not found.")
                return False

            # Check if user has permission to remove dataset
            for dataset in self.categories[category_name]:
                if (dataset["name"] == dataset_name and 
                    dataset.get("catalog_owner") == DataCatalog.secured_access[access_key]):
                    self.categories[category_name].remove(dataset)
                    if self._save_categories():
                        print(f"Dataset '{dataset_name}' removed from category '{category_name}'.")
                        return True

            print(f"Dataset '{dataset_name}' not found or you don't have permission to remove it.")
            return False

        except Exception as e:
            print(f"Failed to remove dataset from category: {e}")
            return False