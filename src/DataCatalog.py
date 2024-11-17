from src.DataLake import DataLake
from typing import Any, Dict, List, Optional
from src.DataLake import DataLake
from typing import Any, Dict, List, Optional

class DataCatalog(DataLake):
    def __init__(self, base_path: str = "data_lake") -> None:
        """
        Initialize the Data Catalog and Data Lake.
        
        Args:
            base_path (str): Path to store raw and processed data.
        """
        super().__init__(base_path)  # Initialize the DataLake functionality

        self._categories: Dict[str, Any] = {
            "Data Category 1": {"datasets": []},
            "Data Category 2": {"datasets": []},
            "Data Category 3": {"datasets": []},
            "Data Category 4": {"datasets": []},
            "Company": {
                "Company Financials": {
                    "datasets": [],
                    "Quarterly": {"datasets": []},
                    "Annual": {"datasets": []}
                },
                "Company Level Actuals": {"datasets": []},
                "Analyst Recommendations": {"datasets": []},
                "Company Guidance": {"datasets": []}
            },
            "ESG": {
                "Environmental": {
                    "Carbon Emissions": {"datasets": []},
                    "Climate Risk": {"datasets": []},
                    "datasets": []
                }
            },
            "Reference Data": {
                "Classifications": {
                    "Industry": {"datasets": []},
                    "Region": {"datasets": []},
                    "datasets": []
                },
                "Bloomberg Classifications": {"datasets": []}
            },
            "Equities/Funds": {
                "Derivatives": {
                    "Options": {"datasets": []},
                    "Futures": {"datasets": []},
                    "datasets": []
                },
                "Equity Futures": {"datasets": []},
                "Pricing & Analytics": {"datasets": []}
            },
            "Fixed Income": {
                "GSAC": {"datasets": []},
                "Pricing & Analytics": {
                    "Government": {"datasets": []},
                    "Corporate": {"datasets": []},
                    "datasets": []
                },
                "Pricing": {"datasets": []}
            },
            "Mortgages": {
                "Pricing & Analytics": {
                    "Residential": {"datasets": []},
                    "Commercial": {"datasets": []},
                    "datasets": []
                },
                "Analytics": {"datasets": []}
            },
            "Macro": {
                "Economic Data": {
                    "GDP": {"datasets": []},
                    "Inflation": {"datasets": []},
                    "datasets": []
                },
                "Country Headline Metrics": {"datasets": []},
                "Actuals - Periodic": {"datasets": []}
            }
        }

    @property
    def category_names(self) -> List[str]:
        """Get the list of main category names."""
        return list(self._categories.keys())
    
    @property
    def categories(self) -> Dict[str, Any]:
        """Get the complete catalog structure."""
        return self._categories

    def _get_category_path(self, path: List[str]) -> Optional[Dict[str, Any]]:
        """
        Navigate to a specific path in the category structure.
        
        Args:
            path (List[str]): List of category levels to navigate.
            
        Returns:
            Optional[Dict[str, Any]]: The category dict at the specified path.
        """
        current = self._categories
        for level in path:
            if level not in current:
                return None
            current = current[level]
        return current

    def list_subcategories(self, path: List[str]) -> List[str]:
        """
        List all subcategories at a given path.
        
        Args:
            path (List[str]): Path to the category level.
            
        Returns:
            List[str]: List of subcategory names (excluding 'datasets' key).
        """
        category = self._get_category_path(path)
        if category and isinstance(category, dict):
            return [k for k in category.keys() if k != "datasets"]
        return []

    @DataLake.access_decorator
    def add_dataset(
        self, 
        path: List[str],
        dataset_name: str, 
        access_key: int = -1,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a dataset to a specific category path.
        
        Args:
            path (List[str]): Path to the category level.
            dataset_name (str): Name of the dataset.
            access_key (int): Access key to verify permission.
            metadata (Optional[Dict]): Metadata associated with the dataset.
        """
        try:
            category = self._get_category_path(path)
            if not category:
                print(f"Invalid category path: {'/'.join(path)}")
                return

            # Validate dataset exists in DataLake
            if not self.get_metadata(dataset_name, access_key):
                print(f"Dataset '{dataset_name}' not found in DataLake.")
                return

            # Get metadata
            dataset_metadata = metadata if metadata else self.get_metadata(dataset_name, access_key)
            
            # Check if dataset already exists
            if not "datasets" in category:
                category["datasets"] = []
                
            if any(d.get("name") == dataset_name for d in category["datasets"]):
                print(f"Dataset '{dataset_name}' already exists at path: {'/'.join(path)}")
                return

            # Add dataset
            category["datasets"].append({
                "name": dataset_name,
                "metadata": dataset_metadata
            })
            print(f"Dataset '{dataset_name}' added to path: {'/'.join(path)}")

        except Exception as e:
            print(f"Failed to add dataset: {e}")

    def get_datasets(self, path: List[str]) -> List[Dict[str, Any]]:
        """
        Get all datasets at a specific category path.
        
        Args:
            path (List[str]): Path to the category level.
            
        Returns:
            List[Dict[str, Any]]: List of datasets at the specified path.
        """
        try:
            category = self._get_category_path(path)
            if category and isinstance(category, dict):
                return category.get("datasets", [])
            return []
        except Exception as e:
            print(f"Failed to get datasets: {e}")
            return []

    @DataLake.access_decorator
    def search_datasets(
        self, 
        keyword: str,
        access_key: int = -1
    ) -> List[Dict[str, Any]]:
        """
        Search for datasets by keyword across all categories.
        
        Args:
            keyword (str): Keyword to search for.
            access_key (int): Access key to verify permission.
            
        Returns:
            List[Dict[str, Any]]: List of matching datasets with their full path.
        """
        def search_recursive(category: Dict[str, Any], current_path: List[str]) -> List[Dict[str, Any]]:
            results = []
            
            # Search datasets at current level
            for dataset in category.get("datasets", []):
                if (keyword in dataset["name"].lower() or 
                    keyword in str(dataset["metadata"]).lower()):
                    results.append({
                        "name": dataset["name"],
                        "path": current_path.copy(),
                        "metadata": dataset["metadata"]
                    })
            
            # Search in subcategories
            for key, value in category.items():
                if key != "datasets" and isinstance(value, dict):
                    results.extend(search_recursive(value, current_path + [key]))
                    
            return results

        try:
            keyword = keyword.lower()
            results = []
            
            for category_name, category in self._categories.items():
                results.extend(search_recursive(category, [category_name]))
                
            return results
            
        except Exception as e:
            print(f"Search failed: {e}")
            return []

    def get_structure(self, path: List[str] = None) -> Dict[str, Any]:
        """
        Get the structure with dataset counts at a specific path or entire catalog.
        
        Args:
            path (List[str], optional): Path to get structure for. None for entire catalog.
            
        Returns:
            Dict[str, Any]: Structure with dataset counts.
        """
        def count_recursive(category: Dict[str, Any]) -> Dict[str, Any]:
            result = {"dataset_count": len(category.get("datasets", []))}
            
            for key, value in category.items():
                if key != "datasets" and isinstance(value, dict):
                    result[key] = count_recursive(value)
                    
            return result

        try:
            if path:
                category = self._get_category_path(path)
                if not category:
                    return {}
                return count_recursive(category)
            
            return {
                category_name: count_recursive(category)
                for category_name, category in self._categories.items()
            }
            
        except Exception as e:
            print(f"Failed to get structure: {e}")
            return {}
        
        