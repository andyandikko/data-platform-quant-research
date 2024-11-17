import unittest
import pandas as pd
from typing import Any
from src.DataWorkbench import DataWorkbench
import datetime
from datetime import timedelta

class TestDataWorkbench(unittest.TestCase):

    def setUp(self):
        """Set up a DataWorkbench instance for testing."""
        self.workbench = DataWorkbench(base_path="test_data_lake")
        self.dataset_name = "test_dataset"
        self.data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.metadata = {"source": "test", "date": "2024-01-01"}

    def test_register_transformation(self):
        """Test registering a transformation."""
        def dummy_transformation(data):
            return data

        self.workbench.register_transformation("dummy", dummy_transformation, description="Dummy transformation")
        self.assertIn("dummy", self.workbench.transformations, "Transformation was not registered successfully.")
        self.assertEqual(self.workbench.transformations["dummy"]["description"], "Dummy transformation", 
                         "Transformation description mismatch.")

    def test_store_data_failure(self):
        """Test storing data in memory with invalid input."""
        invalid_data = None  # Invalid data
        result = self.workbench.store_data_in_memory(self.dataset_name, invalid_data, self.metadata)
        self.assertFalse(result, "Invalid data should not be stored in memory.")

    def test_transform_with_unknown_transformation(self):
        """Test applying an unknown transformation."""
        self.workbench.store_data_in_memory(self.dataset_name, self.data)
        transformed_data = self.workbench.transform_data(self.dataset_name, "unknown_transformation")
        self.assertIsNone(transformed_data, "Applying an unknown transformation should return None.")

    def test_store_and_retrieve_large_dataset(self):
        """Test storing and retrieving a large dataset."""
        large_data = pd.DataFrame({"A": range(1000000), "B": range(1000000)})
        result = self.workbench.store_data_in_memory("large_dataset", large_data)
        self.assertTrue(result, "Failed to store large dataset in memory.")

        retrieved_data = self.workbench.retrieve_data_in_memory("large_dataset")
        pd.testing.assert_frame_equal(retrieved_data, large_data, "Retrieved data does not match the stored large dataset.")

    def test_chain_transformations_with_failure(self):
        """Test chaining transformations where one fails."""
        def add_column(data):
            data["C"] = data["A"] + data["B"]
            return data

        def fail_transformation(data):
            raise ValueError("Transformation failed.")

        self.workbench.register_transformation("add_column", add_column)
        self.workbench.register_transformation("fail_transformation", fail_transformation)
        self.workbench.store_data_in_memory(self.dataset_name, self.data)

        chained_data = self.workbench.chain_transformations(
            self.dataset_name, ["add_column", "fail_transformation"]
        )
        self.assertIsNone(chained_data, "Chaining should fail when one transformation raises an error.")

    def test_clean_memory_storage_without_timestamp(self):
        """Test cleaning all memory storage without providing a timestamp."""
        self.workbench.store_data_in_memory(self.dataset_name, self.data, self.metadata)
        self.workbench.clean_memory_storage()
        
        retrieved_data = self.workbench.retrieve_data_in_memory(self.dataset_name)
        self.assertIsNone(retrieved_data, "Memory storage was not cleared completely.")

    def test_transform_with_custom_metadata(self):
        """Test applying a transformation with custom metadata."""
        def add_column(data):
            data["C"] = data["A"] + data["B"]
            return data

        custom_metadata = {"transformed_by": "add_column"}
        self.workbench.register_transformation("add_column", add_column)
        self.workbench.store_data_in_memory(self.dataset_name, self.data)

        transformed_data = self.workbench.transform_data(
            self.dataset_name, "add_column", metadata=custom_metadata, store_in_memory=True
        )
        self.assertEqual(self.workbench.data_storage[f"{self.dataset_name}_transformed"]["metadata"], 
                         custom_metadata, "Custom metadata was not stored correctly.")

    def test_get_empty_processing_history(self):
        """Test retrieving processing history for a dataset with no transformations."""
        history = self.workbench.get_processing_history(self.dataset_name)
        self.assertEqual(len(history), 0, "Processing history should be empty for a dataset with no transformations.")

    def test_transform_with_invalid_function(self):
        """Test applying a transformation with an invalid function."""
        self.workbench.store_data_in_memory(self.dataset_name, self.data)
        invalid_function = 123  # Invalid transformation function

        self.assertIsNone(self.workbench.transform_data(self.dataset_name, invalid_function))

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

    