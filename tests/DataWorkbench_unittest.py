import unittest
import pandas as pd
from typing import Any
from src.DataWorkbench import DataWorkbench

class TestDataWorkbench(unittest.TestCase):
    def setUp(self):
        """
        Setup the test environment by initializing a DataWorkbench instance
        and preparing test datasets.
        """
        self.workbench = DataWorkbench()
        self.test_data = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [100, 200, 300]
        })
        self.transformation_func = lambda df: df.assign(value=df['value'] * 2)
        self.cleaning_func = lambda df: df[df['value'] > 150]

    def test_store_data_in_memory(self):
        """
        Test storing a dataset in memory.
        """
        dataset_name = "test_memory_dataset"
        self.workbench.store_data_in_memory(dataset_name, self.test_data)
        self.assertIn(dataset_name, self.workbench.data_storage)
        self.assertTrue(self.test_data.equals(self.workbench.data_storage[dataset_name]))

    def test_retrieve_data_in_memory(self):
        """
        Test retrieving a dataset from memory.
        """
        dataset_name = "test_memory_dataset"
        self.workbench.store_data_in_memory(dataset_name, self.test_data)
        retrieved_data = self.workbench.retrieve_data_in_memory(dataset_name)
        self.assertIsNotNone(retrieved_data)
        self.assertTrue(self.test_data.equals(retrieved_data))

    def test_transform_data(self):
        """
        Test applying a transformation function to a dataset.
        """
        dataset_name = "test_transform_dataset"
        self.workbench.store_data(dataset_name, self.test_data, access_key=134)
        transformed_data = self.workbench.transform_data(
            dataset_name, 
            self.transformation_func, 
            access_key=134, 
            update_in_datalake=False
        )
        self.assertIsNotNone(transformed_data)
        self.assertEqual(
            list(transformed_data['value']), 
            [200, 400, 600]
        )

    def test_clean_data(self):
        """
        Test applying a cleaning function to a dataset.
        """
        dataset_name = "test_clean_dataset"
        self.workbench.store_data(dataset_name, self.test_data, access_key=134)
        self.workbench.clean_data(dataset_name, self.cleaning_func, access_key=134)
        cleaned_data = self.workbench.retrieve_data(dataset_name, access_key=134)
        self.assertEqual(len(cleaned_data), 2)
        self.assertTrue((cleaned_data['value'] > 150).all())

    def test_retrieve_nonexistent_memory_data(self):
        """
        Test retrieving a non-existent dataset from memory.
        """
        dataset_name = "nonexistent_dataset"
        retrieved_data = self.workbench.retrieve_data_in_memory(dataset_name)
        self.assertIsNone(retrieved_data)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
