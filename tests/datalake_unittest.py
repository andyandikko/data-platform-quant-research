import unittest
import pandas as pd
import numpy as np
import os
import shutil
from unittest.mock import patch, MagicMock
from src.data_lake import DataLake

class TestDataLake(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_base_path = "test_data_lake"
        self.data_lake = DataLake(base_path=self.test_base_path)
        self.valid_access_key = 134  # Access key for "Andy"
        
        # Sample test data
        self.test_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        self.test_dict = {"key": "value"}
        self.test_list = [1, 2, 3]
        
    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.test_base_path):
            shutil.rmtree(self.test_base_path)
            
    def test_initialization(self):
        """Test DataLake initialization and directory creation."""
        self.assertTrue(os.path.exists(f"{self.test_base_path}/raw"))
        self.assertTrue(os.path.exists(f"{self.test_base_path}/processed"))
        
    def test_store_data_invalid_access(self):
        """Test storing data with invalid access key."""
        result = self.data_lake.store_data(
            "test_dataset",
            self.test_df,
            access_key=-1
        )
        self.assertIsNone(result)
        
    def test_store_retrieve_dataframe(self):
        """Test storing and retrieving a pandas DataFrame."""
        # Store data
        self.data_lake.store_data(
            "test_df",
            self.test_df,
            access_key=self.valid_access_key,
            force=True
        )
        
        # Retrieve data
        retrieved_df = self.data_lake.retrieve_data(
            "test_df",
            access_key=self.valid_access_key
        )
        
        pd.testing.assert_frame_equal(self.test_df, retrieved_df)
        
    def test_store_retrieve_dict(self):
        """Test storing and retrieving a dictionary."""
        self.data_lake.store_data(
            "test_dict",
            self.test_dict,
            access_key=self.valid_access_key,
            force=True
        )
        
        retrieved_dict = self.data_lake.retrieve_data(
            "test_dict",
            access_key=self.valid_access_key
        )
        
        self.assertEqual(self.test_dict, retrieved_dict)
        
    def test_metadata_storage(self):
        """Test metadata storage and retrieval."""
        custom_metadata = {"description": "Test dataset"}
        
        self.data_lake.store_data(
            "test_metadata",
            self.test_df,
            access_key=self.valid_access_key,
            metadata=custom_metadata,
            force=True
        )
        
        metadata = self.data_lake.get_metadata(
            "test_metadata",
            access_key=self.valid_access_key
        )
        
        self.assertIn("description", metadata)
        self.assertEqual(metadata["description"], "Test dataset")
        self.assertEqual(metadata["Author"], "Andy")
        self.assertIn("modification_time", metadata)
        
    def test_list_datasets(self):
        """Test listing datasets."""
        # Store multiple datasets
        datasets = ["dataset1", "dataset2"]
        for dataset in datasets:
            self.data_lake.store_data(
                dataset,
                self.test_df,
                access_key=self.valid_access_key,
                force=True
            )
            
        listed_datasets = self.data_lake.list_datasets(
            access_key=self.valid_access_key
        )
        
        for dataset in datasets:
            self.assertIn(dataset, listed_datasets)
            
    def test_delete_dataset(self):
        """Test dataset deletion."""
        # Store a dataset
        self.data_lake.store_data(
            "test_delete",
            self.test_df,
            access_key=self.valid_access_key,
            force=True
        )
        
        # Delete the dataset
        self.data_lake.delete_dataset(
            "test_delete",
            access_key=self.valid_access_key
        )
        
        # Verify deletion
        self.assertNotIn(
            "test_delete",
            self.data_lake.list_datasets(access_key=self.valid_access_key)
        )
        
    def test_execute_sql(self):
        """Test SQL query execution with valid and invalid queries."""
        # Store test DataFrame
        self.data_lake.store_data(
            "test_sql",
            self.test_df,
            access_key=self.valid_access_key,
            force=True
        )

        # Ensure dataset is correctly loaded in the namespace
        namespace = {
            "test_sql": self.data_lake.retrieve_data("test_sql", access_key=self.valid_access_key)
        }
        self.assertIn("test_sql", namespace)
        self.assertIsInstance(namespace["test_sql"], pd.DataFrame)

        # Valid SQL query
        valid_query = "SELECT * FROM test_sql WHERE A > 1"
        result = self.data_lake.execute_sql(valid_query, access_key=self.valid_access_key)

        # Assert result is a DataFrame with correct content
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # Should return 2 rows where A > 1
        self.assertTrue((result['A'] > 1).all())

        # Invalid SQL query
        invalid_query = "SELECT non_existent_column FROM test_sql"
        result = self.data_lake.execute_sql(invalid_query, access_key=self.valid_access_key)

        # Assert None is returned for invalid query
        self.assertIsNone(result, "Invalid query should return None")


    def test_execute_sql_empty_namespace(self):
        """Test SQL query execution when no DataFrames are stored."""
        # Execute SQL query without any DataFrame datasets
        query = "SELECT * FROM non_existent_table"
        result = self.data_lake.execute_sql(query, access_key=self.valid_access_key)

        # Assert None is returned
        self.assertIsNone(result, "No DataFrames should result in None for SQL queries")
        
    def test_execute_sql_multiple_datasets(self):
        """Test SQL query execution across multiple datasets."""
        # Store multiple DataFrames
        df1 = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        df2 = pd.DataFrame({"id": [3, 4], "value": [30, 40]})

        self.data_lake.store_data("dataset1", df1, access_key=self.valid_access_key, force=True)
        self.data_lake.store_data("dataset2", df2, access_key=self.valid_access_key, force=True)

        # Execute SQL query
        query = """
        SELECT * FROM dataset1
        UNION ALL
        SELECT * FROM dataset2
        """
        result = self.data_lake.execute_sql(query, access_key=self.valid_access_key)

        # Assert the result is a DataFrame with combined content
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)  # 2 rows from each dataset
        self.assertListEqual(result['id'].tolist(), [1, 2, 3, 4])
        self.assertListEqual(result['value'].tolist(), [10, 20, 30, 40])


        
    def test_processed_vs_raw_storage(self):
        """Test storing and retrieving processed vs raw data."""
        # Store raw data
        self.data_lake.store_data(
            "test_data",
            self.test_df,
            access_key=self.valid_access_key,
            processed=False,
            force=True
        )
        
        # Store processed version
        processed_df = self.test_df * 2
        self.data_lake.store_data(
            "test_data",
            processed_df,
            access_key=self.valid_access_key,
            processed=True,
            force=True
        )
        
        # Retrieve both versions
        raw_data = self.data_lake.retrieve_data(
            "test_data",
            access_key=self.valid_access_key,
            processed=False
        )
        processed_data = self.data_lake.retrieve_data(
            "test_data",
            access_key=self.valid_access_key,
            processed=True
        )
        
        pd.testing.assert_frame_equal(raw_data, self.test_df)
        pd.testing.assert_frame_equal(processed_data, processed_df)
        
    @patch('builtins.input', return_value='no')
    def test_overwrite_protection(self, mock_input):
        """Test dataset overwrite protection."""
        # Store initial data
        self.data_lake.store_data(
            "test_overwrite",
            self.test_df,
            access_key=self.valid_access_key,
            force=True
        )
        
        # Attempt to overwrite without force
        new_df = pd.DataFrame({'A': [4, 5, 6]})
        self.data_lake.store_data(
            "test_overwrite",
            new_df,
            access_key=self.valid_access_key
        )
        
        # Verify original data remains
        retrieved_df = self.data_lake.retrieve_data(
            "test_overwrite",
            access_key=self.valid_access_key
        )
        pd.testing.assert_frame_equal(retrieved_df, self.test_df)
        
    def test_execute_sql_with_excel(self):
        """Test SQL query execution on data loaded from an Excel file."""
        # Load the Excel file
        file_path = "./data/spy_data.xlsx"
        sheet_name = "total returns"
        spy_data = pd.read_excel(file_path, sheet_name=sheet_name, index_col="date")

        # Store the data in the Data Lake
        self.data_lake.store_data(
            "spy_data",
            spy_data,
            access_key=self.valid_access_key,
            force=True
        )

        # Ensure the data is correctly stored
        retrieved_data = self.data_lake.retrieve_data("spy_data", access_key=self.valid_access_key)
        pd.testing.assert_frame_equal(spy_data, retrieved_data)

        # Perform an SQL query: Select rows where SPY > 0
        sql_query = "SELECT * FROM spy_data WHERE SPY > 0"
        result = self.data_lake.execute_sql(sql_query, access_key=self.valid_access_key)

        # Assert the result is a DataFrame and contains the expected rows
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)  # Ensure there are rows in the result
        self.assertTrue((result["SPY"] > 0).all())  # Validate the query logic
        print(result)
        print(result["SPY"].head())


if __name__ == '__main__':
    unittest.main()