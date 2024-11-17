import unittest
import pandas as pd
from unittest.mock import patch
from src.DataCatalog import DataCatalog
import os
import json
from src.DataLake import DataLake

class TestDataCatalog(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.base_path = "test_data_lake"
        self.catalog = DataCatalog(base_path=self.base_path)
        self.catalog_key = 134  # Catalog access key for "Andy"
        self.datalake_key = 134  # DataLake access key for "Andy"

        # Load SPY data
        self.file_path = "./data/spy_data.xlsx"
        self.sheet_name = "total returns"
        try:
            self.spy_data = pd.read_excel(
                self.file_path, 
                sheet_name=self.sheet_name, 
                index_col="date"
            )
        except FileNotFoundError:
            raise unittest.SkipTest(
                f"SPY data file not found: {self.file_path}. "
                "Please ensure spy_data.xlsx exists in the data directory."
            )

    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.base_path):
            import shutil
            shutil.rmtree(self.base_path)
        if os.path.exists(DataCatalog.CATALOG_FILE):
            os.remove(DataCatalog.CATALOG_FILE)
        if os.path.exists("metadata.json"):
            os.remove("metadata.json")

    def test_spy_data_organization(self):
        """Test organizing SPY data in appropriate categories."""
        # Store SPY data in DataLake
        super(DataCatalog, self.catalog).store_data(
            dataset_name="spy_data",
            data=self.spy_data,
            access_key=self.datalake_key,
            metadata={"description": "SPY total returns data"},
            force=True
        )

        # Create category structure
        categories = [
            "Equities/Funds",
            "Market Data",
            "Historical Data"
        ]
        
        for category in categories:
            self.catalog.add_category(
                name=category,
                access_key=self.catalog_key
            )

        # Add SPY data to multiple categories
        results = []
        for category in categories:
            result = self.catalog.add_dataset_to_category(
                category_name=category,
                dataset_name="spy_data",
                access_key=self.catalog_key,
                datalake_key=self.datalake_key,
                metadata={"category_specific": f"SPY data in {category}"}
            )
            results.append(result)

        # Verify additions
        self.assertTrue(all(results), "Failed to add SPY data to all categories")

        # Verify data accessibility in each category
        for category in categories:
            datasets = self.catalog.list_datasets_in_category(
                category_name=category,
                access_key=self.catalog_key
            )
            self.assertIn("spy_data", datasets)

    def test_spy_data_search(self):
        """Test searching for SPY-related data."""
        # Store data
        super(DataCatalog, self.catalog).store_data(
            dataset_name="spy_data",
            data=self.spy_data,
            access_key=self.datalake_key,
            force=True
        )

        # Add to categories with different metadata
        self.catalog.add_category(
            name="Equities/Funds",
            access_key=self.catalog_key
        )
        
        self.catalog.add_dataset_to_category(
            category_name="Equities/Funds",
            dataset_name="spy_data",
            access_key=self.catalog_key,
            datalake_key=self.datalake_key,
            metadata={
                "instrument": "SPY",
                "type": "ETF",
                "market": "US Equities"
            }
        )

        # Test various search terms
        search_terms = ["spy", "etf", "equities", "us"]
        for term in search_terms:
            results = self.catalog.search_datasets(
                keyword=term,
                access_key=self.catalog_key
            )
            self.assertGreater(
                len(results), 0, 
                f"No results found for term: {term}"
            )

    def test_spy_data_metadata_handling(self):
        """Test handling of SPY data metadata."""
        # Store with detailed metadata
        metadata = {
            "instrument": "SPY",
            "type": "ETF",
            "market": "US Equities",
            "description": "S&P 500 ETF Trust",
            "provider": "State Street Global Advisors",
            "frequency": "Daily",
            "start_date": str(self.spy_data.index.min()),
            "end_date": str(self.spy_data.index.max()),
            "columns": list(self.spy_data.columns),
            "row_count": len(self.spy_data)
        }

        super(DataCatalog, self.catalog).store_data(
            dataset_name="spy_data",
            data=self.spy_data,
            access_key=self.datalake_key,
            metadata=metadata,
            force=True
        )

        self.catalog.add_category(
            name="Equities/Funds",
            access_key=self.catalog_key
        )

        # Add with metadata
        self.catalog.add_dataset_to_category(
            category_name="Equities/Funds",
            dataset_name="spy_data",
            access_key=self.catalog_key,
            datalake_key=self.datalake_key,
            metadata=metadata
        )

        # Search and verify metadata
        results = self.catalog.search_datasets(
            keyword="spy",
            access_key=self.catalog_key
        )
        
        self.assertEqual(len(results), 1)
        result_metadata = results[0]["metadata"]
        for key, value in metadata.items():
            self.assertEqual(
                result_metadata[key],
                value,
                f"Metadata mismatch for {key}"
            )

    def test_spy_data_versioning(self):
        """Test handling different versions of SPY data."""
        # Create different versions of data
        spy_daily = self.spy_data.copy()
        spy_weekly = self.spy_data.resample('W').last()
        spy_monthly = self.spy_data.resample('M').last()

        versions = {
            "spy_daily": spy_daily,
            "spy_weekly": spy_weekly,
            "spy_monthly": spy_monthly
        }

        # Store all versions
        for name, data in versions.items():
            super(DataCatalog, self.catalog).store_data(
                dataset_name=name,
                data=data,
                access_key=self.datalake_key,
                metadata={"frequency": name.split('_')[1]},
                force=True
            )

        # Add category
        self.catalog.add_category(
            name="SPY Versions",
            access_key=self.catalog_key
        )

        # Add all versions to catalog
        for name in versions:
            result = self.catalog.add_dataset_to_category(
                category_name="SPY Versions",
                dataset_name=name,
                access_key=self.catalog_key,
                datalake_key=self.datalake_key
            )
            self.assertTrue(result)

        # Verify all versions are searchable
        results = self.catalog.search_datasets(
            keyword="spy",
            access_key=self.catalog_key
        )
        self.assertEqual(len(results), 3)

    def test_spy_data_access_patterns(self):
        """Test different access patterns for SPY data."""
        # Store base data
        super(DataCatalog, self.catalog).store_data(
            dataset_name="spy_data",
            data=self.spy_data,
            access_key=self.datalake_key,
            force=True
        )

        # Set up categories
        self.catalog.add_category(
            name="Public Data",
            access_key=self.catalog_key
        )
        self.catalog.add_category(
            name="Restricted Data",
            access_key=self.catalog_key
        )

        # Test different access patterns
        access_keys = [134, 245, 367, -1]  # Different users and invalid key
        categories = ["Public Data", "Restricted Data"]

        for key in access_keys:
            for category in categories:
                # Try to add dataset
                result = self.catalog.add_dataset_to_category(
                    category_name=category,
                    dataset_name="spy_data",
                    access_key=key,
                    datalake_key=self.datalake_key
                )
                
                # Verify expected behavior
                if key in DataCatalog.secured_access:
                    self.assertIsNotNone(result)
                else:
                    self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()


# import unittest
# import pandas as pd
# from unittest.mock import patch
# from src.DataCatalog import DataCatalog
# import os
# import json
# from src.DataLake import DataLake

# class TestDataCatalog(unittest.TestCase):
#     def setUp(self):
#         """Set up test fixtures before each test method."""
#         self.base_path = "test_data_lake"
#         self.catalog = DataCatalog(base_path=self.base_path)
#         self.catalog_key = 134  # Catalog access key for "Andy"
#         self.datalake_key = 134  # DataLake access key for "Andy"
        
#         # Create simple test data
#         self.test_df = pd.DataFrame({
#             'A': [1, 2, 3],
#             'B': ['x', 'y', 'z']
#         })

#     def tearDown(self):
#         """Clean up after each test method."""
#         if os.path.exists(self.base_path):
#             import shutil
#             shutil.rmtree(self.base_path)
#         if os.path.exists(DataCatalog.CATALOG_FILE):
#             os.remove(DataCatalog.CATALOG_FILE)
#         if os.path.exists("metadata.json"):
#             os.remove("metadata.json")

#     def test_category_management(self):
#         """Test category addition and verification."""
#         # Test invalid access
#         invalid_result = self.catalog.add_category(
#             name="InvalidCategory",
#             access_key=-1
#         )
#         self.assertIsNone(invalid_result)
#         self.assertNotIn("InvalidCategory", self.catalog.categories)

#         # Test valid access
#         valid_result = self.catalog.add_category(
#             name="TestCategory",
#             access_key=self.catalog_key
#         )
#         self.assertTrue(valid_result)
#         self.assertIn("TestCategory", self.catalog.categories)
#         self.assertEqual(self.catalog.categories["TestCategory"], [])

#     def test_dataset_management(self):
#         """Test dataset addition and retrieval."""
#         # First store in DataLake
#         super(DataCatalog, self.catalog).store_data(
#             dataset_name="test_data",
#             data=self.test_df,
#             access_key=self.datalake_key,
#             force=True
#         )

#         # Add category
#         self.catalog.add_category(
#             name="TestCategory",
#             access_key=self.catalog_key
#         )

#         # Test adding dataset with catalog access
#         result = self.catalog.add_dataset_to_category(
#             category_name="TestCategory",
#             dataset_name="test_data",
#             access_key=self.catalog_key,
#             datalake_key=self.datalake_key
#         )
#         self.assertTrue(result)

#         # List datasets
#         datasets = self.catalog.list_datasets_in_category(
#             category_name="TestCategory",
#             access_key=self.catalog_key
#         )
#         self.assertIsNotNone(datasets)
#         self.assertIn("test_data", datasets)

#     def test_access_control_separation(self):
#         """Test separation of catalog and DataLake access control."""
#         # Store in DataLake with DataLake access
#         super(DataCatalog, self.catalog).store_data(
#             dataset_name="test_data",
#             data=self.test_df,
#             access_key=self.datalake_key,
#             force=True
#         )

#         # Add category with catalog access
#         self.catalog.add_category(
#             name="TestCategory",
#             access_key=self.catalog_key
#         )

#         # Test different access key combinations
#         # Case 1: Valid catalog key, invalid DataLake key
#         result1 = self.catalog.add_dataset_to_category(
#             category_name="TestCategory",
#             dataset_name="test_data",
#             access_key=self.catalog_key,
#             datalake_key=-1
#         )
#         self.assertFalse(result1)

#         # Case 2: Invalid catalog key, valid DataLake key
#         result2 = self.catalog.add_dataset_to_category(
#             category_name="TestCategory",
#             dataset_name="test_data",
#             access_key=-1,
#             datalake_key=self.datalake_key
#         )
#         self.assertIsNone(result2)

#         # Case 3: Both valid keys
#         result3 = self.catalog.add_dataset_to_category(
#             category_name="TestCategory",
#             dataset_name="test_data",
#             access_key=self.catalog_key,
#             datalake_key=self.datalake_key
#         )
#         self.assertTrue(result3)

#     def test_owner_based_operations(self):
#         """Test owner-based permissions."""
#         # Set up initial data
#         super(DataCatalog, self.catalog).store_data(
#             dataset_name="test_data",
#             data=self.test_df,
#             access_key=self.datalake_key,
#             force=True
#         )

#         self.catalog.add_category(
#             name="TestCategory",
#             access_key=self.catalog_key
#         )

#         # Add dataset as Andy
#         self.catalog.add_dataset_to_category(
#             category_name="TestCategory",
#             dataset_name="test_data",
#             access_key=self.catalog_key,
#             datalake_key=self.datalake_key
#         )

#         # Try to remove dataset as Matt (should fail)
#         matt_key = 245
#         remove_result = self.catalog.remove_from_category(
#             category_name="TestCategory",
#             dataset_name="test_data",
#             access_key=matt_key
#         )
#         self.assertFalse(remove_result)

#         # Remove dataset as Andy (should succeed)
#         remove_result = self.catalog.remove_from_category(
#             category_name="TestCategory",
#             dataset_name="test_data",
#             access_key=self.catalog_key
#         )
#         self.assertTrue(remove_result)

#     def test_search_with_owner_info(self):
#         """Test search results include owner information."""
#         # Set up test data
#         super(DataCatalog, self.catalog).store_data(
#             dataset_name="test_data",
#             data=self.test_df,
#             access_key=self.datalake_key,
#             force=True
#         )

#         self.catalog.add_category(
#             name="TestCategory",
#             access_key=self.catalog_key
#         )

#         self.catalog.add_dataset_to_category(
#             category_name="TestCategory",
#             dataset_name="test_data",
#             access_key=self.catalog_key,
#             datalake_key=self.datalake_key
#         )

#         # Search and verify owner info
#         results = self.catalog.search_datasets(
#             keyword="test",
#             access_key=self.catalog_key
#         )
#         self.assertEqual(len(results), 1)
#         self.assertEqual(results[0]["catalog_owner"], "Andy")

#     def test_data_persistence(self):
#         """Test data persistence with ownership."""
#         # Set up initial data
#         super(DataCatalog, self.catalog).store_data(
#             dataset_name="test_data",
#             data=self.test_df,
#             access_key=self.datalake_key,
#             force=True
#         )

#         self.catalog.add_category(
#             name="TestCategory",
#             access_key=self.catalog_key
#         )

#         self.catalog.add_dataset_to_category(
#             category_name="TestCategory",
#             dataset_name="test_data",
#             access_key=self.catalog_key,
#             datalake_key=self.datalake_key
#         )

#         # Create new catalog instance
#         new_catalog = DataCatalog(base_path=self.base_path)
        
#         # Verify data and ownership persistence
#         results = new_catalog.search_datasets(
#             keyword="test",
#             access_key=self.catalog_key
#         )
#         self.assertEqual(len(results), 1)
#         self.assertEqual(results[0]["catalog_owner"], "Andy")

# if __name__ == "__main__":
#     unittest.main()
    
    