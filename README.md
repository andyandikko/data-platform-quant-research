# Data Platform for Quantitative Research

This project is a collaborative effort to design and implement a data platform that simplifies the workflow for quantitative researchers. It includes:

- **Data Lake**: A scalable storage solution for raw and processed data.
- **Data Catalog**: Metadata-rich inventory for dataset discovery.
- **Data Workbench**: Tools for data transformation and preparation.
- **Quant Data Models**: Standardized models for structured data access.

## Components
- `src/`: Contains Python implementations of each platform component.
- `docs/`: Documentation and user guides.
- `tests/`: Unit tests for platform functionalities.

## How to Collaborate
1. Clone the repository:
   ```bash
   git clone https://github.com/andyandikko/data-platform-quant-research.git


# Data Lake System

## Overview
The **Data Lake System** is a Python-based platform for securely storing, retrieving, and managing datasets. It supports structured (e.g., DataFrames), semi-structured (e.g., dictionaries, lists), and unstructured data (e.g., strings, binary files) using the pickle format for serialization. Key features include metadata management, SQL querying, and secure access control through user-specific access keys.

---

## Features
- **Secure Access Control**: Users must provide valid access keys for operations.
- **Data Storage**: Store any Python object in the Data Lake using pickle.
- **Data Retrieval**: Retrieve stored datasets for analysis or reuse.
- **SQL Query Execution**: Execute SQL queries on structured datasets stored in the Data Lake.
- **Dataset Listing**: List all datasets, with optional filtering by processing status.
- **Metadata Management**: Automatically collect and retrieve metadata for datasets.
- **Dataset Deletion**: Securely delete datasets and associated metadata.

---

## Setup
1. Clone the repository or copy the `DataLake` class into your Python project.
2. Install required dependencies:
   - `pandas`
   - `numpy`
   - `pandasql`
   - `pickle` (built-in)
3. Usage of Data Lake:
   
   ```python
   datalake = DataLake(base_path="data_lake")
   # Create a sample DataFrame
   data = pd.DataFrame({
       "timestamp": ["2024-01-01T10:00:00", "2024-01-01T10:01:00", "2024-01-01T10:02:00"],
       "price": [100, 101, 99],
       "volume": [200, 250, 180],
   })
   
   datalake.store_data(
       dataset_name="example_dataset",
       data=data,
       access_key=134,
       processed=False,
       metadata={"description": "Sample dataset for testing"}
   )
   # Retrieve the stored dataset
   retrieved_data = datalake.retrieve_data(
    dataset_name="example_dataset",
    access_key=134,
    processed=False
   )
   print(retrieved_data)
   # Execute an SQL query on the dataset
   sql_query = "SELECT * FROM example_dataset WHERE price > 100"
   query_result = datalake.execute_sql(sql_query, access_key=134)
   print(query_result)
   # List all raw datasets
   datasets = datalake.list_datasets(access_key=134, processed=False)
   print("Available raw datasets:", datasets)
   # Retrieve metadata for a specific dataset
   metadata = datalake.get_metadata(dataset_name="example_dataset", access_key=134)
   print("Metadata for 'example_dataset':", metadata)
   # Delete a dataset
   datalake.delete_dataset(dataset_name="example_dataset", access_key=134, processed=False)

   ```
