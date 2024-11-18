# Data Platform for Quantitative Research

This project goal is to design and implement a data platform that simplifies the workflow for quantitative researchers. It includes:

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


# **Data Lake System**

## **Overview**
The Data Lake System is a Python-based platform for securely storing, retrieving, and managing datasets. It supports:
- **Structured Data**: Pandas DataFrames.
- **Semi-Structured Data**: Dictionaries, lists.
- **Unstructured Data**: Strings, binary files.

The system uses the **pickle** format for serialization and incorporates secure access controls via user-specific access keys. It includes metadata management, SQL querying for structured datasets, and dataset organization by processing status.

---

## **Features**
1. **Secure Access Control**: Operations require a valid access key for authentication.
2. **Data Storage**: Store any Python object in the Data Lake using pickle.
3. **Data Retrieval**: Access stored datasets for analysis or reuse.
4. **SQL Query Execution**: Execute SQL queries on structured datasets.
5. **Dataset Listing**: List available datasets with optional filtering by processing status.
6. **Metadata Management**: Automatically collects and allows retrieval of dataset metadata.
7. **Dataset Deletion**: Securely delete datasets and their metadata.

---

## **Setup**

### **Clone the Repository**
Clone this repository or copy the `DataLake` class into your project.

### **Install Dependencies**
Run the following command to install required dependencies:
```bash
pip install pandas numpy pandasql
```



## Initialize the Data Lake
```python
from datalake import DataLake
datalake = DataLake(base_path="data_lake")
```

## Step 2: Storing Data
Store data (e.g., a Pandas DataFrame) in the Data Lake:

```python
import pandas as pd

# Create a sample DataFrame
data = pd.DataFrame({
    "timestamp": ["2024-01-01T10:00:00", "2024-01-01T10:01:00", "2024-01-01T10:02:00"],
    "price": [100, 101, 99],
    "volume": [200, 250, 180],
})

# Store the data
datalake.store_data(
    dataset_name="example_dataset",
    data=data,
    access_key=134,  # Valid access key
    processed=False,
    metadata={"description": "Sample dataset for testing"}
)
```

```css
Data stored at: data_lake/raw/example_dataset.pkl
Metadata updated: {'Author': 'Andy', 'processed': False, 'modification_time': '2024-11-16T12:00:00', 'data_type': 'DataFrame', 'data_structure': {'columns': ['timestamp', 'price', 'volume'], 'data_types': {'timestamp': 'object', 'price': 'int64', 'volume': 'int64'}, 'row_count': 3, 'index': None}, 'file_size': '4.50 KB'}
```
## Step 3: Retrieve Data
Retrieve stored data for further use:

```python
# Retrieve the data
retrieved_data = datalake.retrieve_data(
    dataset_name="example_dataset",
    access_key=134,
    processed=False
)
print(retrieved_data)
```

Output:

```yaml
             timestamp  price  volume
0  2024-01-01T10:00:00    100     200
1  2024-01-01T10:01:00    101     250
2  2024-01-01T10:02:00     99     180
```
## Step 4: Execute SQL Queries
Perform SQL queries on structured datasets like DataFrames:

```python
# Execute an SQL query
sql_query = "SELECT * FROM example_dataset WHERE price > 100"
query_result = datalake.execute_sql(sql_query, access_key=134)
print(query_result)
```

## Step 5: List Datasets
List all datasets stored in the Data Lake:

```python
# List datasets
datasets = datalake.list_datasets(access_key=134, processed=False)
print("Available raw datasets:", datasets)
```
## Step 6: Retrieve Metadata
Retrieve metadata for a specific dataset:

```python
# Retrieve metadata
metadata = datalake.get_metadata(dataset_name="example_dataset", access_key=134)
print("Metadata for 'example_dataset':", metadata)
```

## Step 7: Delete Datasets
Securely delete a dataset and its metadata:
```python
# Delete a dataset
datalake.delete_dataset(dataset_name="example_dataset", access_key=134, processed=False)
```

## **Error Handling**

### **Access Denied**
Ensure you provide a valid `access_key` for all operations.

