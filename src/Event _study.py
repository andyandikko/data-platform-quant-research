"""
# Event Study Script - Report and Workflow Explanation

## Overview
This Python script demonstrates an **Event Study Analysis** by utilizing a set of classes and platform components. It evaluates the impact of news sentiment on intraday stock price movements. The script is structured as follows:

1. **Data Preparation and Validation**: Load, clean, and validate data for intraday prices and news events.
2. **Platform Initialization**: Leverage `DataWorkbench` and `DataCatalog` for data storage and management.
3. **Quantitative Modeling**: Use `IntradayDataModel` and `NewsDataModel` to process and analyze the data.
4. **Data Analysis**: Merge datasets and calculate the sentiment impact on stock returns.

## Workflow Components

### 1. `IntradayDataModel` and `NewsDataModel`
These are specialized models for working with financial intraday data and news data.

- **`IntradayDataModel`**:
  - The IntradayDataModel is designed to process and analyze intraday financial data, such as stock prices and volumes. 
  - It supports features like technical indicator calculation, transformations, and VWAP (Volume Weighted Average Price) analysis.
  - Provides methods like `vwap_calculation` to compute VWAP (Volume Weighted Average Price).   
  - Allows transformations on intraday data, such as calculating returns.
  - We leveraged the ta library for technical analysis calculations and current workflow can be extended to include more technical indicators.

- **`NewsDataModel`**:
  - Enables sentiment analysis and entity extraction from news headlines, using a pretrained bert model.
  - Natural language processing allowing sentiment analysis and entity extraction.

### 2. `DataWorkbench`
This is the central component for storing and retrieving datasets. Key features include:
- **Data Storage**: Stores raw and processed data into a structured DataLake.
- **Transformation Support**: Facilitates applying transformations to datasets.

### 3. `DataCatalog`
The `DataCatalog` organizes datasets into searchable categories, allowing easy retrieval for event studies or other workflows.

## Workflow Steps

### Step 1: Data Loading and Preparation
- **Data Sources**: Intraday and news data are loaded from `.txt` files.
- **Datetime Parsing**: Timestamps are converted to datetime objects for accurate alignment.
- **Validation**: Ensures all required columns (`timestamp`, `price`, etc.) are present in the datasets.

### Step 2: Data Storage
- The `DataWorkbench` stores the datasets into a DataLake:
  - `intraday_data_event_study`
  - `news_data_event_study`
- Metadata (e.g., data type and description) is associated with each dataset.
- Datasets are added to the `DataCatalog` under organized categories.

### Step 3: Data Retrieval and Processing
- Data is retrieved from the DataLake and loaded into the respective Quant Models:
  - **Intraday Data**: Processed with VWAP and returns calculations.
  - **News Data**: Analyzed for sentiment scores and extracted entities.

### Step 4: Data Merging
- **Alignment**: Intraday and news data are merged based on timestamps using `pd.merge_asof`, allowing a backward merge for news impact analysis.
- **Sentiment Impact**: Calculates the effect of sentiment on stock returns using the formula:
  \\\[ \text{sentiment\_impact} = \text{returns} \times \text{sentiment\_score} \\\]

### Step 5: Summary Analysis
- Aggregates results by headline, calculating average `sentiment_impact` and `returns` for each news event.

## Outputs
1. **Merged Data**: A dataset combining intraday and news data with sentiment impacts.
2. **Summary Results**: Displays aggregated sentiment impact and returns per headline.

## Error Handling

import pandas as pd
from src.QuantModels import IntradayDataModel, NewsDataModel
from src.DataWorkbench import DataWorkbench
from src.DataCatalog import DataCatalog

"""

def main():
    # Define paths to the sample data files
    intraday_data_path = "./data/sample_intraday_data.txt"
    news_data_path = "./data/sample_news_data.txt"

    # Load the sample data from text files
    intraday_data = pd.read_csv(intraday_data_path, sep="\t")
    news_data = pd.read_csv(news_data_path, sep="\t")

    # Ensure proper datetime parsing for timestamps
    intraday_data['timestamp'] = pd.to_datetime(intraday_data['timestamp'])
    news_data['timestamp'] = pd.to_datetime(news_data['timestamp'])

    # Validate the structure of the data
    assert {'timestamp', 'price', 'volume', 'symbol'}.issubset(intraday_data.columns), \
        "Intraday data is missing required columns!"
    assert {'timestamp', 'headline', 'relevance', 'source'}.issubset(news_data.columns), \
        "News data is missing required columns!"

    # Initialize platform components
    base_path = "example_event_study"
    workbench = DataWorkbench(base_path)
    catalog = DataCatalog(base_path)
    access_key = 134  # Andy's access key for secure access

    # Store the loaded data into the DataLake
    workbench.store_data(
        dataset_name="intraday_data_event_study",
        data=intraday_data,
        access_key=access_key,
        metadata={"type": "intraday", "description": "Sample intraday data for event study"},
        force=True
    )
    workbench.store_data(
        dataset_name="news_data_event_study",
        data=news_data,
        access_key=access_key,
        metadata={"type": "news", "description": "Sample news data for event study"},
        force=True
    )

    # Add datasets to the catalog for easy searching and organization
    catalog.add_category("EventStudy/Intraday", access_key=access_key)
    catalog.add_dataset_to_category(
        "EventStudy/Intraday",
        "intraday_data_event_study",
        access_key=access_key
    )
    catalog.add_category("EventStudy/News", access_key=access_key)
    catalog.add_dataset_to_category(
        "EventStudy/News",
        "news_data_event_study",
        access_key=access_key
    )

    # Load the data from the DataLake into Quant Models
    intraday_model = IntradayDataModel.from_workbench(workbench, "intraday_data_event_study", access_key)
    news_model = NewsDataModel.from_workbench(workbench, "news_data_event_study", access_key)

    if intraday_model and news_model:
        # Validate loaded data
        if not intraday_model.validate_data():
            print("Intraday data failed validation.")
            return

        if not news_model.validate_data():
            print("News data failed validation.")
            return

        print("Data loaded successfully into models.")

        # Step 1: Transform the intraday data
        intraday_transformed = intraday_model.apply_transformations([
            IntradayDataModel.vwap_calculation,
            lambda df: df.assign(returns=df['price'].pct_change())
        ])

        if not intraday_transformed.success:
            print(f"Intraday data transformation failed: {intraday_transformed.message}")
            return

        # Step 2: Analyze the news data
        news_analysis_result = news_model.analyze_articles()
        if not news_analysis_result.success:
            print(f"News data analysis failed: {news_analysis_result.message}")
            return

        # Step 3: Merge intraday and news data for event study
        intraday_data = intraday_transformed.data
        news_data = news_analysis_result.data

        # Align the two datasets based on timestamp
        merged_data = pd.merge_asof(
            intraday_data.sort_values("timestamp"),
            news_data.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )

        if merged_data.empty:
            print("Merged data is empty. Check timestamp alignment between datasets.")
            return

        # Step 4: Calculate the impact of sentiment on intraday returns
        merged_data['sentiment_impact'] = merged_data['returns'] * merged_data['sentiment_score']

        # Step 5: Aggregate results for summary analysis
        sentiment_analysis = merged_data.groupby('headline')[['sentiment_impact', 'returns']].mean()

        # Display results
        print("\nEvent Study Results:")
        print(sentiment_analysis)
    else:
        print("Failed to load data into models.")


# Execute the script if run as the main program
if __name__ == "__main__":
    main()


