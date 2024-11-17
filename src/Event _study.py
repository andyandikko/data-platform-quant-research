import pandas as pd
from src.QuantModels import IntradayDataModel, NewsDataModel
from src.DataWorkbench import DataWorkbench
from src.DataCatalog import DataCatalog

def main():
    # Define paths to the sample data files
    intraday_data_path = "./data/sample_intraday_data.txt"
    news_data_path = "./data/sample_news_data.txt"

    # Load the sample data from text files
    intraday_data = pd.read_csv(intraday_data_path, sep=",")
    news_data = pd.read_csv(news_data_path, sep=",")

    # Initialize platform components
    base_path = "example_event_study"  # Base path for DataLake
    workbench = DataWorkbench(base_path)  # Data management system
    catalog = DataCatalog(base_path)  # Metadata management system
    access_key = 134  # Andy's access key for secure access

    # Store the loaded data into the DataLake
    # Store intraday data
    workbench.store_data(
        dataset_name="intraday_data_event_study",
        data=intraday_data,
        access_key=access_key,
        metadata={"type": "intraday", "description": "Sample intraday data for event study"},
        force=True  # Overwrite if already exists
    )
    # Store news data
    workbench.store_data(
        dataset_name="news_data_event_study",
        data=news_data,
        access_key=access_key,
        metadata={"type": "news", "description": "Sample news data for event study"},
        force=True  # Overwrite if already exists
    )

    # Add datasets to the catalog for easy searching and organization
    catalog.add_category("EventStudy/Intraday", access_key=access_key)
    catalog.add_dataset_to_category(
        "EventStudy/Intraday",
        "intraday_data",
        access_key=access_key
    )
    catalog.add_category("EventStudy/News", access_key=access_key)
    catalog.add_dataset_to_category(
        "EventStudy/News",
        "news_data",
        access_key=access_key
    )

    # Load the data from the DataLake into Quant Models
    intraday_model = IntradayDataModel.from_workbench(workbench, "intraday_data_event_study", access_key)
    news_model = NewsDataModel.from_workbench(workbench, "news_data_event_study", access_key)
    
    # Check if models loaded successfully
    if intraday_model and news_model:
        
        # Step 1: Transform the intraday data
        # Apply VWAP calculation and returns calculation as transformations
        intraday_transformed = intraday_model.apply_transformations([
            IntradayDataModel.vwap_calculation,  # VWAP calculation
            lambda df: df.assign(returns=df['price'].pct_change())  # Calculate returns
        ])

        # Step 2: Analyze the news data
        # Apply sentiment analysis and entity extraction on the news data
        news_analysis_result = news_model.analyze_articles()

        # Step 3: Merge intraday and news data for event study
        if intraday_transformed.success and news_analysis_result.success:
            intraday_data = intraday_transformed.data
            news_data = news_analysis_result.data

            # Align the two datasets based on timestamp
            merged_data = pd.merge_asof(
                intraday_data.sort_values("timestamp"),
                news_data.sort_values("timestamp"),
                on="timestamp",
                direction="backward"
            )

            # Step 4: Calculate the impact of sentiment on intraday returns
            merged_data['sentiment_impact'] = merged_data['returns'] * merged_data['sentiment_score']
            

            # Step 5: Aggregate results for summary analysis
            sentiment_analysis = merged_data.groupby('headline')[['sentiment_impact', 'returns']].mean()

            # Display results
            print("\nEvent Study Results:")
            print(sentiment_analysis)
        else:
            print("Data transformation or analysis failed.")
    else:
        print("Failed to load data into models.")

# Execute the script if run as the main program
if __name__ == "__main__":
    main()
