from typing import Dict, List, Any
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random
from src.DataWorkbench import DataWorkbench
from src.DataCatalog import DataCatalog
from src.QuantModels import IntradayDataModel, NewsDataModel

def run_integration_examples():
    """Run complete integration examples with different data types."""
    
    # Initialize platform components
    base_path = "example_data_lake"
    workbench = DataWorkbench(base_path)
    catalog = DataCatalog(base_path)
    access_key = 134  # Andy's access key

    # Example 1: Market Data Workflow
    print("\n=== Example 1: Market Data Workflow ===")
    market_data_workflow(workbench, catalog, access_key)

    # Example 2: News Data Workflow
    print("\n=== Example 2: News Data Workflow ===")
    news_data_workflow(workbench, catalog, access_key)

    # Example 3: Mixed Data Types Workflow
    print("\n=== Example 3: Mixed Data Types Workflow ===")
    mixed_data_workflow(workbench, catalog, access_key)

def create_fake_market_data() -> pd.DataFrame:
    """Create fake market data."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
        freq='1min'
    )
    
    data = pd.DataFrame({
        'timestamp': dates,
        'price': np.random.normal(100, 5, len(dates)).cumsum(),
        'volume': np.random.randint(1000, 10000, len(dates)),
        'symbol': 'FAKE'
    })
    return data

def create_fake_news_data() -> pd.DataFrame:
    """Create fake news data."""
    headlines = [
        "Company XYZ announces record profits",
        "Market volatility increases amid global concerns",
        "Tech sector sees strong growth",
        "New regulations impact financial sector",
        "Merger announcement drives stock movement"
    ]
    
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
        freq='D'
    )
    
    data = pd.DataFrame({
        'timestamp': dates,
        'headline': [random.choice(headlines) for _ in range(len(dates))],
        'relevance': np.random.uniform(0, 1, len(dates)),
        'source': 'Example News'
    })
    return data

def create_fake_metadata() -> Dict:
    """Create fake metadata."""
    return {
        'data_source': 'example',
        'frequency': '1min',
        'description': 'Example dataset for testing',
        'tags': ['test', 'example', 'fake'],
        'created_at': datetime.now().isoformat()
    }

def market_data_workflow(workbench: DataWorkbench, catalog: DataCatalog, access_key: int):
    """Example workflow for market data."""
    try:
        # 1. Create and process market data
        market_data = create_fake_market_data()
        model = IntradayDataModel(market_data)
        
        # 2. Apply transformations
        transformations = [
            IntradayDataModel.vwap_calculation,
            IntradayDataModel.volatility_calculation(window=20),
            lambda df: df.assign(returns=df['price'].pct_change())
        ]
        
        result = model.apply_transformations(transformations)
        if not result.success:
            print(f"Transformation failed: {result.message}")
            return
            
        # 3. Store in workbench/datalake
        dataset_name = "market_data_example"
        model.to_workbench(
            workbench=workbench,
            dataset_name=dataset_name,
            access_key=access_key,
            store_in_memory=True,
            save_to_datalake=True
        )
        
        # 4. Add to catalog
        catalog.add_category("Market/Intraday", access_key=access_key)
        catalog.add_dataset_to_category(
            category_name="Market/Intraday",
            dataset_name=dataset_name,
            access_key=access_key,
            metadata=model.metadata
        )
        
        # 5. Retrieve and verify
        retrieved_data = workbench.retrieve_data(
            dataset_name=dataset_name,
            access_key=access_key
        )
        print(f"Market data shape: {retrieved_data.shape}")
        print(f"Available columns: {retrieved_data.columns.tolist()}")
        
    except Exception as e:
        print(f"Market data workflow failed: {e}")

def news_data_workflow(workbench: DataWorkbench, catalog: DataCatalog, access_key: int):
    """Example workflow for news data."""
    try:
        # 1. Create and process news data
        news_data = create_fake_news_data()
        model = NewsDataModel(news_data)
        
        # 2. Analyze articles
        analysis_result = model.analyze_articles()
        if not analysis_result.success:
            print(f"Failed to analyze articles: {analysis_result.message}")
            return
        
        # 3. Store results
        dataset_name = "news_data_example"
        model.to_workbench(
            workbench=workbench,
            dataset_name=dataset_name,
            access_key=access_key,
            store_in_memory=True,
            save_to_datalake=True
        )
        
        # 4. Add to catalog
        catalog.add_category("News/Daily", access_key=access_key)
        catalog.add_dataset_to_category(
            category_name="News/Daily",
            dataset_name=dataset_name,
            access_key=access_key,
            metadata=model.metadata
        )
        
        # 5. Retrieve and analyze top headlines
        top_headlines = model.get_top_headlines(n=3)
        print("\nTop headlines:")
        print(top_headlines)
    except Exception as e:
        print(f"News data workflow failed: {e}")


def mixed_data_workflow(workbench: DataWorkbench, catalog: DataCatalog, access_key: int):
    """Example workflow with mixed data types."""
    try:
        # 1. Create different data types
        json_data = {
            "config": {
                "sampling_rate": "1min",
                "symbols": ["FAKE1", "FAKE2"],
                "indicators": ["MA", "VWAP"]
            },
            "parameters": {
                "ma_window": 20,
                "vol_window": 10
            }
        }
        
        text_data = """
        Analysis Report
        Date: 2024-01-01
        Symbol: FAKE
        Findings: Example analysis text
        """
        
        # 2. Store different types
        workbench.store_data(
            dataset_name="config_data",
            data=json_data,
            access_key=access_key,
            metadata={"type": "configuration"}
        )
        
        workbench.store_data(
            dataset_name="report_data",
            data=text_data,
            access_key=access_key,
            metadata={"type": "analysis_report"}
        )
        
        # 3. Organize in catalog
        catalog.add_category("Configuration", access_key=access_key)
        catalog.add_category("Reports", access_key=access_key)
        
        catalog.add_dataset_to_category(
            "Configuration",
            "config_data",
            access_key=access_key
        )
        
        catalog.add_dataset_to_category(
            "Reports",
            "report_data",
            access_key=access_key
        )
        
        # 4. Search and retrieve
        config = workbench.retrieve_data("config_data", access_key=access_key)
        report = workbench.retrieve_data("report_data", access_key=access_key)
        
        print("\nStored data types:")
        print(f"Config type: {type(config)}")
        print(f"Report type: {type(report)}")
        
        # 5. Search catalog
        results = catalog.search_datasets("report", access_key=access_key)
        print(f"\nFound {len(results)} datasets matching 'report'")
        
    except Exception as e:
        print(f"Mixed data workflow failed: {e}")
def quant_analysis_example():
    """Example of how a quant would use the platform."""
    
    # Initialize components
    workbench = DataWorkbench("quant_analysis")
    catalog = DataCatalog("quant_analysis")
    access_key = 134  # Andy's access key
    
    try:
        # 1. Load market data
        market_model = IntradayDataModel()
        result = market_model.load_from_yfinance(
            ticker="SPY",
            interval="1m",
            period="1d",
            add_indicators=True
        )
        
        if not result.success:
            print(f"Failed to load market data: {result.message}")
            return
            
        # 2. Store and categorize data
        market_model.to_workbench(
            workbench=workbench,
            dataset_name="spy_intraday",
            access_key=access_key
        )
        
        catalog.add_category("Analysis/SPY", access_key=access_key)
        catalog.add_dataset_to_category(
            "Analysis/SPY",
            "spy_intraday",
            access_key=access_key
        )
        
        # 3. Define custom analysis
        def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
            return data.assign(returns=data['price'].pct_change())

        def custom_volatility_analysis(data: pd.DataFrame) -> pd.DataFrame:
            if 'returns' not in data.columns:
                raise ValueError("'returns' column is missing. Calculate returns first.")
            return data.assign(
                rolling_vol=data['returns'].rolling(20).std() * np.sqrt(252),
                vol_ratio=data['volume'] / data['volume'].rolling(20).mean()
            )
        
        # 4. Register transformations
        workbench.register_transformation(
            name="calculate_returns",
            func=calculate_returns,
            description="Calculate percentage returns"
        )
        
        workbench.register_transformation(
            name="volatility_analysis",
            func=custom_volatility_analysis,
            description="Calculate volatility metrics"
        )
        
        # 5. Apply transformations
        workbench.transform_data(
            dataset_name="spy_intraday",
            transformation_func="calculate_returns",
            access_key=access_key,
            update_in_datalake=True
        )
        
        analyzed_data = workbench.transform_data(
            dataset_name="spy_intraday",
            transformation_func="volatility_analysis",
            access_key=access_key,
            update_in_datalake=True
        )
        
        if analyzed_data is not None:
            print("\nAnalysis Results:")
            print(f"Average Volatility: {analyzed_data['rolling_vol'].mean():.4f}")
            print(f"Average Volume Ratio: {analyzed_data['vol_ratio'].mean():.4f}")
        
        # 6. Check processing history
        history = workbench.get_processing_history("spy_intraday")
        print("\nProcessing Steps:")
        for step in history:
            print(f"- {step['transformation']} at {step['timestamp']}")
        print(history)
    except Exception as e:
        print(f"Analysis failed: {e}")


def main():
    """Main execution function."""
    print("Starting integration examples...")
    run_integration_examples()
    print("\nCompleted integration examples.")
    print("Simualte Quant analyst usage examples...")
    quant_analysis_example()
    print("\nCompleted simualte Quant analyst usage examples.")
    
    

if __name__ == "__main__":
    main()
