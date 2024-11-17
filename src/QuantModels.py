"""
# **Detailed Explanation of IntradayDataModel and NewsDataModel**

## **Overview**
The `IntradayDataModel` and `NewsDataModel` are specialized Python classes designed to handle and analyze financial data. These models are part of a Quantitative Analysis platform that utilizes a modular approach with components like:
- **DataWorkbench**: Handles data storage and retrieval.
- **QuantModels**: Provides domain-specific data manipulation and analysis functionalities.

---

## **1. IntradayDataModel**

### **Purpose**
The `IntradayDataModel` is designed to process and analyze intraday financial data, such as stock prices and volumes. It supports features like technical indicator calculation, transformations, and VWAP (Volume Weighted Average Price) analysis.

### **Required Columns**
The following columns must exist in the intraday data for it to be valid:
- `timestamp`
- `price`
- `volume`
- `symbol`

### **Key Features and Methods**

#### **Initialization**
```python
def __init__(self, data: Optional[pd.DataFrame] = None)
```
Initializes the model with optional input data and metadata, such as creation time and model type.

---

#### **Data Validation**
```python
def validate_data(self) -> bool
```
Validates if the required columns are present in the dataset.

---

#### **Technical Indicators**
```python
@staticmethod
def create_technical_indicators() -> List[Callable]
```
Generates common technical indicators such as:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- Bollinger Bands
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)

---

#### **Data Loading**
```python
def load_from_yfinance(self, ticker: str, start: str, end: str, interval: str = "1m", period: str = None, add_indicators: bool = False)
```
Fetches data from Yahoo Finance and supports:
- Date range (`start`, `end`) or a fixed `period`.
- Adding pre-defined technical indicators.

Returns a `TransformationResult` containing:
- Transformed data
- Metadata
- Success or failure message

---

#### **VWAP Calculation**
```python
@staticmethod
def vwap_calculation(data: pd.DataFrame) -> pd.DataFrame
```
Calculates the VWAP, which is essential for intraday price analysis.

---

#### **Volatility Calculation**
```python
@staticmethod
def volatility_calculation(window: int = 20) -> Callable
```
Returns a transformation function to calculate rolling volatility over a specified window.

---

#### **Transformations**
```python
def apply_transformations(self, transformations: List[Callable], update_data: bool = True) -> TransformationResult
```
Applies a series of transformations to the data. Logs each transformation step and updates the model's data if `update_data=True`.

---

#### **Export to Workbench**
```python
def to_workbench(self, workbench: DataWorkbench, dataset_name: str, access_key: int)
```
Stores the processed data into the `DataWorkbench`.

---

#### **Load from Workbench**
```python
@classmethod
def from_workbench(cls, workbench: DataWorkbench, dataset_name: str, access_key: int)
```
Loads the model from data stored in the `DataWorkbench`.

---

## **2. NewsDataModel**

### **Purpose**
The `NewsDataModel` is designed to process financial news and extract insights using sentiment analysis and named entity recognition (NER).

### **Required Columns**
The following columns must exist in the news data for it to be valid:
- `timestamp`
- `headline`
- `relevance`

### **Key Features and Methods**

#### **Initialization**
```python
def __init__(self, data: Optional[pd.DataFrame] = None, model_name="bert-base-uncased")
```
Initializes the model with:
- Pre-trained **BERT** models for sentiment analysis and NER.
- Optional input data and metadata.

---

#### **Data Validation**
```python
def validate_data(self) -> bool
```
Validates if the required columns are present in the dataset.

---

#### **Text Cleaning**
```python
@staticmethod
def clean_text(text: str) -> str
```
Preprocesses the text by removing special characters and converting it to lowercase.

---

#### **Sentiment Analysis**
```python
def predict_sentiment(self, text: str) -> float
```
Uses a pre-trained BERT model to predict the sentiment score (0 for negative, 1 for positive).

---

#### **Entity Extraction**
```python
def extract_entities(self, text: str) -> List[str]
```
Extracts named entities (e.g., company names, financial terms) using BERT's NER pipeline.

---

#### **Article Analysis**
```python
def analyze_articles(self) -> TransformationResult
```
Processes all articles in the dataset:
- Calculates sentiment scores for headlines.
- Extracts entities from the text.
- Logs the analysis step.

Returns a `TransformationResult`.

---

#### **Top Headlines**
```python
def get_top_headlines(self, n: int = 5) -> pd.DataFrame
```
Fetches the top N headlines based on sentiment scores.

---

#### **Export to Workbench**
```python
def to_workbench(self, workbench: DataWorkbench, dataset_name: str, access_key: int)
```
Stores the processed data into the `DataWorkbench`.

---

#### **Load from Workbench**
```python
@classmethod
def from_workbench(cls, workbench: DataWorkbench, dataset_name: str, access_key: int)
```
Loads the model from data stored in the `DataWorkbench`.

---

## **Workflow: Integrating IntradayDataModel and NewsDataModel**

1. **Data Ingestion**
   - Use `load_from_yfinance` for intraday data or load data directly from `DataWorkbench`.
   - Load news data with `analyze_articles` for sentiment scoring and NER.

2. **Transformation**
   - Apply transformations such as VWAP and returns calculation to intraday data.
   - Extract sentiment scores and entities from news data.

3. **Analysis**
   - Merge datasets on the `timestamp` field using pandas' `merge_asof`.
   - Calculate correlations or impacts (e.g., sentiment impact on price returns).

4. **Export and Storage**
   - Save processed data to the `DataWorkbench`.
   - Use the `DataCatalog` for organizing and searching datasets.

---

## **Potential Use Cases**

1. **Intraday Trading Strategies**
   - Calculate VWAP and RSI for entry/exit signals.
   - Analyze price trends using rolling volatility.

2. **Event Studies**
   - Study the effect of major news headlines on intraday price movements.
   - Measure sentiment impact on returns.

3. **Portfolio Optimization**
   - Integrate sentiment-driven adjustments into portfolio weights.
   - Use intraday volatility as a risk factor.

4. **Data-Driven Alerts**
   - Generate alerts for high sentiment-impact events.
   - Track unusual intraday price and volume patterns.

---

## **Conclusion**
The `IntradayDataModel` and `NewsDataModel` classes are designed to handle distinct but complementary aspects of financial data. Together, they provide a robust framework for analyzing the relationship between market movements and news events. By integrating with the `DataWorkbench` and `DataCatalog`, these models ensure seamless data management and reproducibility.

"""

from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import pandas_ta as ta
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from torch.nn.functional import softmax
import re
from src.DataWorkbench import DataWorkbench

@dataclass
class TransformationResult:
    """Container for transformation results."""
    data: pd.DataFrame
    metadata: Dict[str, Any]
    success: bool
    message: str = ""

class BaseQuantModel(ABC):
    """Abstract base class for quant models."""
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        self.data = data
        self.metadata: Dict[str, Any] = {}
        self.processing_history: List[Dict] = []
        
    @abstractmethod
    def validate_data(self) -> bool:
        """Validate data structure."""
        pass
    
    def log_transformation(self, transformation_name: str, params: Dict = None) -> None:
        """Log transformation details."""
        self.processing_history.append({
            "transformation": transformation_name,
            "parameters": params or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def get_processing_history(self) -> List[Dict]:
        """Get processing history."""
        return self.processing_history
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata."""
        self.metadata[key] = value

class IntradayDataModel(BaseQuantModel):
    """Enhanced Intraday Data Model."""
    
    REQUIRED_COLUMNS = {'timestamp', 'price', 'volume', 'symbol'}
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        super().__init__(data)
        self.metadata.update({
            "model_type": "intraday",
            "created_at": datetime.now().isoformat()
        })

    def validate_data(self) -> bool:
        """Validate intraday data structure."""
        if self.data is None:
            return False
        return all(col in self.data.columns for col in self.REQUIRED_COLUMNS)

    @staticmethod
    def create_technical_indicators() -> List[Callable]:
        """Create standard technical indicators."""
        return [
            lambda df: df.assign(sma_20=df['price'].rolling(20).mean()),
            lambda df: df.assign(ema_20=df['price'].ewm(span=20).mean()),
            lambda df: df.assign(
                bollinger_upper=df['price'].rolling(20).mean() + 
                2 * df['price'].rolling(20).std()
            ),
            lambda df: df.assign(rsi_14=ta.rsi(df['price'], length=14)),
            lambda df: df.assign(
                macd=ta.macd(df['price'])['MACD_12_26_9']
            )
        ]

    def load_from_yfinance(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1m",
        period: Optional[str] = None,  # Add period as an optional argument
        add_indicators: bool = False
    ) -> TransformationResult:
        """Enhanced yfinance data loading."""
        try:
            yf_ticker = yf.Ticker(ticker)
            if period:
                history = yf_ticker.history(period=period, interval=interval)
            else:
                history = yf_ticker.history(start=start, end=end, interval=interval)
            
            if history.empty:
                return TransformationResult(
                    data=pd.DataFrame(),
                    metadata={},
                    success=False,
                    message=f"No data returned for {ticker}"
                )

            # Transform data
            self.data = pd.DataFrame({
                'timestamp': history.index,
                'price': history['Close'],
                'volume': history['Volume'],
                'symbol': ticker,
                'open': history['Open'],
                'high': history['High'],
                'low': history['Low']
            }).reset_index(drop=True)
            
            # Add technical indicators if requested
            if add_indicators:
                for indicator_func in self.create_technical_indicators():
                    self.data = indicator_func(self.data)
            
            # Update metadata
            self.metadata.update({
                "ticker": ticker,
                "interval": interval,
                "start_date": str(self.data['timestamp'].min()),
                "end_date": str(self.data['timestamp'].max()),
                "rows": len(self.data)
            })
            
            self.log_transformation(
                "load_from_yfinance",
                {"ticker": ticker, "interval": interval, "period": period}
            )
            
            return TransformationResult(
                data=self.data,
                metadata=self.metadata,
                success=True,
                message="Data loaded successfully"
            )
        except Exception as e:
            return TransformationResult(
                data=pd.DataFrame(),
                metadata={},
                success=False,
                message=f"Error loading data: {str(e)}"
            )

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate key market metrics."""
        if not self.validate_data():
            return {}
            
        metrics = {
            "volatility": self.data['price'].std(),
            "avg_volume": self.data['volume'].mean(),
            "price_range": self.data['price'].max() - self.data['price'].min(),
            "volume_price_corr": self.data['price'].corr(self.data['volume'])
        }
        
        self.update_metadata("metrics", metrics)
        return metrics

    @staticmethod
    def vwap_calculation(data: pd.DataFrame) -> pd.DataFrame:
        """VWAP calculation as a transformation function."""
        df = data.copy()
        df['vwap'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
        return df

    @staticmethod
    def volatility_calculation(window: int = 20) -> Callable:
        """Volatility calculation as a transformation function."""
        def calculate(data: pd.DataFrame) -> pd.DataFrame:
            df = data.copy()
            df[f'volatility_{window}'] = df['price'].rolling(window).std()
            return df
        return calculate

    def apply_transformations(
        self,
        transformations: List[Union[Callable, Dict]],
        update_data: bool = True
    ) -> TransformationResult:
        """Apply multiple transformations."""
        if not self.validate_data():
            return TransformationResult(
                data=pd.DataFrame(),
                metadata={},
                success=False,
                message="Invalid data structure"
            )
            
        try:
            result = self.data.copy()
            
            for transform in transformations:
                if isinstance(transform, dict):
                    func_name = transform['func']
                    params = transform.get('params', {})
                    
                    if hasattr(self, func_name):
                        func = getattr(self, func_name)
                        if callable(params):
                            result = func(params)(result)
                        else:
                            result = func(result, **params)
                else:
                    result = transform(result)
                
                self.log_transformation(
                    str(transform if callable(transform) else transform['func']),
                    params if isinstance(transform, dict) else None
                )
            
            if update_data:
                self.data = result
                
            return TransformationResult(
                data=result,
                metadata=self.metadata,
                success=True,
                message="Transformations applied successfully"
            )
            
        except Exception as e:
            return TransformationResult(
                data=self.data,
                metadata=self.metadata,
                success=False,
                message=f"Transformation failed: {str(e)}"
            )

    def to_workbench(
        self,
        workbench: 'DataWorkbench',
        dataset_name: str,
        access_key: int,
        store_in_memory: bool = True,
        save_to_datalake: bool = True
    ) -> bool:
        """Export data to DataWorkbench."""
        if not self.validate_data():
            return False
            
        try:
            if store_in_memory:
                workbench.store_data_in_memory(
                    dataset_name=dataset_name,
                    data=self.data,
                    metadata=self.metadata
                )
                
            if save_to_datalake:
                workbench.store_data(
                    dataset_name=dataset_name,
                    data=self.data,
                    access_key=access_key,
                    metadata=self.metadata,
                    force=True
                )
                
            return True
            
        except Exception as e:
            print(f"Failed to export to workbench: {e}")
            return False

    @classmethod
    def from_workbench(
        cls,
        workbench: 'DataWorkbench',
        dataset_name: str,
        access_key: int
    ) -> Optional['IntradayDataModel']:
        """Create model instance from workbench data."""
        try:
            # Try memory first
            data = workbench.retrieve_data_in_memory(dataset_name)
            
            # If not in memory, try DataLake
            if data is None:
                data = workbench.retrieve_data(
                    dataset_name=dataset_name,
                    access_key=access_key
                )
                
            if data is None:
                return None
                
            return cls(data)
            
        except Exception as e:
            print(f"Failed to load from workbench: {e}")
            return None
        
   
class NewsDataModel(BaseQuantModel):
    """News data model with sentiment analysis and entity extraction."""
    
    REQUIRED_COLUMNS = {"timestamp", "headline", "relevance"}
    
    def __init__(self, data: Optional[pd.DataFrame] = None, model_name="bert-base-uncased"):
        super().__init__(data)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.ner_pipeline = pipeline("ner", model=model_name, device=self.device.index if self.device != torch.device("cpu") else -1)
        self.metadata.update({
            "model_type": "news",
            "created_at": datetime.now().isoformat()
        })

    def validate_data(self) -> bool:
        """Validate news data structure."""
        if self.data is None:
            return False
        return self.REQUIRED_COLUMNS.issubset(self.data.columns)

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text for processing."""
        text = text.lower()
        return re.sub(r'[^a-zA-Z0-9\s\%\$\.\,\!\?\-]', '', text)

    def predict_sentiment(self, text: str) -> float:
        """Predict sentiment score using a pre-trained model."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = softmax(outputs.logits, dim=-1)
            return probs[0, 1].item()  # Assuming label 1 is positive sentiment

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        return [entity['word'] for entity in self.ner_pipeline(text)]

    def analyze_articles(self) -> TransformationResult:
        """Analyze all articles in the dataset."""
        if not self.validate_data():
            return TransformationResult(
                data=pd.DataFrame(),
                metadata={},
                success=False,
                message="Invalid data structure"
            )

        try:
            self.data["sentiment_score"] = self.data["headline"].apply(self.predict_sentiment)
            self.data["entities"] = self.data["headline"].apply(
                lambda x: self.extract_entities(self.clean_text(x))
            )
            
            self.log_transformation("analyze_articles")
            return TransformationResult(
                data=self.data,
                metadata=self.metadata,
                success=True,
                message="Articles analyzed successfully"
            )
        except Exception as e:
            return TransformationResult(
                data=self.data,
                metadata=self.metadata,
                success=False,
                message=f"Analysis failed: {str(e)}"
            )

    def to_workbench(
        self,
        workbench: 'DataWorkbench',
        dataset_name: str,
        access_key: int,
        store_in_memory: bool = True,
        save_to_datalake: bool = True
    ) -> bool:
        """Export data to DataWorkbench."""
        if not self.validate_data():
            return False
        
        try:
            if store_in_memory:
                workbench.store_data_in_memory(
                    dataset_name=dataset_name,
                    data=self.data,
                    metadata=self.metadata
                )
                
            if save_to_datalake:
                workbench.store_data(
                    dataset_name=dataset_name,
                    data=self.data,
                    access_key=access_key,
                    metadata=self.metadata,
                    force=True
                )
                
            return True
            
        except Exception as e:
            print(f"Failed to export to workbench: {e}")
            return False

    @classmethod
    def from_workbench(
        cls,
        workbench: 'DataWorkbench',
        dataset_name: str,
        access_key: int
    ) -> Optional['NewsDataModel']:
        """Create model instance from workbench data."""
        try:
            # Try memory first
            data = workbench.retrieve_data_in_memory(dataset_name)
            
            # If not in memory, try DataLake
            if data is None:
                data = workbench.retrieve_data(
                    dataset_name=dataset_name,
                    access_key=access_key
                )
                
            if data is None:
                return None
                
            return cls(data)
            
        except Exception as e:
            print(f"Failed to load from workbench: {e}")
            return None
    def get_top_headlines(self, n: int = 5) -> pd.DataFrame:
        """
        Get the top N headlines based on sentiment score.

        Args:
            n (int): Number of top headlines to return.

        Returns:
            pd.DataFrame: Top N headlines sorted by sentiment score.
        """
        if "sentiment_score" not in self.data.columns:
            raise ValueError("Sentiment scores have not been calculated. Run `analyze_articles` first.")

        return self.data.nlargest(n, "sentiment_score")[["timestamp", "headline", "sentiment_score", "entities"]]
        
          
        