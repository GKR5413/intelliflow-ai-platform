"""
Data preprocessing and feature engineering for fraud detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.creation import MathFeatures
import logging
from datetime import datetime, timedelta
import joblib
import os

logger = logging.getLogger(__name__)


class FraudDataProcessor:
    """
    Comprehensive data processor for fraud detection with feature engineering
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = []
        self.is_fitted = False
        
        # Default configuration
        self.default_config = {
            'scaling_method': 'robust',  # 'standard', 'robust', 'minmax'
            'encoding_method': 'onehot',  # 'onehot', 'ordinal', 'target'
            'imputation_method': 'median',  # 'mean', 'median', 'mode', 'knn'
            'handle_imbalance': True,
            'imbalance_method': 'smote',  # 'smote', 'adasyn', 'undersample'
            'feature_selection': True,
            'create_derived_features': True,
            'outlier_detection': True,
            'outlier_method': 'iqr'  # 'iqr', 'zscore', 'isolation_forest'
        }
        
        # Update config with defaults
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'is_fraud') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit the processor and transform the data
        """
        logger.info("Starting data preprocessing and feature engineering")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Store original feature names
        self.original_features = X.columns.tolist()
        
        # Step 1: Handle missing values
        X = self._handle_missing_values(X, fit=True)
        
        # Step 2: Create derived features
        if self.config['create_derived_features']:
            X = self._create_derived_features(X)
        
        # Step 3: Handle categorical variables
        X = self._handle_categorical_features(X, fit=True)
        
        # Step 4: Feature scaling
        X = self._scale_features(X, fit=True)
        
        # Step 5: Handle outliers
        if self.config['outlier_detection']:
            X = self._handle_outliers(X, fit=True)
        
        # Step 6: Feature selection
        if self.config['feature_selection']:
            X = self._select_features(X, y, fit=True)
        
        # Step 7: Handle class imbalance
        if self.config['handle_imbalance']:
            X, y = self._handle_imbalance(X, y)
        
        self.feature_names = X.columns.tolist()
        self.is_fitted = True
        
        logger.info(f"Data preprocessing completed. Final shape: {X.shape}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted processors
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transforming new data")
        
        X = df.copy()
        
        # Apply same transformations as in fit_transform
        X = self._handle_missing_values(X, fit=False)
        
        if self.config['create_derived_features']:
            X = self._create_derived_features(X)
        
        X = self._handle_categorical_features(X, fit=False)
        X = self._scale_features(X, fit=False)
        
        if self.config['outlier_detection']:
            X = self._handle_outliers(X, fit=False)
        
        if self.config['feature_selection']:
            X = self._select_features(X, fit=False)
        
        # Ensure same columns as training data
        X = X.reindex(columns=self.feature_names, fill_value=0)
        
        return X
    
    def _handle_missing_values(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        logger.info("Handling missing values")
        
        if fit:
            # Separate numeric and categorical columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            
            # Create imputers
            if self.config['imputation_method'] == 'knn':
                self.imputers['numeric'] = KNNImputer(n_neighbors=5)
                self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
            else:
                strategy_map = {
                    'mean': 'mean',
                    'median': 'median',
                    'mode': 'most_frequent'
                }
                strategy = strategy_map.get(self.config['imputation_method'], 'median')
                
                self.imputers['numeric'] = SimpleImputer(strategy=strategy if strategy != 'most_frequent' else 'median')
                self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
            
            # Fit and transform
            if len(numeric_cols) > 0:
                X[numeric_cols] = self.imputers['numeric'].fit_transform(X[numeric_cols])
            if len(categorical_cols) > 0:
                X[categorical_cols] = self.imputers['categorical'].fit_transform(X[categorical_cols])
        else:
            # Transform only
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            
            if len(numeric_cols) > 0 and 'numeric' in self.imputers:
                X[numeric_cols] = self.imputers['numeric'].transform(X[numeric_cols])
            if len(categorical_cols) > 0 and 'categorical' in self.imputers:
                X[categorical_cols] = self.imputers['categorical'].transform(X[categorical_cols])
        
        return X
    
    def _create_derived_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for fraud detection"""
        logger.info("Creating derived features")
        
        X = X.copy()
        
        # Transaction amount features
        if 'transaction_amount' in X.columns:
            X['amount_log'] = np.log1p(X['transaction_amount'])
            X['amount_sqrt'] = np.sqrt(X['transaction_amount'])
            
            # Amount categories
            X['amount_category'] = pd.cut(X['transaction_amount'], 
                                        bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                                        labels=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme'])
        
        # Time-based features
        if 'transaction_timestamp' in X.columns:
            X['transaction_timestamp'] = pd.to_datetime(X['transaction_timestamp'])
            X['hour'] = X['transaction_timestamp'].dt.hour
            X['day_of_week'] = X['transaction_timestamp'].dt.dayofweek
            X['is_weekend'] = (X['day_of_week'] >= 5).astype(int)
            X['is_night'] = ((X['hour'] >= 22) | (X['hour'] <= 6)).astype(int)
            X['is_business_hours'] = ((X['hour'] >= 9) & (X['hour'] <= 17)).astype(int)
        
        # Velocity features (requires user_id)
        if 'user_id' in X.columns and 'transaction_timestamp' in X.columns:
            X = X.sort_values(['user_id', 'transaction_timestamp'])
            X['transactions_last_hour'] = X.groupby('user_id')['transaction_timestamp'].rolling('1H').count().values
            X['transactions_last_day'] = X.groupby('user_id')['transaction_timestamp'].rolling('1D').count().values
            X['amount_last_hour'] = X.groupby('user_id')['transaction_amount'].rolling('1H').sum().values
            X['amount_last_day'] = X.groupby('user_id')['transaction_amount'].rolling('1D').sum().values
        
        # Merchant features
        if 'merchant_category' in X.columns:
            # High-risk merchant categories
            high_risk_categories = ['gambling', 'adult_entertainment', 'cryptocurrency']
            X['is_high_risk_merchant'] = X['merchant_category'].isin(high_risk_categories).astype(int)
        
        # Geographic features
        if 'user_country' in X.columns and 'merchant_country' in X.columns:
            X['is_international'] = (X['user_country'] != X['merchant_country']).astype(int)
        
        # Device features
        if 'device_id' in X.columns and 'user_id' in X.columns:
            # Device consistency
            device_counts = X.groupby('user_id')['device_id'].nunique()
            X['user_device_count'] = X['user_id'].map(device_counts)
            X['is_new_device'] = (X['user_device_count'] > 1).astype(int)
        
        # Payment method features
        if 'payment_method' in X.columns:
            high_risk_methods = ['cryptocurrency', 'prepaid_card']
            X['is_high_risk_payment'] = X['payment_method'].isin(high_risk_methods).astype(int)
        
        # Drop timestamp column if it exists (converted to features)
        if 'transaction_timestamp' in X.columns:
            X = X.drop('transaction_timestamp', axis=1)
        
        return X
    
    def _handle_categorical_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle categorical feature encoding"""
        logger.info("Handling categorical features")
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            return X
        
        if fit:
            if self.config['encoding_method'] == 'onehot':
                self.encoders['categorical'] = OneHotEncoder(
                    variables=categorical_cols,
                    drop_last=True,
                    ignore_format=True
                )
            elif self.config['encoding_method'] == 'ordinal':
                self.encoders['categorical'] = OrdinalEncoder(
                    variables=categorical_cols,
                    ignore_format=True
                )
            
            X = self.encoders['categorical'].fit_transform(X)
        else:
            if 'categorical' in self.encoders:
                X = self.encoders['categorical'].transform(X)
        
        return X
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale numerical features"""
        logger.info("Scaling features")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return X
        
        if fit:
            if self.config['scaling_method'] == 'standard':
                self.scalers['numeric'] = StandardScaler()
            elif self.config['scaling_method'] == 'robust':
                self.scalers['numeric'] = RobustScaler()
            elif self.config['scaling_method'] == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                self.scalers['numeric'] = MinMaxScaler()
            
            X[numeric_cols] = self.scalers['numeric'].fit_transform(X[numeric_cols])
        else:
            if 'numeric' in self.scalers:
                X[numeric_cols] = self.scalers['numeric'].transform(X[numeric_cols])
        
        return X
    
    def _handle_outliers(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle outliers in the dataset"""
        logger.info("Handling outliers")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return X
        
        if fit:
            if self.config['outlier_method'] == 'iqr':
                self.outlier_bounds = {}
                for col in numeric_cols:
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    self.outlier_bounds[col] = (lower_bound, upper_bound)
            elif self.config['outlier_method'] == 'zscore':
                self.outlier_bounds = {}
                for col in numeric_cols:
                    mean = X[col].mean()
                    std = X[col].std()
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                    self.outlier_bounds[col] = (lower_bound, upper_bound)
        
        # Apply outlier bounds
        if hasattr(self, 'outlier_bounds'):
            for col, (lower_bound, upper_bound) in self.outlier_bounds.items():
                if col in X.columns:
                    X[col] = X[col].clip(lower_bound, upper_bound)
        
        return X
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = False) -> pd.DataFrame:
        """Select most important features"""
        logger.info("Selecting features")
        
        if fit and y is not None:
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            
            # Use mutual information for feature selection
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(100, X.shape[1])  # Select top 100 features or all if less
            )
            
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_mask = self.feature_selector.get_support()
            self.selected_features = X.columns[selected_mask].tolist()
            
            return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        
        elif hasattr(self, 'selected_features'):
            return X[self.selected_features]
        
        return X
    
    def _handle_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance"""
        logger.info("Handling class imbalance")
        
        if self.config['imbalance_method'] == 'smote':
            sampler = SMOTE(random_state=42)
        elif self.config['imbalance_method'] == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif self.config['imbalance_method'] == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        else:
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    def save_processor(self, filepath: str):
        """Save the fitted processor"""
        logger.info(f"Saving processor to {filepath}")
        
        processor_data = {
            'config': self.config,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        # Save additional attributes if they exist
        if hasattr(self, 'outlier_bounds'):
            processor_data['outlier_bounds'] = self.outlier_bounds
        if hasattr(self, 'feature_selector'):
            processor_data['feature_selector'] = self.feature_selector
        if hasattr(self, 'selected_features'):
            processor_data['selected_features'] = self.selected_features
        if hasattr(self, 'original_features'):
            processor_data['original_features'] = self.original_features
        
        joblib.dump(processor_data, filepath)
    
    @classmethod
    def load_processor(cls, filepath: str) -> 'FraudDataProcessor':
        """Load a fitted processor"""
        logger.info(f"Loading processor from {filepath}")
        
        processor_data = joblib.load(filepath)
        
        processor = cls(config=processor_data['config'])
        processor.scalers = processor_data['scalers']
        processor.encoders = processor_data['encoders']
        processor.imputers = processor_data['imputers']
        processor.feature_names = processor_data['feature_names']
        processor.is_fitted = processor_data['is_fitted']
        
        # Load additional attributes if they exist
        for attr in ['outlier_bounds', 'feature_selector', 'selected_features', 'original_features']:
            if attr in processor_data:
                setattr(processor, attr, processor_data[attr])
        
        return processor


def create_sample_fraud_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Create sample fraud detection dataset for testing
    """
    np.random.seed(42)
    
    data = {
        'user_id': np.random.randint(1, 1000, n_samples),
        'transaction_amount': np.random.lognormal(3, 1.5, n_samples),
        'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'gambling', 'adult_entertainment'], n_samples),
        'payment_method': np.random.choice(['credit_card', 'debit_card', 'bank_transfer', 'cryptocurrency'], n_samples),
        'user_country': np.random.choice(['US', 'CA', 'UK', 'FR', 'DE', 'CN', 'IN'], n_samples),
        'merchant_country': np.random.choice(['US', 'CA', 'UK', 'FR', 'DE', 'CN', 'IN'], n_samples),
        'device_id': np.random.randint(1, 5000, n_samples),
        'transaction_timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='15min')
    }
    
    df = pd.DataFrame(data)
    
    # Create fraud labels based on some rules
    fraud_conditions = (
        (df['transaction_amount'] > 1000) |
        (df['merchant_category'].isin(['gambling', 'adult_entertainment'])) |
        (df['payment_method'] == 'cryptocurrency') |
        (df['user_country'] != df['merchant_country'])
    )
    
    # Add some randomness
    df['is_fraud'] = (fraud_conditions & (np.random.random(n_samples) < 0.3)).astype(int)
    
    # Ensure some minimum fraud rate
    if df['is_fraud'].sum() < n_samples * 0.05:
        fraud_indices = np.random.choice(df.index, size=int(n_samples * 0.05) - df['is_fraud'].sum(), replace=False)
        df.loc[fraud_indices, 'is_fraud'] = 1
    
    return df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    df = create_sample_fraud_data(10000)
    print(f"Sample data shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.3f}")
    
    # Initialize processor
    processor = FraudDataProcessor()
    
    # Fit and transform data
    X, y = processor.fit_transform(df, target_col='is_fraud')
    
    print(f"Processed data shape: {X.shape}")
    print(f"Feature names: {X.columns.tolist()[:10]}...")  # Show first 10 features
    
    # Save processor
    os.makedirs('models', exist_ok=True)
    processor.save_processor('models/fraud_processor.joblib')
    
    print("Data processing completed successfully!")
