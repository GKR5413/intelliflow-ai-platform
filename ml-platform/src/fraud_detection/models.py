"""
Fraud detection models using scikit-learn and TensorFlow
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from abc import ABC, abstractmethod

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, GridSearchCV

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

import joblib
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all fraud detection models"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.feature_importance = None
        self.training_history = {}
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        pass
    
    def save_model(self, filepath: str):
        """Save the model"""
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a saved model"""
        model_data = joblib.load(filepath)
        instance = cls(model_data['model_name'])
        instance.model = model_data['model']
        instance.is_fitted = model_data['is_fitted']
        instance.feature_importance = model_data.get('feature_importance')
        instance.training_history = model_data.get('training_history', {})
        logger.info(f"Model loaded from {filepath}")
        return instance


class SklearnModel(BaseModel):
    """Wrapper for scikit-learn models"""
    
    def __init__(self, model_name: str, model_class, **model_params):
        super().__init__(model_name)
        self.model_class = model_class
        self.model_params = model_params
        self.model = model_class(**model_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit the sklearn model"""
        logger.info(f"Training {self.model_name} model")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Extract feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = dict(zip(X.columns, np.abs(self.model.coef_[0])))
        
        logger.info(f"{self.model_name} model training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # For models like SVM that don't have predict_proba
            decision_scores = self.model.decision_function(X)
            # Convert to probabilities using sigmoid
            probabilities = 1 / (1 + np.exp(-decision_scores))
            return np.column_stack([1 - probabilities, probabilities])
        else:
            raise NotImplementedError(f"Model {self.model_name} doesn't support probability prediction")


class TensorFlowModel(BaseModel):
    """TensorFlow/Keras model for fraud detection"""
    
    def __init__(self, model_name: str = "tensorflow_dnn", **kwargs):
        super().__init__(model_name)
        self.model_params = kwargs
        self.history = None
    
    def build_model(self, input_dim: int) -> keras.Model:
        """Build the neural network architecture"""
        
        # Get architecture parameters
        hidden_layers = self.model_params.get('hidden_layers', [256, 128, 64])
        dropout_rate = self.model_params.get('dropout_rate', 0.3)
        activation = self.model_params.get('activation', 'relu')
        use_batch_norm = self.model_params.get('use_batch_norm', True)
        
        # Input layer
        inputs = Input(shape=(input_dim,), name='input_features')
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(hidden_layers):
            x = Dense(units, activation=activation, name=f'dense_{i+1}')(x)
            
            if use_batch_norm:
                x = BatchNormalization(name=f'batch_norm_{i+1}')(x)
            
            x = Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='fraud_detection_model')
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            validation_data: Tuple[pd.DataFrame, pd.Series] = None,
            **kwargs):
        """Fit the TensorFlow model"""
        
        logger.info(f"Training {self.model_name} model")
        
        # Build model if not exists
        if self.model is None:
            self.model = self.build_model(X.shape[1])
        
        # Compile model
        optimizer = self.model_params.get('optimizer', 'adam')
        learning_rate = self.model_params.get('learning_rate', 0.001)
        
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        # Prepare callbacks
        callbacks_list = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=kwargs.get('patience', 10),
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        
        # Learning rate reduction
        lr_reducer = ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        callbacks_list.append(lr_reducer)
        
        # Model checkpoint
        checkpoint_path = kwargs.get('checkpoint_path', 'models/best_model.h5')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss' if validation_data else 'loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks_list.append(checkpoint)
        
        # Training parameters
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 256)
        
        # Train model
        self.history = self.model.fit(
            X.values, y.values,
            validation_data=(validation_data[0].values, validation_data[1].values) if validation_data else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=kwargs.get('verbose', 1)
        )
        
        self.is_fitted = True
        self.training_history = self.history.history
        
        logger.info(f"{self.model_name} model training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        probabilities = self.model.predict(X.values)
        return (probabilities.flatten() > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        probabilities = self.model.predict(X.values).flatten()
        return np.column_stack([1 - probabilities, probabilities])


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models"""
    
    def __init__(self, model_name: str = "ensemble", base_models: List[BaseModel] = None):
        super().__init__(model_name)
        self.base_models = base_models or []
        self.weights = None
        
    def add_model(self, model: BaseModel, weight: float = 1.0):
        """Add a base model to the ensemble"""
        self.base_models.append((model, weight))
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit all base models"""
        logger.info(f"Training ensemble model with {len(self.base_models)} base models")
        
        for model, weight in self.base_models:
            if not model.is_fitted:
                model.fit(X, y, **kwargs)
        
        self.is_fitted = True
        logger.info("Ensemble model training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions using majority voting"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        for model, weight in self.base_models:
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0)
        return (ensemble_pred > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using weighted average"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        probabilities = []
        weights = []
        
        for model, weight in self.base_models:
            proba = model.predict_proba(X)[:, 1]  # Get fraud probability
            probabilities.append(proba)
            weights.append(weight)
        
        # Weighted average
        ensemble_proba = np.average(probabilities, axis=0, weights=weights)
        return np.column_stack([1 - ensemble_proba, ensemble_proba])


class ModelFactory:
    """Factory class for creating different model types"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """Create a model based on the specified type"""
        
        if model_type == "random_forest":
            return SklearnModel(
                "random_forest",
                RandomForestClassifier,
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
        
        elif model_type == "gradient_boosting":
            return SklearnModel(
                "gradient_boosting",
                GradientBoostingClassifier,
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 6),
                subsample=kwargs.get('subsample', 0.8),
                random_state=42
            )
        
        elif model_type == "logistic_regression":
            return SklearnModel(
                "logistic_regression",
                LogisticRegression,
                C=kwargs.get('C', 1.0),
                penalty=kwargs.get('penalty', 'l2'),
                solver=kwargs.get('solver', 'liblinear'),
                random_state=42,
                max_iter=1000
            )
        
        elif model_type == "svm":
            return SklearnModel(
                "svm",
                SVC,
                C=kwargs.get('C', 1.0),
                kernel=kwargs.get('kernel', 'rbf'),
                probability=True,
                random_state=42
            )
        
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            return SklearnModel(
                "xgboost",
                xgb.XGBClassifier,
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 6),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=42,
                eval_metric='logloss'
            )
        
        elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            return SklearnModel(
                "lightgbm",
                lgb.LGBMClassifier,
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 6),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=42,
                verbosity=-1
            )
        
        elif model_type == "tensorflow_dnn":
            return TensorFlowModel("tensorflow_dnn", **kwargs)
        
        elif model_type == "isolation_forest":
            return SklearnModel(
                "isolation_forest",
                IsolationForest,
                contamination=kwargs.get('contamination', 0.1),
                random_state=42,
                n_jobs=-1
            )
        
        elif model_type == "ensemble":
            return EnsembleModel("ensemble")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class ModelEvaluator:
    """Class for evaluating model performance"""
    
    @staticmethod
    def evaluate_model(model: BaseModel, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_proba)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        
        # Find optimal threshold
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element (precision=1, recall=0)
        optimal_threshold = thresholds[optimal_idx]
        
        evaluation_results = {
            'model_name': model.model_name,
            'auc_score': auc_score,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'optimal_threshold': optimal_threshold,
            'precision_recall_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist()
            },
            'feature_importance': model.feature_importance
        }
        
        logger.info(f"Model {model.model_name} - AUC: {auc_score:.4f}")
        
        return evaluation_results
    
    @staticmethod
    def compare_models(models: List[BaseModel], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Compare multiple models"""
        
        results = []
        
        for model in models:
            eval_results = ModelEvaluator.evaluate_model(model, X_test, y_test)
            
            results.append({
                'model_name': model.model_name,
                'auc_score': eval_results['auc_score'],
                'accuracy': eval_results['classification_report']['accuracy'],
                'precision': eval_results['classification_report']['1']['precision'],
                'recall': eval_results['classification_report']['1']['recall'],
                'f1_score': eval_results['classification_report']['1']['f1-score']
            })
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('auc_score', ascending=False)
        
        return comparison_df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would typically use real data
    from data_processor import create_sample_fraud_data, FraudDataProcessor
    
    # Create sample data
    df = create_sample_fraud_data(5000)
    
    # Process data
    processor = FraudDataProcessor()
    X, y = processor.fit_transform(df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and train models
    models = [
        ModelFactory.create_model("random_forest"),
        ModelFactory.create_model("gradient_boosting"),
        ModelFactory.create_model("logistic_regression"),
        ModelFactory.create_model("tensorflow_dnn", hidden_layers=[128, 64], epochs=10)
    ]
    
    # Train models
    for model in models:
        model.fit(X_train, y_train)
    
    # Evaluate models
    comparison = ModelEvaluator.compare_models(models, X_test, y_test)
    print("\nModel Comparison:")
    print(comparison)
    
    # Save best model
    best_model = models[0]  # Would be determined by evaluation
    os.makedirs('models', exist_ok=True)
    best_model.save_model('models/best_fraud_model.joblib')
    
    print("\nModel training and evaluation completed successfully!")
