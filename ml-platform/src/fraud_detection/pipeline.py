"""
Complete ML pipeline for fraud detection
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import mlflow
import mlflow.sklearn
import mlflow.tensorflow

from data_processor import FraudDataProcessor, create_sample_fraud_data
from models import ModelFactory, ModelEvaluator, BaseModel

logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """
    Complete ML pipeline for fraud detection including data processing,
    model training, evaluation, and deployment
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.processor = None
        self.models = {}
        self.best_model = None
        self.evaluation_results = {}
        self.experiment_id = None
        
        # Setup MLflow
        if self.config.get('use_mlflow', True):
            self._setup_mlflow()
    
    def _get_default_config(self) -> Dict:
        """Get default pipeline configuration"""
        return {
            # Data processing
            'test_size': 0.2,
            'validation_size': 0.2,
            'random_state': 42,
            'stratify': True,
            
            # Models to train
            'models': [
                {'type': 'random_forest', 'params': {'n_estimators': 100, 'max_depth': 10}},
                {'type': 'gradient_boosting', 'params': {'n_estimators': 100, 'learning_rate': 0.1}},
                {'type': 'xgboost', 'params': {'n_estimators': 100, 'learning_rate': 0.1}},
                {'type': 'lightgbm', 'params': {'n_estimators': 100, 'learning_rate': 0.1}},
                {'type': 'logistic_regression', 'params': {'C': 1.0}},
                {'type': 'tensorflow_dnn', 'params': {'hidden_layers': [256, 128, 64], 'epochs': 50}}
            ],
            
            # Data processor config
            'processor_config': {
                'scaling_method': 'robust',
                'encoding_method': 'onehot',
                'handle_imbalance': True,
                'imbalance_method': 'smote',
                'create_derived_features': True,
                'feature_selection': True
            },
            
            # MLflow
            'use_mlflow': True,
            'experiment_name': 'fraud_detection',
            'mlflow_tracking_uri': 'http://localhost:5000',
            
            # Model selection
            'selection_metric': 'auc_score',
            'cv_folds': 5,
            
            # Output paths
            'model_output_dir': 'models',
            'results_output_dir': 'results',
            'plots_output_dir': 'plots'
        }
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking"""
        try:
            if self.config.get('mlflow_tracking_uri'):
                mlflow.set_tracking_uri(self.config['mlflow_tracking_uri'])
            
            experiment_name = self.config.get('experiment_name', 'fraud_detection')
            
            # Try to get existing experiment
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                self.experiment_id = experiment.experiment_id
            except:
                # Create new experiment
                self.experiment_id = mlflow.create_experiment(experiment_name)
            
            logger.info(f"MLflow experiment '{experiment_name}' initialized with ID: {self.experiment_id}")
            
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}. Continuing without MLflow.")
            self.config['use_mlflow'] = False
    
    def load_data(self, data_source: str = None, **kwargs) -> pd.DataFrame:
        """
        Load data from various sources
        """
        logger.info("Loading data...")
        
        if data_source is None:
            # Create sample data for demonstration
            df = create_sample_fraud_data(kwargs.get('n_samples', 10000))
            logger.info("Created sample fraud data for demonstration")
        
        elif data_source.endswith('.csv'):
            df = pd.read_csv(data_source)
            logger.info(f"Loaded data from CSV: {data_source}")
        
        elif data_source.endswith('.parquet'):
            df = pd.read_parquet(data_source)
            logger.info(f"Loaded data from Parquet: {data_source}")
        
        elif data_source == 'database':
            # Would implement database connection here
            raise NotImplementedError("Database loading not implemented")
        
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
        
        logger.info(f"Data loaded with shape: {df.shape}")
        logger.info(f"Fraud rate: {df.get('is_fraud', pd.Series()).mean():.3f}")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame, target_col: str = 'is_fraud') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Preprocess data and split into train/validation/test sets
        """
        logger.info("Preprocessing data...")
        
        # Initialize processor
        self.processor = FraudDataProcessor(config=self.config['processor_config'])
        
        # Fit and transform data
        X, y = self.processor.fit_transform(df, target_col=target_col)
        
        # Split data
        test_size = self.config['test_size']
        val_size = self.config['validation_size']
        random_state = self.config['random_state']
        stratify = y if self.config['stratify'] else None
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
        stratify_temp = y_temp if self.config['stratify'] else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=stratify_temp
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """
        Train multiple models
        """
        logger.info("Training models...")
        
        model_configs = self.config['models']
        
        for model_config in model_configs:
            model_type = model_config['type']
            model_params = model_config.get('params', {})
            
            logger.info(f"Training {model_type} model...")
            
            try:
                # Create model
                model = ModelFactory.create_model(model_type, **model_params)
                
                # Train with MLflow tracking
                if self.config['use_mlflow']:
                    with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                        # Log parameters
                        mlflow.log_params(model_params)
                        mlflow.log_param("model_type", model_type)
                        
                        # Train model
                        if model_type == 'tensorflow_dnn' and X_val is not None:
                            model.fit(X_train, y_train, validation_data=(X_val, y_val))
                        else:
                            model.fit(X_train, y_train)
                        
                        # Log model
                        if model_type == 'tensorflow_dnn':
                            mlflow.tensorflow.log_model(model.model, f"model_{model_type}")
                        else:
                            mlflow.sklearn.log_model(model.model, f"model_{model_type}")
                        
                        # Cross-validation if not TensorFlow
                        if model_type != 'tensorflow_dnn':
                            cv_scores = cross_val_score(
                                model.model, X_train, y_train, 
                                cv=self.config['cv_folds'], 
                                scoring='roc_auc'
                            )
                            mlflow.log_metric("cv_auc_mean", cv_scores.mean())
                            mlflow.log_metric("cv_auc_std", cv_scores.std())
                
                else:
                    # Train without MLflow
                    if model_type == 'tensorflow_dnn' and X_val is not None:
                        model.fit(X_train, y_train, validation_data=(X_val, y_val))
                    else:
                        model.fit(X_train, y_train)
                
                self.models[model_type] = model
                logger.info(f"{model_type} model training completed")
                
            except Exception as e:
                logger.error(f"Failed to train {model_type} model: {e}")
                continue
        
        logger.info(f"Completed training {len(self.models)} models")
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate all trained models
        """
        logger.info("Evaluating models...")
        
        model_list = list(self.models.values())
        
        if not model_list:
            logger.error("No models to evaluate")
            return
        
        # Compare models
        comparison_df = ModelEvaluator.compare_models(model_list, X_test, y_test)
        
        # Store detailed evaluation results
        for model in model_list:
            eval_results = ModelEvaluator.evaluate_model(model, X_test, y_test)
            self.evaluation_results[model.model_name] = eval_results
            
            # Log to MLflow if enabled
            if self.config['use_mlflow']:
                with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"eval_{model.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    mlflow.log_metrics({
                        'test_auc': eval_results['auc_score'],
                        'test_accuracy': eval_results['classification_report']['accuracy'],
                        'test_precision': eval_results['classification_report']['1']['precision'],
                        'test_recall': eval_results['classification_report']['1']['recall'],
                        'test_f1': eval_results['classification_report']['1']['f1-score']
                    })
        
        # Select best model
        selection_metric = self.config['selection_metric']
        if selection_metric in comparison_df.columns:
            best_model_name = comparison_df.loc[comparison_df[selection_metric].idxmax(), 'model_name']
            self.best_model = self.models[best_model_name]
            logger.info(f"Best model selected: {best_model_name} ({selection_metric}: {comparison_df[selection_metric].max():.4f})")
        
        return comparison_df
    
    def save_artifacts(self):
        """
        Save models, processor, and results
        """
        logger.info("Saving artifacts...")
        
        # Create output directories
        for dir_name in [self.config['model_output_dir'], self.config['results_output_dir']]:
            os.makedirs(dir_name, exist_ok=True)
        
        # Save processor
        if self.processor:
            processor_path = os.path.join(self.config['model_output_dir'], 'fraud_processor.joblib')
            self.processor.save_processor(processor_path)
        
        # Save best model
        if self.best_model:
            model_path = os.path.join(self.config['model_output_dir'], f'best_model_{self.best_model.model_name}.joblib')
            self.best_model.save_model(model_path)
        
        # Save all models
        for model_name, model in self.models.items():
            model_path = os.path.join(self.config['model_output_dir'], f'{model_name}_model.joblib')
            model.save_model(model_path)
        
        # Save evaluation results
        results_path = os.path.join(self.config['results_output_dir'], 'evaluation_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in self.evaluation_results.items():
            serializable_results[model_name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[model_name][key] = value.tolist()
                elif isinstance(value, dict) and 'precision' in value:
                    # Handle precision_recall_curve
                    serializable_results[model_name][key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v 
                        for k, v in value.items()
                    }
                else:
                    serializable_results[model_name][key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save configuration
        config_path = os.path.join(self.config['results_output_dir'], 'pipeline_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info("Artifacts saved successfully")
    
    def run_pipeline(self, data_source: str = None, target_col: str = 'is_fraud', **kwargs):
        """
        Run the complete ML pipeline
        """
        logger.info("Starting fraud detection ML pipeline...")
        
        try:
            # Load data
            df = self.load_data(data_source, **kwargs)
            
            # Preprocess data
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(df, target_col)
            
            # Train models
            self.train_models(X_train, y_train, X_val, y_val)
            
            # Evaluate models
            comparison_df = self.evaluate_models(X_test, y_test)
            
            # Save artifacts
            self.save_artifacts()
            
            logger.info("Pipeline completed successfully!")
            
            return {
                'best_model': self.best_model.model_name if self.best_model else None,
                'model_comparison': comparison_df,
                'evaluation_results': self.evaluation_results
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions using the best model
        """
        if not self.best_model:
            raise ValueError("No trained model available for prediction")
        
        if not self.processor:
            raise ValueError("Data processor not available")
        
        # Preprocess data
        X_processed = self.processor.transform(data)
        
        # Make predictions
        predictions = self.best_model.predict(X_processed)
        probabilities = self.best_model.predict_proba(X_processed)[:, 1]
        
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    @classmethod
    def load_pipeline(cls, model_dir: str) -> 'FraudDetectionPipeline':
        """
        Load a saved pipeline
        """
        # Load configuration
        config_path = os.path.join(model_dir, '..', 'results', 'pipeline_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        pipeline = cls(config)
        
        # Load processor
        processor_path = os.path.join(model_dir, 'fraud_processor.joblib')
        pipeline.processor = FraudDataProcessor.load_processor(processor_path)
        
        # Load best model (find it by looking for best_model_*.joblib)
        import glob
        best_model_files = glob.glob(os.path.join(model_dir, 'best_model_*.joblib'))
        if best_model_files:
            pipeline.best_model = BaseModel.load_model(best_model_files[0])
        
        return pipeline


def create_pipeline_config() -> Dict:
    """
    Create a comprehensive pipeline configuration
    """
    return {
        'test_size': 0.2,
        'validation_size': 0.2,
        'random_state': 42,
        'stratify': True,
        
        'models': [
            {
                'type': 'random_forest',
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                }
            },
            {
                'type': 'xgboost',
                'params': {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 8,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }
            },
            {
                'type': 'lightgbm',
                'params': {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 8,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }
            },
            {
                'type': 'tensorflow_dnn',
                'params': {
                    'hidden_layers': [512, 256, 128, 64],
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'batch_size': 512
                }
            }
        ],
        
        'processor_config': {
            'scaling_method': 'robust',
            'encoding_method': 'onehot',
            'handle_imbalance': True,
            'imbalance_method': 'smote',
            'create_derived_features': True,
            'feature_selection': True,
            'outlier_detection': True
        },
        
        'use_mlflow': True,
        'experiment_name': 'fraud_detection_production',
        'selection_metric': 'auc_score',
        'cv_folds': 5,
        
        'model_output_dir': 'models',
        'results_output_dir': 'results',
        'plots_output_dir': 'plots'
    }


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run pipeline
    config = create_pipeline_config()
    pipeline = FraudDetectionPipeline(config)
    
    # Run pipeline with sample data
    results = pipeline.run_pipeline(n_samples=20000)
    
    print("\n" + "="*50)
    print("PIPELINE RESULTS")
    print("="*50)
    print(f"Best model: {results['best_model']}")
    print("\nModel comparison:")
    print(results['model_comparison'])
    
    # Test prediction
    test_data = create_sample_fraud_data(100)
    test_data = test_data.drop('is_fraud', axis=1)  # Remove target
    
    predictions = pipeline.predict(test_data)
    print(f"\nTest predictions - Fraud detected: {predictions['predictions'].sum()}/100")
    print(f"Average fraud probability: {predictions['probabilities'].mean():.3f}")
    
    print("\nPipeline completed successfully!")
