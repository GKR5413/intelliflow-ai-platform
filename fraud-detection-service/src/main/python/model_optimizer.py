"""
Model loading and inference optimization for fraud detection
"""

import os
import logging
import time
import threading
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json
import pickle
import gc
from pathlib import Path
import hashlib

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.pyfunc

# ML frameworks
import sklearn
from sklearn.base import BaseEstimator
import tensorflow as tf
import torch
import onnx
import onnxruntime as ort

# Optimization libraries
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

try:
    import openvino.runtime as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

# Memory and performance monitoring
import psutil
from memory_profiler import profile
import cProfile
import pstats

# Caching and serialization
import ray
from functools import lru_cache
import diskcache

# Monitoring
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Configure logging
logger = structlog.get_logger()

# Prometheus metrics
MODEL_LOAD_TIME = Histogram('model_load_duration_seconds', 'Model loading time', ['model_type', 'optimization'])
INFERENCE_LATENCY = Histogram('inference_duration_seconds', 'Inference latency', ['model_type', 'batch_size'])
MODEL_MEMORY_USAGE = Gauge('model_memory_usage_bytes', 'Model memory usage', ['model_type'])
OPTIMIZATION_SUCCESS = Counter('model_optimization_success_total', 'Successful optimizations', ['optimization_type'])
OPTIMIZATION_ERRORS = Counter('model_optimization_errors_total', 'Optimization errors', ['optimization_type'])


class ModelType(Enum):
    """Model type enumeration"""
    SKLEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"


class OptimizationType(Enum):
    """Optimization type enumeration"""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    ONNX_CONVERSION = "onnx_conversion"
    BATCHING = "batching"
    CACHING = "caching"


@dataclass
class ModelConfig:
    """Model configuration"""
    model_path: str
    model_type: ModelType
    optimizations: List[OptimizationType]
    batch_size: int = 32
    max_batch_size: int = 256
    cache_predictions: bool = True
    use_gpu: bool = False
    precision: str = "float32"  # float32, float16, int8
    target_latency_ms: float = 100.0
    target_memory_mb: float = 512.0


@dataclass
class OptimizationResult:
    """Optimization result"""
    original_size_mb: float
    optimized_size_mb: float
    original_latency_ms: float
    optimized_latency_ms: float
    accuracy_loss: float
    optimization_type: OptimizationType
    success: bool
    error_message: Optional[str] = None


class ModelOptimizer:
    """Advanced model optimization for inference"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.original_model = None
        self.optimized_model = None
        self.optimization_results = {}
        
        # Performance tracking
        self.benchmark_results = {}
        self.memory_usage = {}
        
        # Caching
        self.prediction_cache = diskcache.Cache('model_cache')
        
    def load_model(self) -> Any:
        """Load model based on type"""
        start_time = time.time()
        
        try:
            if self.config.model_type == ModelType.SKLEARN:
                model = self._load_sklearn_model()
            elif self.config.model_type == ModelType.TENSORFLOW:
                model = self._load_tensorflow_model()
            elif self.config.model_type == ModelType.PYTORCH:
                model = self._load_pytorch_model()
            elif self.config.model_type == ModelType.ONNX:
                model = self._load_onnx_model()
            elif self.config.model_type == ModelType.LIGHTGBM:
                model = self._load_lightgbm_model()
            elif self.config.model_type == ModelType.XGBOOST:
                model = self._load_xgboost_model()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            self.original_model = model
            
            # Record loading time
            load_time = time.time() - start_time
            MODEL_LOAD_TIME.labels(
                model_type=self.config.model_type.value,
                optimization='none'
            ).observe(load_time)
            
            # Record memory usage
            memory_usage = self._get_model_memory_usage(model)
            MODEL_MEMORY_USAGE.labels(model_type=self.config.model_type.value).set(memory_usage)
            
            logger.info(f"Model loaded successfully in {load_time:.3f}s, memory: {memory_usage/1024/1024:.1f}MB")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def optimize_model(self) -> Any:
        """Apply optimizations to the model"""
        if self.original_model is None:
            raise ValueError("Model not loaded")
        
        optimized_model = self.original_model
        
        for optimization in self.config.optimizations:
            try:
                logger.info(f"Applying optimization: {optimization.value}")
                
                if optimization == OptimizationType.QUANTIZATION:
                    optimized_model = self._apply_quantization(optimized_model)
                elif optimization == OptimizationType.PRUNING:
                    optimized_model = self._apply_pruning(optimized_model)
                elif optimization == OptimizationType.TENSORRT:
                    optimized_model = self._apply_tensorrt(optimized_model)
                elif optimization == OptimizationType.OPENVINO:
                    optimized_model = self._apply_openvino(optimized_model)
                elif optimization == OptimizationType.ONNX_CONVERSION:
                    optimized_model = self._apply_onnx_conversion(optimized_model)
                else:
                    logger.warning(f"Optimization {optimization.value} not implemented")
                
                OPTIMIZATION_SUCCESS.labels(optimization_type=optimization.value).inc()
                
            except Exception as e:
                logger.error(f"Optimization {optimization.value} failed: {e}")
                OPTIMIZATION_ERRORS.labels(optimization_type=optimization.value).inc()
        
        self.optimized_model = optimized_model
        return optimized_model
    
    def _load_sklearn_model(self) -> BaseEstimator:
        """Load scikit-learn model"""
        if self.config.model_path.endswith('.joblib'):
            return joblib.load(self.config.model_path)
        else:
            return mlflow.sklearn.load_model(self.config.model_path)
    
    def _load_tensorflow_model(self) -> tf.keras.Model:
        """Load TensorFlow model"""
        if os.path.isdir(self.config.model_path):
            return tf.keras.models.load_model(self.config.model_path)
        else:
            return mlflow.tensorflow.load_model(self.config.model_path)
    
    def _load_pytorch_model(self) -> torch.nn.Module:
        """Load PyTorch model"""
        if self.config.model_path.endswith('.pth'):
            return torch.load(self.config.model_path, map_location='cpu')
        else:
            return mlflow.pytorch.load_model(self.config.model_path)
    
    def _load_onnx_model(self) -> ort.InferenceSession:
        """Load ONNX model"""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.config.use_gpu else ['CPUExecutionProvider']
        return ort.InferenceSession(self.config.model_path, providers=providers)
    
    def _load_lightgbm_model(self) -> Any:
        """Load LightGBM model"""
        import lightgbm as lgb
        return lgb.Booster(model_file=self.config.model_path)
    
    def _load_xgboost_model(self) -> Any:
        """Load XGBoost model"""
        import xgboost as xgb
        return xgb.Booster(model_file=self.config.model_path)
    
    def _apply_quantization(self, model: Any) -> Any:
        """Apply quantization optimization"""
        try:
            if self.config.model_type == ModelType.TENSORFLOW:
                return self._quantize_tensorflow_model(model)
            elif self.config.model_type == ModelType.PYTORCH:
                return self._quantize_pytorch_model(model)
            else:
                logger.warning(f"Quantization not supported for {self.config.model_type}")
                return model
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model
    
    def _quantize_tensorflow_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """Quantize TensorFlow model"""
        import tensorflow_model_optimization as tfmot
        
        # Post-training quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if self.config.precision == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
        elif self.config.precision == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Create TFLite interpreter wrapper
        class TFLiteModelWrapper:
            def __init__(self, tflite_model):
                self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            
            def predict(self, x):
                if isinstance(x, pd.DataFrame):
                    x = x.values
                if x.dtype != np.float32:
                    x = x.astype(np.float32)
                
                self.interpreter.set_tensor(self.input_details[0]['index'], x)
                self.interpreter.invoke()
                return self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return TFLiteModelWrapper(tflite_model)
    
    def _quantize_pytorch_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Quantize PyTorch model"""
        # Post-training quantization
        model.eval()
        
        if self.config.precision == "int8":
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            # Convert to half precision
            quantized_model = model.half()
        
        return quantized_model
    
    def _apply_pruning(self, model: Any) -> Any:
        """Apply model pruning"""
        try:
            if self.config.model_type == ModelType.TENSORFLOW:
                return self._prune_tensorflow_model(model)
            elif self.config.model_type == ModelType.PYTORCH:
                return self._prune_pytorch_model(model)
            else:
                logger.warning(f"Pruning not supported for {self.config.model_type}")
                return model
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return model
    
    def _prune_tensorflow_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """Prune TensorFlow model"""
        import tensorflow_model_optimization as tfmot
        
        # Apply magnitude-based pruning
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.50,
                final_sparsity=0.80,
                begin_step=0,
                end_step=1000
            )
        }
        
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        
        # Compile and run a few steps to apply pruning
        model_for_pruning.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Remove pruning wrappers
        model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        
        return model_for_export
    
    def _prune_pytorch_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Prune PyTorch model"""
        import torch.nn.utils.prune as prune
        
        # Apply magnitude-based pruning to linear layers
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.3)
                prune.remove(module, 'weight')
        
        return model
    
    def _apply_tensorrt(self, model: Any) -> Any:
        """Apply TensorRT optimization"""
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT not available")
            return model
        
        try:
            if self.config.model_type == ModelType.TENSORFLOW:
                return self._tensorrt_tensorflow(model)
            elif self.config.model_type == ModelType.ONNX:
                return self._tensorrt_onnx(model)
            else:
                logger.warning(f"TensorRT optimization not supported for {self.config.model_type}")
                return model
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            return model
    
    def _tensorrt_tensorflow(self, model: tf.keras.Model) -> Any:
        """Optimize TensorFlow model with TensorRT"""
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
        
        # Convert to TensorRT
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=None,
            input_saved_model_tags=['serve'],
            input_saved_model_signature_key='serving_default',
            precision_mode=trt.TrtPrecisionMode.FP16 if self.config.precision == "float16" else trt.TrtPrecisionMode.FP32,
            maximum_cached_engines=100
        )
        
        # Convert and save
        temp_dir = "/tmp/tensorrt_model"
        os.makedirs(temp_dir, exist_ok=True)
        
        tf.saved_model.save(model, temp_dir)
        converter.input_saved_model_dir = temp_dir
        
        converter.convert()
        converter.save(temp_dir + "_optimized")
        
        # Load optimized model
        return tf.saved_model.load(temp_dir + "_optimized")
    
    def _apply_openvino(self, model: Any) -> Any:
        """Apply OpenVINO optimization"""
        if not OPENVINO_AVAILABLE:
            logger.warning("OpenVINO not available")
            return model
        
        try:
            # Convert model to OpenVINO IR format
            if self.config.model_type == ModelType.TENSORFLOW:
                return self._openvino_tensorflow(model)
            elif self.config.model_type == ModelType.ONNX:
                return self._openvino_onnx(model)
            else:
                logger.warning(f"OpenVINO optimization not supported for {self.config.model_type}")
                return model
        except Exception as e:
            logger.error(f"OpenVINO optimization failed: {e}")
            return model
    
    def _openvino_tensorflow(self, model: tf.keras.Model) -> Any:
        """Optimize TensorFlow model with OpenVINO"""
        from openvino.tools import mo
        
        # Save model temporarily
        temp_model_path = "/tmp/temp_tf_model"
        tf.saved_model.save(model, temp_model_path)
        
        # Convert to OpenVINO IR
        ir_model = mo.convert_model(temp_model_path)
        
        # Create OpenVINO runtime
        core = ov.Core()
        compiled_model = core.compile_model(ir_model, "CPU")
        
        # Create wrapper
        class OpenVINOWrapper:
            def __init__(self, compiled_model):
                self.compiled_model = compiled_model
                self.infer_request = compiled_model.create_infer_request()
            
            def predict(self, x):
                if isinstance(x, pd.DataFrame):
                    x = x.values
                if x.dtype != np.float32:
                    x = x.astype(np.float32)
                
                self.infer_request.infer([x])
                return self.infer_request.get_output_tensor().data
        
        return OpenVINOWrapper(compiled_model)
    
    def _apply_onnx_conversion(self, model: Any) -> ort.InferenceSession:
        """Convert model to ONNX format"""
        try:
            if self.config.model_type == ModelType.TENSORFLOW:
                return self._tensorflow_to_onnx(model)
            elif self.config.model_type == ModelType.PYTORCH:
                return self._pytorch_to_onnx(model)
            elif self.config.model_type == ModelType.SKLEARN:
                return self._sklearn_to_onnx(model)
            else:
                logger.warning(f"ONNX conversion not supported for {self.config.model_type}")
                return model
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            return model
    
    def _tensorflow_to_onnx(self, model: tf.keras.Model) -> ort.InferenceSession:
        """Convert TensorFlow model to ONNX"""
        import tf2onnx
        
        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
        
        # Save temporarily
        temp_path = "/tmp/temp_model.onnx"
        with open(temp_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.config.use_gpu else ['CPUExecutionProvider']
        return ort.InferenceSession(temp_path, providers=providers)
    
    def _pytorch_to_onnx(self, model: torch.nn.Module) -> ort.InferenceSession:
        """Convert PyTorch model to ONNX"""
        # Create dummy input
        dummy_input = torch.randn(1, 10)  # Adjust based on your model
        
        # Export to ONNX
        temp_path = "/tmp/temp_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            temp_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True
        )
        
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.config.use_gpu else ['CPUExecutionProvider']
        return ort.InferenceSession(temp_path, providers=providers)
    
    def _sklearn_to_onnx(self, model: BaseEstimator) -> ort.InferenceSession:
        """Convert scikit-learn model to ONNX"""
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        # Define input type
        initial_type = [('float_input', FloatTensorType([None, 10]))]  # Adjust based on features
        
        # Convert to ONNX
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        # Save temporarily
        temp_path = "/tmp/temp_model.onnx"
        with open(temp_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.config.use_gpu else ['CPUExecutionProvider']
        return ort.InferenceSession(temp_path, providers=providers)
    
    def benchmark_model(self, test_data: pd.DataFrame, num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark model performance"""
        results = {}
        
        for model_name, model in [("original", self.original_model), ("optimized", self.optimized_model)]:
            if model is None:
                continue
            
            latencies = []
            memory_usage_before = psutil.Process().memory_info().rss
            
            for _ in range(num_iterations):
                start_time = time.time()
                
                # Make prediction
                if hasattr(model, 'predict'):
                    _ = model.predict(test_data)
                elif hasattr(model, 'run'):  # ONNX
                    input_name = model.get_inputs()[0].name
                    _ = model.run(None, {input_name: test_data.values.astype(np.float32)})
                
                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            
            memory_usage_after = psutil.Process().memory_info().rss
            memory_usage = memory_usage_after - memory_usage_before
            
            results[model_name] = {
                'avg_latency_ms': np.mean(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'memory_usage_mb': memory_usage / 1024 / 1024,
                'throughput_qps': 1000 / np.mean(latencies)
            }
            
            # Record metrics
            INFERENCE_LATENCY.labels(
                model_type=self.config.model_type.value,
                batch_size=len(test_data)
            ).observe(np.mean(latencies) / 1000)
        
        self.benchmark_results = results
        return results
    
    def _get_model_memory_usage(self, model: Any) -> int:
        """Get model memory usage in bytes"""
        try:
            if hasattr(model, '__sizeof__'):
                return model.__sizeof__()
            elif hasattr(model, 'get_size'):
                return model.get_size()
            else:
                # Estimate using pickle size
                return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            return 0
    
    def save_optimized_model(self, output_path: str):
        """Save optimized model"""
        if self.optimized_model is None:
            raise ValueError("No optimized model to save")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save based on model type
        if isinstance(self.optimized_model, ort.InferenceSession):
            # ONNX model - copy the file
            import shutil
            shutil.copy(self.optimized_model._model_path, output_path)
        elif hasattr(self.optimized_model, 'save'):
            # TensorFlow model
            self.optimized_model.save(output_path)
        else:
            # Pickle-based models
            joblib.dump(self.optimized_model, output_path)
        
        # Save optimization metadata
        metadata = {
            'original_config': self.config.__dict__,
            'optimization_results': self.optimization_results,
            'benchmark_results': self.benchmark_results,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = output_path.replace('.joblib', '_metadata.json').replace('.onnx', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Optimized model saved to {output_path}")
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        report = {
            'model_config': self.config.__dict__,
            'optimization_results': self.optimization_results,
            'benchmark_results': self.benchmark_results,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if self.benchmark_results:
            original = self.benchmark_results.get('original', {})
            optimized = self.benchmark_results.get('optimized', {})
            
            if original and optimized:
                latency_improvement = (original['avg_latency_ms'] - optimized['avg_latency_ms']) / original['avg_latency_ms']
                memory_reduction = (original['memory_usage_mb'] - optimized['memory_usage_mb']) / original['memory_usage_mb']
                
                if latency_improvement > 0.2:
                    recommendations.append(f"Significant latency improvement: {latency_improvement:.1%}")
                
                if memory_reduction > 0.3:
                    recommendations.append(f"Significant memory reduction: {memory_reduction:.1%}")
                
                if optimized['avg_latency_ms'] > self.config.target_latency_ms:
                    recommendations.append("Consider more aggressive optimizations to meet latency target")
                
                if optimized['memory_usage_mb'] > self.config.target_memory_mb:
                    recommendations.append("Consider model compression to meet memory target")
        
        return recommendations


class ModelLoader:
    """Optimized model loading with caching and warm-up"""
    
    def __init__(self, cache_dir: str = "/tmp/model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models = {}
        self.model_metadata = {}
        
    def load_model_optimized(self, model_config: ModelConfig) -> Any:
        """Load model with optimization caching"""
        
        # Generate cache key
        cache_key = self._generate_cache_key(model_config)
        
        # Check if optimized model is cached
        cached_model_path = self.cache_dir / f"{cache_key}.optimized"
        
        if cached_model_path.exists():
            logger.info(f"Loading cached optimized model: {cache_key}")
            return self._load_cached_model(cached_model_path, model_config)
        
        # Load and optimize model
        logger.info(f"Optimizing model: {model_config.model_path}")
        optimizer = ModelOptimizer(model_config)
        
        # Load original model
        optimizer.load_model()
        
        # Apply optimizations
        optimized_model = optimizer.optimize_model()
        
        # Save optimized model to cache
        optimizer.save_optimized_model(str(cached_model_path))
        
        # Store in memory cache
        self.loaded_models[cache_key] = optimized_model
        self.model_metadata[cache_key] = {
            'config': model_config,
            'optimizer': optimizer,
            'load_time': datetime.now()
        }
        
        return optimized_model
    
    def _generate_cache_key(self, config: ModelConfig) -> str:
        """Generate cache key for model configuration"""
        config_str = json.dumps(config.__dict__, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _load_cached_model(self, cached_path: Path, config: ModelConfig) -> Any:
        """Load cached optimized model"""
        cache_key = self._generate_cache_key(config)
        
        # Check memory cache first
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # Load from disk cache
        if config.model_type == ModelType.ONNX or str(cached_path).endswith('.onnx'):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if config.use_gpu else ['CPUExecutionProvider']
            model = ort.InferenceSession(str(cached_path), providers=providers)
        else:
            model = joblib.load(cached_path)
        
        # Store in memory cache
        self.loaded_models[cache_key] = model
        
        return model
    
    def warm_up_model(self, model: Any, config: ModelConfig, num_warmup: int = 10):
        """Warm up model with dummy predictions"""
        logger.info(f"Warming up model with {num_warmup} iterations")
        
        # Create dummy data
        dummy_data = pd.DataFrame(np.random.randn(config.batch_size, 10))  # Adjust based on features
        
        for _ in range(num_warmup):
            try:
                if hasattr(model, 'predict'):
                    _ = model.predict(dummy_data)
                elif hasattr(model, 'run'):  # ONNX
                    input_name = model.get_inputs()[0].name
                    _ = model.run(None, {input_name: dummy_data.values.astype(np.float32)})
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")
                break
        
        logger.info("Model warmup completed")
    
    def get_model_info(self, cache_key: str) -> Dict[str, Any]:
        """Get information about loaded model"""
        if cache_key not in self.model_metadata:
            return {}
        
        metadata = self.model_metadata[cache_key]
        return {
            'model_config': metadata['config'].__dict__,
            'load_time': metadata['load_time'].isoformat(),
            'optimization_results': getattr(metadata['optimizer'], 'optimization_results', {}),
            'benchmark_results': getattr(metadata['optimizer'], 'benchmark_results', {})
        }
    
    def cleanup_cache(self, max_age_days: int = 7):
        """Cleanup old cached models"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        for cache_file in self.cache_dir.glob("*.optimized"):
            if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_time:
                cache_file.unlink()
                # Also remove metadata file
                metadata_file = cache_file.with_suffix('.json')
                if metadata_file.exists():
                    metadata_file.unlink()
                
                logger.info(f"Removed old cached model: {cache_file}")


# Factory functions
def create_model_config(
    model_path: str,
    model_type: str,
    optimizations: List[str] = None,
    **kwargs
) -> ModelConfig:
    """Factory function to create model configuration"""
    
    return ModelConfig(
        model_path=model_path,
        model_type=ModelType(model_type),
        optimizations=[OptimizationType(opt) for opt in (optimizations or [])],
        **kwargs
    )


def optimize_fraud_model(
    model_path: str,
    model_type: str = "sklearn",
    output_path: str = None,
    optimizations: List[str] = None
) -> Dict[str, Any]:
    """High-level function to optimize fraud detection model"""
    
    # Default optimizations based on model type
    if optimizations is None:
        if model_type == "sklearn":
            optimizations = ["onnx_conversion"]
        elif model_type == "tensorflow":
            optimizations = ["quantization", "tensorrt"]
        elif model_type == "pytorch":
            optimizations = ["quantization", "onnx_conversion"]
        else:
            optimizations = []
    
    # Create configuration
    config = create_model_config(
        model_path=model_path,
        model_type=model_type,
        optimizations=optimizations,
        target_latency_ms=50.0,  # 50ms target for fraud detection
        target_memory_mb=256.0   # 256MB memory target
    )
    
    # Create optimizer
    optimizer = ModelOptimizer(config)
    
    # Load and optimize
    optimizer.load_model()
    optimized_model = optimizer.optimize_model()
    
    # Benchmark if test data is available
    try:
        test_data = pd.DataFrame(np.random.randn(100, 10))  # Dummy test data
        benchmark_results = optimizer.benchmark_model(test_data)
    except Exception as e:
        logger.warning(f"Benchmarking failed: {e}")
        benchmark_results = {}
    
    # Save optimized model
    if output_path:
        optimizer.save_optimized_model(output_path)
    
    # Generate report
    report = optimizer.generate_optimization_report()
    
    return report


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Example: Optimize a sklearn model
    report = optimize_fraud_model(
        model_path="models/fraud_model.joblib",
        model_type="sklearn",
        output_path="models/fraud_model_optimized.onnx",
        optimizations=["onnx_conversion"]
    )
    
    print("Optimization Report:")
    print(json.dumps(report, indent=2, default=str))
