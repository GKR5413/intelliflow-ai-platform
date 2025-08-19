"""
Comprehensive A/B testing framework for ML models with traffic splitting,
performance monitoring, and automated rollback mechanisms
"""

import os
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
import redis
import threading
import asyncio
from abc import ABC, abstractmethod

# Statistical libraries
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import statsmodels.stats.power as smp
from statsmodels.stats.proportion import proportions_ztest

# Monitoring and alerting
import psutil
import requests
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment status enumeration"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class TrafficSplitStrategy(Enum):
    """Traffic split strategy enumeration"""
    RANDOM = "random"
    USER_ID_HASH = "user_id_hash"
    GEOGRAPHIC = "geographic"
    DEVICE_TYPE = "device_type"
    TIME_BASED = "time_based"


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_id: str
    name: str
    description: str
    models: Dict[str, str]  # variant_name -> model_uri
    traffic_split: Dict[str, float]  # variant_name -> percentage
    split_strategy: TrafficSplitStrategy
    start_date: datetime
    end_date: datetime
    success_metrics: List[str]
    guardrail_metrics: List[str]
    minimum_sample_size: int
    statistical_power: float = 0.8
    significance_level: float = 0.05
    auto_rollback_enabled: bool = True
    rollback_conditions: Dict[str, Any] = None
    target_population: Dict[str, Any] = None  # Targeting criteria
    created_by: str = ""
    
    def __post_init__(self):
        if self.rollback_conditions is None:
            self.rollback_conditions = {
                'error_rate_threshold': 0.05,
                'latency_threshold_ms': 1000,
                'success_rate_threshold': 0.95
            }


@dataclass
class ExperimentResult:
    """Experiment result data"""
    variant_name: str
    metric_name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class StatisticalTest:
    """Statistical test result"""
    test_name: str
    p_value: float
    test_statistic: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    sample_size_control: int
    sample_size_treatment: int


class MetricsCollector(ABC):
    """Abstract base class for metrics collection"""
    
    @abstractmethod
    def collect_metrics(self, experiment_id: str, variant: str) -> Dict[str, float]:
        """Collect metrics for a specific variant"""
        pass


class RedisMetricsCollector(MetricsCollector):
    """Redis-based metrics collector"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
    
    def collect_metrics(self, experiment_id: str, variant: str) -> Dict[str, float]:
        """Collect metrics from Redis"""
        try:
            metrics = {}
            base_key = f"ab_test:{experiment_id}:{variant}"
            
            # Get all metric keys for this variant
            metric_keys = self.redis_client.keys(f"{base_key}:*")
            
            for key in metric_keys:
                metric_name = key.decode('utf-8').split(':')[-1]
                value = self.redis_client.get(key)
                if value:
                    metrics[metric_name] = float(value)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics from Redis: {e}")
            return {}


class TrafficSplitter:
    """Handles traffic splitting for A/B testing"""
    
    def __init__(self, redis_client: redis.Redis = None):
        self.redis_client = redis_client
    
    def assign_variant(self, 
                      experiment_config: ExperimentConfig,
                      user_context: Dict[str, Any]) -> str:
        """Assign user to experiment variant"""
        
        # Check if user already assigned
        if self.redis_client:
            user_id = user_context.get('user_id')
            if user_id:
                cached_variant = self.redis_client.get(f"ab_assignment:{experiment_config.experiment_id}:{user_id}")
                if cached_variant:
                    return cached_variant.decode('utf-8')
        
        # Apply targeting criteria if specified
        if not self._matches_target_population(experiment_config, user_context):
            return 'control'  # Default to control if doesn't match criteria
        
        # Assign variant based on strategy
        variant = self._assign_by_strategy(experiment_config, user_context)
        
        # Cache assignment
        if self.redis_client and user_context.get('user_id'):
            self.redis_client.setex(
                f"ab_assignment:{experiment_config.experiment_id}:{user_context['user_id']}",
                timedelta(days=30),
                variant
            )
        
        return variant
    
    def _matches_target_population(self, 
                                 experiment_config: ExperimentConfig,
                                 user_context: Dict[str, Any]) -> bool:
        """Check if user matches target population criteria"""
        if not experiment_config.target_population:
            return True
        
        criteria = experiment_config.target_population
        
        # Geographic targeting
        if 'countries' in criteria:
            user_country = user_context.get('country')
            if user_country not in criteria['countries']:
                return False
        
        # Device type targeting
        if 'device_types' in criteria:
            device_type = user_context.get('device_type')
            if device_type not in criteria['device_types']:
                return False
        
        # User segment targeting
        if 'user_segments' in criteria:
            user_segment = user_context.get('user_segment')
            if user_segment not in criteria['user_segments']:
                return False
        
        # Age range targeting
        if 'age_range' in criteria:
            user_age = user_context.get('age')
            if user_age is not None:
                min_age, max_age = criteria['age_range']
                if not (min_age <= user_age <= max_age):
                    return False
        
        return True
    
    def _assign_by_strategy(self, 
                           experiment_config: ExperimentConfig,
                           user_context: Dict[str, Any]) -> str:
        """Assign variant based on split strategy"""
        
        if experiment_config.split_strategy == TrafficSplitStrategy.RANDOM:
            return self._random_assignment(experiment_config.traffic_split)
        
        elif experiment_config.split_strategy == TrafficSplitStrategy.USER_ID_HASH:
            user_id = user_context.get('user_id', 'anonymous')
            return self._hash_based_assignment(str(user_id), experiment_config.traffic_split)
        
        elif experiment_config.split_strategy == TrafficSplitStrategy.GEOGRAPHIC:
            country = user_context.get('country', 'unknown')
            return self._hash_based_assignment(country, experiment_config.traffic_split)
        
        elif experiment_config.split_strategy == TrafficSplitStrategy.DEVICE_TYPE:
            device_type = user_context.get('device_type', 'unknown')
            return self._hash_based_assignment(device_type, experiment_config.traffic_split)
        
        elif experiment_config.split_strategy == TrafficSplitStrategy.TIME_BASED:
            return self._time_based_assignment(experiment_config.traffic_split)
        
        else:
            return self._random_assignment(experiment_config.traffic_split)
    
    def _random_assignment(self, traffic_split: Dict[str, float]) -> str:
        """Random variant assignment"""
        rand_val = np.random.random()
        cumulative = 0.0
        
        for variant, percentage in traffic_split.items():
            cumulative += percentage
            if rand_val <= cumulative:
                return variant
        
        # Fallback to first variant
        return list(traffic_split.keys())[0]
    
    def _hash_based_assignment(self, key: str, traffic_split: Dict[str, float]) -> str:
        """Hash-based deterministic assignment"""
        import hashlib
        
        # Create hash of the key
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100000
        percentage = hash_val / 100000.0
        
        cumulative = 0.0
        for variant, split_percentage in traffic_split.items():
            cumulative += split_percentage
            if percentage <= cumulative:
                return variant
        
        return list(traffic_split.keys())[0]
    
    def _time_based_assignment(self, traffic_split: Dict[str, float]) -> str:
        """Time-based assignment (hour of day)"""
        current_hour = datetime.now().hour
        
        # Use hour to determine assignment
        hour_percentage = (current_hour % 24) / 24.0
        
        cumulative = 0.0
        for variant, percentage in traffic_split.items():
            cumulative += percentage
            if hour_percentage <= cumulative:
                return variant
        
        return list(traffic_split.keys())[0]


class StatisticalAnalyzer:
    """Performs statistical analysis for A/B tests"""
    
    def __init__(self):
        self.significance_level = 0.05
    
    def analyze_experiment(self, 
                          experiment_data: Dict[str, List[ExperimentResult]],
                          config: ExperimentConfig) -> Dict[str, List[StatisticalTest]]:
        """Perform comprehensive statistical analysis"""
        
        results = {}
        
        # Get control variant (assume first variant is control)
        control_variant = list(experiment_data.keys())[0]
        
        for metric_name in config.success_metrics:
            results[metric_name] = []
            
            # Get control data
            control_data = self._extract_metric_values(
                experiment_data[control_variant], metric_name
            )
            
            # Compare with each treatment variant
            for variant_name, variant_results in experiment_data.items():
                if variant_name == control_variant:
                    continue
                
                treatment_data = self._extract_metric_values(variant_results, metric_name)
                
                if len(control_data) == 0 or len(treatment_data) == 0:
                    continue
                
                # Perform appropriate statistical test
                test_result = self._perform_statistical_test(
                    control_data, treatment_data, metric_name, 
                    control_variant, variant_name
                )
                
                results[metric_name].append(test_result)
        
        return results
    
    def _extract_metric_values(self, 
                              experiment_results: List[ExperimentResult],
                              metric_name: str) -> List[float]:
        """Extract values for a specific metric"""
        return [result.value for result in experiment_results 
                if result.metric_name == metric_name]
    
    def _perform_statistical_test(self, 
                                 control_data: List[float],
                                 treatment_data: List[float],
                                 metric_name: str,
                                 control_variant: str,
                                 treatment_variant: str) -> StatisticalTest:
        """Perform appropriate statistical test based on data type"""
        
        control_array = np.array(control_data)
        treatment_array = np.array(treatment_data)
        
        # Determine test type based on metric name or data characteristics
        if 'rate' in metric_name.lower() or 'conversion' in metric_name.lower():
            # Proportion test for rates/conversions
            return self._proportion_test(control_array, treatment_array, 
                                       control_variant, treatment_variant)
        
        elif self._is_continuous_data(control_array, treatment_array):
            # T-test for continuous data
            return self._t_test(control_array, treatment_array,
                               control_variant, treatment_variant)
        
        else:
            # Mann-Whitney U test for non-parametric data
            return self._mann_whitney_test(control_array, treatment_array,
                                         control_variant, treatment_variant)
    
    def _is_continuous_data(self, control_data: np.ndarray, treatment_data: np.ndarray) -> bool:
        """Check if data appears to be continuous"""
        # Simple heuristic: if we have many unique values, treat as continuous
        control_unique = len(np.unique(control_data))
        treatment_unique = len(np.unique(treatment_data))
        
        return (control_unique > 10 and treatment_unique > 10 and
                control_unique > len(control_data) * 0.5 and
                treatment_unique > len(treatment_data) * 0.5)
    
    def _proportion_test(self, 
                        control_data: np.ndarray,
                        treatment_data: np.ndarray,
                        control_variant: str,
                        treatment_variant: str) -> StatisticalTest:
        """Perform proportion test"""
        
        # Assume data is binary (0/1) for conversions
        control_successes = np.sum(control_data)
        treatment_successes = np.sum(treatment_data)
        
        control_n = len(control_data)
        treatment_n = len(treatment_data)
        
        # Perform z-test for proportions
        counts = np.array([control_successes, treatment_successes])
        nobs = np.array([control_n, treatment_n])
        
        z_stat, p_value = proportions_ztest(counts, nobs)
        
        # Calculate effect size (difference in proportions)
        control_rate = control_successes / control_n if control_n > 0 else 0
        treatment_rate = treatment_successes / treatment_n if treatment_n > 0 else 0
        effect_size = treatment_rate - control_rate
        
        # Calculate confidence interval
        se = np.sqrt((control_rate * (1 - control_rate) / control_n) + 
                    (treatment_rate * (1 - treatment_rate) / treatment_n))
        margin = stats.norm.ppf(1 - self.significance_level/2) * se
        ci = (effect_size - margin, effect_size + margin)
        
        return StatisticalTest(
            test_name="Proportion Z-Test",
            p_value=p_value,
            test_statistic=z_stat,
            effect_size=effect_size,
            confidence_interval=ci,
            is_significant=p_value < self.significance_level,
            sample_size_control=control_n,
            sample_size_treatment=treatment_n
        )
    
    def _t_test(self, 
               control_data: np.ndarray,
               treatment_data: np.ndarray,
               control_variant: str,
               treatment_variant: str) -> StatisticalTest:
        """Perform independent t-test"""
        
        t_stat, p_value = ttest_ind(control_data, treatment_data)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_data) - 1) * np.var(control_data, ddof=1) +
                             (len(treatment_data) - 1) * np.var(treatment_data, ddof=1)) /
                            (len(control_data) + len(treatment_data) - 2))
        
        effect_size = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std
        
        # Calculate confidence interval for mean difference
        se = pooled_std * np.sqrt(1/len(control_data) + 1/len(treatment_data))
        df = len(control_data) + len(treatment_data) - 2
        t_critical = stats.t.ppf(1 - self.significance_level/2, df)
        margin = t_critical * se
        
        mean_diff = np.mean(treatment_data) - np.mean(control_data)
        ci = (mean_diff - margin, mean_diff + margin)
        
        return StatisticalTest(
            test_name="Independent T-Test",
            p_value=p_value,
            test_statistic=t_stat,
            effect_size=effect_size,
            confidence_interval=ci,
            is_significant=p_value < self.significance_level,
            sample_size_control=len(control_data),
            sample_size_treatment=len(treatment_data)
        )
    
    def _mann_whitney_test(self, 
                          control_data: np.ndarray,
                          treatment_data: np.ndarray,
                          control_variant: str,
                          treatment_variant: str) -> StatisticalTest:
        """Perform Mann-Whitney U test"""
        
        u_stat, p_value = mannwhitneyu(control_data, treatment_data, alternative='two-sided')
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(control_data), len(treatment_data)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        
        # For non-parametric tests, confidence intervals are more complex
        # Using a simplified approach
        ci = (effect_size - 0.1, effect_size + 0.1)  # Placeholder
        
        return StatisticalTest(
            test_name="Mann-Whitney U Test",
            p_value=p_value,
            test_statistic=u_stat,
            effect_size=effect_size,
            confidence_interval=ci,
            is_significant=p_value < self.significance_level,
            sample_size_control=n1,
            sample_size_treatment=n2
        )
    
    def calculate_sample_size(self, 
                             effect_size: float,
                             power: float = 0.8,
                             alpha: float = 0.05) -> int:
        """Calculate required sample size for desired power"""
        try:
            # Using power analysis for two-sample t-test
            sample_size = smp.ttest_power(effect_size, power, alpha, alternative='two-sided')
            return int(np.ceil(sample_size))
        except:
            # Fallback calculation
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            sample_size = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            return int(np.ceil(sample_size))


class ABTestMonitor:
    """Monitors A/B test performance and triggers alerts"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.alerts = []
        
        # Prometheus metrics
        self.request_counter = Counter('ab_test_requests_total', 
                                     'Total requests', ['experiment_id', 'variant'])
        self.error_counter = Counter('ab_test_errors_total',
                                   'Total errors', ['experiment_id', 'variant'])
        self.latency_histogram = Histogram('ab_test_latency_seconds',
                                         'Request latency', ['experiment_id', 'variant'])
        self.conversion_gauge = Gauge('ab_test_conversion_rate',
                                    'Conversion rate', ['experiment_id', 'variant'])
    
    def record_request(self, experiment_id: str, variant: str, latency: float):
        """Record a request"""
        self.request_counter.labels(experiment_id=experiment_id, variant=variant).inc()
        self.latency_histogram.labels(experiment_id=experiment_id, variant=variant).observe(latency)
    
    def record_error(self, experiment_id: str, variant: str):
        """Record an error"""
        self.error_counter.labels(experiment_id=experiment_id, variant=variant).inc()
    
    def record_conversion(self, experiment_id: str, variant: str, rate: float):
        """Record conversion rate"""
        self.conversion_gauge.labels(experiment_id=experiment_id, variant=variant).set(rate)
    
    def check_guardrails(self, 
                        experiment_config: ExperimentConfig,
                        current_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Check guardrail metrics and return violations"""
        violations = []
        rollback_conditions = experiment_config.rollback_conditions
        
        for variant, metrics in current_metrics.items():
            if variant == 'control':  # Skip control variant
                continue
            
            # Check error rate
            error_rate = metrics.get('error_rate', 0)
            if error_rate > rollback_conditions.get('error_rate_threshold', 0.05):
                violations.append(f"High error rate in {variant}: {error_rate:.3f}")
            
            # Check latency
            avg_latency = metrics.get('avg_latency_ms', 0)
            if avg_latency > rollback_conditions.get('latency_threshold_ms', 1000):
                violations.append(f"High latency in {variant}: {avg_latency:.1f}ms")
            
            # Check success rate
            success_rate = metrics.get('success_rate', 1.0)
            if success_rate < rollback_conditions.get('success_rate_threshold', 0.95):
                violations.append(f"Low success rate in {variant}: {success_rate:.3f}")
        
        return violations
    
    def should_trigger_rollback(self, 
                               experiment_config: ExperimentConfig,
                               violations: List[str]) -> bool:
        """Determine if rollback should be triggered"""
        if not experiment_config.auto_rollback_enabled:
            return False
        
        if len(violations) == 0:
            return False
        
        # Check severity of violations
        critical_violations = [v for v in violations if 'error_rate' in v or 'success_rate' in v]
        
        return len(critical_violations) > 0


class ABTestFramework:
    """Main A/B testing framework"""
    
    def __init__(self, 
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 0):
        
        # Initialize components
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.traffic_splitter = TrafficSplitter(self.redis_client)
        self.metrics_collector = RedisMetricsCollector(self.redis_client)
        self.statistical_analyzer = StatisticalAnalyzer()
        self.monitor = ABTestMonitor(self.redis_client)
        
        # Active experiments
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        
        # Background monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Start Prometheus metrics server
        start_http_server(9090)
        
        logger.info("A/B Test Framework initialized")
    
    def create_experiment(self, config: ExperimentConfig) -> bool:
        """Create a new A/B test experiment"""
        try:
            # Validate configuration
            if not self._validate_experiment_config(config):
                return False
            
            # Store experiment configuration
            config_json = json.dumps(asdict(config), default=str)
            self.redis_client.set(f"ab_experiment:{config.experiment_id}", config_json)
            
            # Set experiment status
            self.redis_client.set(f"ab_experiment_status:{config.experiment_id}", 
                                 ExperimentStatus.DRAFT.value)
            
            logger.info(f"Created experiment: {config.experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            return False
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B test experiment"""
        try:
            # Load experiment configuration
            config = self._load_experiment_config(experiment_id)
            if not config:
                return False
            
            # Check if experiment is ready to start
            if datetime.now() < config.start_date:
                logger.warning(f"Experiment {experiment_id} start date is in the future")
                return False
            
            # Update status
            self.redis_client.set(f"ab_experiment_status:{experiment_id}",
                                 ExperimentStatus.RUNNING.value)
            
            # Add to active experiments
            self.active_experiments[experiment_id] = config
            
            # Start monitoring if not already running
            if not self.monitoring_active:
                self.start_monitoring()
            
            logger.info(f"Started experiment: {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start experiment: {e}")
            return False
    
    def stop_experiment(self, experiment_id: str, reason: str = "Manual stop") -> bool:
        """Stop an A/B test experiment"""
        try:
            # Update status
            self.redis_client.set(f"ab_experiment_status:{experiment_id}",
                                 ExperimentStatus.STOPPED.value)
            
            # Remove from active experiments
            self.active_experiments.pop(experiment_id, None)
            
            # Log stop reason
            self.redis_client.set(f"ab_experiment_stop_reason:{experiment_id}", reason)
            
            logger.info(f"Stopped experiment: {experiment_id}, reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop experiment: {e}")
            return False
    
    def assign_user_to_variant(self, 
                              experiment_id: str,
                              user_context: Dict[str, Any]) -> Optional[str]:
        """Assign user to experiment variant"""
        try:
            # Check if experiment is active
            if experiment_id not in self.active_experiments:
                return None
            
            config = self.active_experiments[experiment_id]
            
            # Check experiment timing
            now = datetime.now()
            if now < config.start_date or now > config.end_date:
                return None
            
            # Assign variant
            variant = self.traffic_splitter.assign_variant(config, user_context)
            
            # Record assignment
            self._record_assignment(experiment_id, variant, user_context)
            
            return variant
            
        except Exception as e:
            logger.error(f"Failed to assign user to variant: {e}")
            return None
    
    def record_metric(self, 
                     experiment_id: str,
                     variant: str,
                     metric_name: str,
                     value: float,
                     metadata: Dict[str, Any] = None) -> bool:
        """Record a metric value for an experiment variant"""
        try:
            # Create experiment result
            result = ExperimentResult(
                variant_name=variant,
                metric_name=metric_name,
                value=value,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
            # Store in Redis
            key = f"ab_metrics:{experiment_id}:{variant}:{metric_name}"
            self.redis_client.lpush(key, json.dumps(asdict(result), default=str))
            
            # Set expiration (keep for 90 days)
            self.redis_client.expire(key, timedelta(days=90))
            
            # Update Prometheus metrics
            if metric_name == 'conversion_rate':
                self.monitor.record_conversion(experiment_id, variant, value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
            return False
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment results"""
        try:
            config = self._load_experiment_config(experiment_id)
            if not config:
                return {}
            
            # Collect all metrics data
            experiment_data = {}
            
            for variant in config.traffic_split.keys():
                experiment_data[variant] = []
                
                for metric_name in config.success_metrics + config.guardrail_metrics:
                    key = f"ab_metrics:{experiment_id}:{variant}:{metric_name}"
                    metric_data = self.redis_client.lrange(key, 0, -1)
                    
                    for data_json in metric_data:
                        result_dict = json.loads(data_json)
                        result_dict['timestamp'] = datetime.fromisoformat(result_dict['timestamp'])
                        result = ExperimentResult(**result_dict)
                        experiment_data[variant].append(result)
            
            # Perform statistical analysis
            statistical_results = self.statistical_analyzer.analyze_experiment(
                experiment_data, config
            )
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(experiment_data, config)
            
            # Check experiment status
            status = self._get_experiment_status(experiment_id)
            
            return {
                'experiment_id': experiment_id,
                'config': asdict(config),
                'status': status,
                'summary_statistics': summary_stats,
                'statistical_tests': statistical_results,
                'total_samples': sum(len(data) for data in experiment_data.values()),
                'data_collection_period': {
                    'start': config.start_date.isoformat(),
                    'end': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get experiment results: {e}")
            return {}
    
    def start_monitoring(self):
        """Start background monitoring of active experiments"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Started experiment monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("Stopped experiment monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                for experiment_id, config in self.active_experiments.items():
                    # Check if experiment should end
                    if datetime.now() > config.end_date:
                        self.stop_experiment(experiment_id, "Reached end date")
                        continue
                    
                    # Collect current metrics
                    current_metrics = self._collect_current_metrics(experiment_id, config)
                    
                    # Check guardrails
                    violations = self.monitor.check_guardrails(config, current_metrics)
                    
                    # Trigger rollback if necessary
                    if self.monitor.should_trigger_rollback(config, violations):
                        self.stop_experiment(experiment_id, f"Guardrail violations: {violations}")
                        # Here you would also trigger actual model rollback
                        self._trigger_model_rollback(experiment_id, config)
                
                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _validate_experiment_config(self, config: ExperimentConfig) -> bool:
        """Validate experiment configuration"""
        
        # Check traffic split sums to 1.0
        total_traffic = sum(config.traffic_split.values())
        if not (0.99 <= total_traffic <= 1.01):  # Allow small floating point errors
            logger.error(f"Traffic split must sum to 1.0, got {total_traffic}")
            return False
        
        # Check dates
        if config.start_date >= config.end_date:
            logger.error("Start date must be before end date")
            return False
        
        # Check minimum sample size
        if config.minimum_sample_size < 100:
            logger.warning("Minimum sample size is very low, consider increasing")
        
        # Check models exist
        for variant, model_uri in config.models.items():
            if not model_uri:
                logger.error(f"Model URI missing for variant: {variant}")
                return False
        
        return True
    
    def _load_experiment_config(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Load experiment configuration from Redis"""
        try:
            config_json = self.redis_client.get(f"ab_experiment:{experiment_id}")
            if not config_json:
                return None
            
            config_dict = json.loads(config_json)
            
            # Convert string dates back to datetime
            config_dict['start_date'] = datetime.fromisoformat(config_dict['start_date'])
            config_dict['end_date'] = datetime.fromisoformat(config_dict['end_date'])
            config_dict['split_strategy'] = TrafficSplitStrategy(config_dict['split_strategy'])
            
            return ExperimentConfig(**config_dict)
            
        except Exception as e:
            logger.error(f"Failed to load experiment config: {e}")
            return None
    
    def _record_assignment(self, 
                          experiment_id: str,
                          variant: str,
                          user_context: Dict[str, Any]):
        """Record user assignment to variant"""
        assignment_data = {
            'experiment_id': experiment_id,
            'variant': variant,
            'user_context': user_context,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store assignment
        key = f"ab_assignments:{experiment_id}:{variant}"
        self.redis_client.lpush(key, json.dumps(assignment_data))
        self.redis_client.expire(key, timedelta(days=90))
    
    def _collect_current_metrics(self, 
                                experiment_id: str,
                                config: ExperimentConfig) -> Dict[str, Dict[str, float]]:
        """Collect current metrics for all variants"""
        current_metrics = {}
        
        for variant in config.traffic_split.keys():
            current_metrics[variant] = self.metrics_collector.collect_metrics(
                experiment_id, variant
            )
        
        return current_metrics
    
    def _calculate_summary_stats(self, 
                                experiment_data: Dict[str, List[ExperimentResult]],
                                config: ExperimentConfig) -> Dict[str, Dict[str, Any]]:
        """Calculate summary statistics for each variant"""
        summary_stats = {}
        
        for variant, results in experiment_data.items():
            variant_stats = {}
            
            for metric_name in config.success_metrics:
                metric_values = [r.value for r in results if r.metric_name == metric_name]
                
                if metric_values:
                    variant_stats[metric_name] = {
                        'mean': np.mean(metric_values),
                        'std': np.std(metric_values),
                        'min': np.min(metric_values),
                        'max': np.max(metric_values),
                        'count': len(metric_values),
                        'median': np.median(metric_values)
                    }
                else:
                    variant_stats[metric_name] = {
                        'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0, 'median': 0
                    }
            
            summary_stats[variant] = variant_stats
        
        return summary_stats
    
    def _get_experiment_status(self, experiment_id: str) -> str:
        """Get current experiment status"""
        status = self.redis_client.get(f"ab_experiment_status:{experiment_id}")
        return status.decode('utf-8') if status else ExperimentStatus.DRAFT.value
    
    def _trigger_model_rollback(self, experiment_id: str, config: ExperimentConfig):
        """Trigger model rollback (placeholder for actual implementation)"""
        logger.warning(f"Triggering model rollback for experiment: {experiment_id}")
        
        # In a real implementation, this would:
        # 1. Stop traffic to treatment variants
        # 2. Route all traffic to control variant
        # 3. Notify operations team
        # 4. Update model serving infrastructure
        # 5. Send alerts to stakeholders
        
        # For now, just log the action
        rollback_data = {
            'experiment_id': experiment_id,
            'action': 'rollback_triggered',
            'timestamp': datetime.now().isoformat(),
            'reason': 'guardrail_violation'
        }
        
        self.redis_client.lpush(f"ab_rollbacks:{experiment_id}", 
                               json.dumps(rollback_data))


# Example usage and testing
def create_sample_experiment() -> ExperimentConfig:
    """Create a sample experiment configuration"""
    
    return ExperimentConfig(
        experiment_id="fraud_model_experiment_001",
        name="Fraud Detection Model A/B Test",
        description="Testing new fraud detection model against baseline",
        models={
            'control': 'models:/fraud_detection_baseline/1',
            'treatment': 'models:/fraud_detection_v2/1'
        },
        traffic_split={
            'control': 0.5,
            'treatment': 0.5
        },
        split_strategy=TrafficSplitStrategy.USER_ID_HASH,
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=14),
        success_metrics=['precision', 'recall', 'f1_score', 'auc'],
        guardrail_metrics=['latency_ms', 'error_rate', 'throughput'],
        minimum_sample_size=10000,
        statistical_power=0.8,
        significance_level=0.05,
        auto_rollback_enabled=True,
        rollback_conditions={
            'error_rate_threshold': 0.05,
            'latency_threshold_ms': 500,
            'success_rate_threshold': 0.95
        },
        target_population={
            'countries': ['US', 'CA', 'UK'],
            'user_segments': ['premium', 'regular']
        },
        created_by="ml_team"
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize framework
    ab_framework = ABTestFramework()
    
    # Create sample experiment
    experiment_config = create_sample_experiment()
    
    # Create and start experiment
    success = ab_framework.create_experiment(experiment_config)
    if success:
        print(f"Created experiment: {experiment_config.experiment_id}")
        
        # Start experiment
        ab_framework.start_experiment(experiment_config.experiment_id)
        print("Experiment started")
        
        # Simulate some user assignments and metrics
        for i in range(100):
            user_context = {
                'user_id': f"user_{i}",
                'country': 'US',
                'device_type': 'mobile',
                'user_segment': 'premium'
            }
            
            variant = ab_framework.assign_user_to_variant(
                experiment_config.experiment_id, user_context
            )
            
            if variant:
                # Simulate metrics
                precision = np.random.normal(0.85 if variant == 'control' else 0.87, 0.05)
                latency = np.random.normal(200 if variant == 'control' else 180, 20)
                
                ab_framework.record_metric(experiment_config.experiment_id, 
                                          variant, 'precision', precision)
                ab_framework.record_metric(experiment_config.experiment_id,
                                          variant, 'latency_ms', latency)
        
        # Get results
        results = ab_framework.get_experiment_results(experiment_config.experiment_id)
        print(f"Experiment results collected: {len(results)} data points")
        
        print("A/B Testing Framework demo completed!")
    else:
        print("Failed to create experiment")
