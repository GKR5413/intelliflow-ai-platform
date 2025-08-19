package com.intelliflow.shared.metrics;

import io.micrometer.core.instrument.*;
import io.micrometer.core.instrument.binder.MeterBinder;
import io.micrometer.core.instrument.config.MeterFilter;
import io.micrometer.prometheus.PrometheusConfig;
import io.micrometer.prometheus.PrometheusMeterRegistry;
import org.springframework.stereotype.Component;
import org.springframework.boot.actuate.metrics.MetricsEndpoint;

import javax.annotation.PostConstruct;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

/**
 * Custom Prometheus metrics for IntelliFlow AI Platform
 * Provides business and technical metrics for monitoring and alerting
 */
@Component
public class CustomMetrics implements MeterBinder {

    private MeterRegistry meterRegistry;
    
    // Business Metrics
    private Counter transactionProcessedCounter;
    private Counter fraudDetectedCounter;
    private Counter userRegistrationCounter;
    private Counter notificationSentCounter;
    private Timer transactionProcessingTimer;
    private Timer fraudDetectionTimer;
    private Gauge activeUsersGauge;
    private Gauge fraudRateGauge;
    
    // Technical Metrics
    private Counter apiRequestCounter;
    private Counter databaseQueryCounter;
    private Counter cacheHitCounter;
    private Counter cacheMissCounter;
    private Timer databaseQueryTimer;
    private Timer cacheOperationTimer;
    
    // System Health Metrics
    private Gauge jvmMemoryUsageGauge;
    private Gauge databaseConnectionsGauge;
    private Gauge redisConnectionsGauge;
    private Counter errorCounter;
    
    // Custom gauge values
    private final AtomicLong activeUsers = new AtomicLong(0);
    private final AtomicLong databaseConnections = new AtomicLong(0);
    private final AtomicLong redisConnections = new AtomicLong(0);
    private final Map<String, AtomicLong> customGaugeValues = new ConcurrentHashMap<>();

    @Override
    public void bindTo(MeterRegistry registry) {
        this.meterRegistry = registry;
        initializeMetrics();
    }

    @PostConstruct
    public void init() {
        // This will be called after dependency injection
    }

    private void initializeMetrics() {
        initializeBusinessMetrics();
        initializeTechnicalMetrics();
        initializeSystemHealthMetrics();
        initializeCustomFilters();
    }

    private void initializeBusinessMetrics() {
        // Transaction metrics
        transactionProcessedCounter = Counter.builder("intelliflow_transactions_processed_total")
                .description("Total number of transactions processed")
                .tag("service", getServiceName())
                .register(meterRegistry);

        fraudDetectedCounter = Counter.builder("intelliflow_fraud_detected_total")
                .description("Total number of fraud cases detected")
                .tag("service", getServiceName())
                .register(meterRegistry);

        userRegistrationCounter = Counter.builder("intelliflow_user_registrations_total")
                .description("Total number of user registrations")
                .tag("service", getServiceName())
                .register(meterRegistry);

        notificationSentCounter = Counter.builder("intelliflow_notifications_sent_total")
                .description("Total number of notifications sent")
                .tag("service", getServiceName())
                .register(meterRegistry);

        transactionProcessingTimer = Timer.builder("intelliflow_transaction_processing_duration_seconds")
                .description("Time taken to process transactions")
                .tag("service", getServiceName())
                .register(meterRegistry);

        fraudDetectionTimer = Timer.builder("intelliflow_fraud_detection_duration_seconds")
                .description("Time taken for fraud detection")
                .tag("service", getServiceName())
                .register(meterRegistry);

        activeUsersGauge = Gauge.builder("intelliflow_active_users")
                .description("Number of currently active users")
                .tag("service", getServiceName())
                .register(meterRegistry, this, metrics -> metrics.activeUsers.get());

        fraudRateGauge = Gauge.builder("intelliflow_fraud_rate")
                .description("Current fraud detection rate")
                .tag("service", getServiceName())
                .register(meterRegistry, this, metrics -> calculateFraudRate());
    }

    private void initializeTechnicalMetrics() {
        // API metrics
        apiRequestCounter = Counter.builder("intelliflow_api_requests_total")
                .description("Total number of API requests")
                .tag("service", getServiceName())
                .register(meterRegistry);

        databaseQueryCounter = Counter.builder("intelliflow_database_queries_total")
                .description("Total number of database queries")
                .tag("service", getServiceName())
                .register(meterRegistry);

        cacheHitCounter = Counter.builder("intelliflow_cache_hits_total")
                .description("Total number of cache hits")
                .tag("service", getServiceName())
                .register(meterRegistry);

        cacheMissCounter = Counter.builder("intelliflow_cache_misses_total")
                .description("Total number of cache misses")
                .tag("service", getServiceName())
                .register(meterRegistry);

        databaseQueryTimer = Timer.builder("intelliflow_database_query_duration_seconds")
                .description("Time taken for database queries")
                .tag("service", getServiceName())
                .register(meterRegistry);

        cacheOperationTimer = Timer.builder("intelliflow_cache_operation_duration_seconds")
                .description("Time taken for cache operations")
                .tag("service", getServiceName())
                .register(meterRegistry);
    }

    private void initializeSystemHealthMetrics() {
        // JVM Memory
        jvmMemoryUsageGauge = Gauge.builder("intelliflow_jvm_memory_usage_bytes")
                .description("JVM memory usage in bytes")
                .tag("service", getServiceName())
                .tag("area", "heap")
                .register(meterRegistry, this, metrics -> Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());

        // Database connections
        databaseConnectionsGauge = Gauge.builder("intelliflow_database_connections_active")
                .description("Active database connections")
                .tag("service", getServiceName())
                .register(meterRegistry, this, metrics -> metrics.databaseConnections.get());

        // Redis connections
        redisConnectionsGauge = Gauge.builder("intelliflow_redis_connections_active")
                .description("Active Redis connections")
                .tag("service", getServiceName())
                .register(meterRegistry, this, metrics -> metrics.redisConnections.get());

        // Error counter
        errorCounter = Counter.builder("intelliflow_errors_total")
                .description("Total number of errors")
                .tag("service", getServiceName())
                .register(meterRegistry);
    }

    private void initializeCustomFilters() {
        // Add custom meter filters for better organization
        meterRegistry.config().meterFilter(
            MeterFilter.commonTags("application", "intelliflow-platform")
        );

        // Filter to add percentiles to timers
        meterRegistry.config().meterFilter(
            MeterFilter.maximumExpectedValue(Meter.Id.class, Duration.ofSeconds(10))
        );
    }

    // Business Metric Methods

    public void incrementTransactionProcessed(String transactionType, String status) {
        transactionProcessedCounter.increment(
            Tags.of(
                "transaction_type", transactionType,
                "status", status
            )
        );
    }

    public void incrementFraudDetected(String fraudType, double score) {
        fraudDetectedCounter.increment(
            Tags.of(
                "fraud_type", fraudType,
                "score_range", getScoreRange(score)
            )
        );
    }

    public void incrementUserRegistration(String registrationSource) {
        userRegistrationCounter.increment(
            Tags.of("source", registrationSource)
        );
    }

    public void incrementNotificationSent(String notificationType, String channel, String status) {
        notificationSentCounter.increment(
            Tags.of(
                "notification_type", notificationType,
                "channel", channel,
                "status", status
            )
        );
    }

    public Timer.Sample startTransactionProcessingTimer() {
        return Timer.start(meterRegistry);
    }

    public void stopTransactionProcessingTimer(Timer.Sample sample, String transactionType) {
        sample.stop(Timer.builder("intelliflow_transaction_processing_duration_seconds")
                .tag("service", getServiceName())
                .tag("transaction_type", transactionType)
                .register(meterRegistry));
    }

    public Timer.Sample startFraudDetectionTimer() {
        return Timer.start(meterRegistry);
    }

    public void stopFraudDetectionTimer(Timer.Sample sample, String modelType) {
        sample.stop(Timer.builder("intelliflow_fraud_detection_duration_seconds")
                .tag("service", getServiceName())
                .tag("model_type", modelType)
                .register(meterRegistry));
    }

    public void recordFraudDetectionLatency(Duration duration, String modelVersion) {
        Timer.builder("intelliflow_fraud_prediction_latency_seconds")
                .tag("service", getServiceName())
                .tag("model_version", modelVersion)
                .register(meterRegistry)
                .record(duration);
    }

    public void recordModelAccuracy(double accuracy, String modelVersion) {
        Gauge.builder("intelliflow_model_accuracy")
                .description("Current model accuracy score")
                .tag("service", getServiceName())
                .tag("model_version", modelVersion)
                .register(meterRegistry, accuracy, value -> value);
    }

    // Technical Metric Methods

    public void incrementApiRequest(String endpoint, String method, String status) {
        apiRequestCounter.increment(
            Tags.of(
                "endpoint", endpoint,
                "method", method,
                "status", status
            )
        );
    }

    public void incrementDatabaseQuery(String operation, String table, String status) {
        databaseQueryCounter.increment(
            Tags.of(
                "operation", operation,
                "table", table,
                "status", status
            )
        );
    }

    public void incrementCacheHit(String cacheKey, String operation) {
        cacheHitCounter.increment(
            Tags.of(
                "cache_key", cacheKey,
                "operation", operation
            )
        );
    }

    public void incrementCacheMiss(String cacheKey, String operation) {
        cacheMissCounter.increment(
            Tags.of(
                "cache_key", cacheKey,
                "operation", operation
            )
        );
    }

    public Timer.Sample startDatabaseQueryTimer() {
        return Timer.start(meterRegistry);
    }

    public void stopDatabaseQueryTimer(Timer.Sample sample, String operation, String table) {
        sample.stop(Timer.builder("intelliflow_database_query_duration_seconds")
                .tag("service", getServiceName())
                .tag("operation", operation)
                .tag("table", table)
                .register(meterRegistry));
    }

    public Timer.Sample startCacheOperationTimer() {
        return Timer.start(meterRegistry);
    }

    public void stopCacheOperationTimer(Timer.Sample sample, String operation) {
        sample.stop(Timer.builder("intelliflow_cache_operation_duration_seconds")
                .tag("service", getServiceName())
                .tag("operation", operation)
                .register(meterRegistry));
    }

    // System Health Metric Methods

    public void incrementError(String errorType, String component) {
        errorCounter.increment(
            Tags.of(
                "error_type", errorType,
                "component", component
            )
        );
    }

    public void setActiveUsers(long count) {
        activeUsers.set(count);
    }

    public void setDatabaseConnections(long count) {
        databaseConnections.set(count);
    }

    public void setRedisConnections(long count) {
        redisConnections.set(count);
    }

    public void recordCustomGauge(String name, double value, String... tags) {
        Gauge.builder("intelliflow_custom_" + name)
                .description("Custom gauge metric: " + name)
                .tag("service", getServiceName())
                .tags(tags)
                .register(meterRegistry, value, val -> val);
    }

    public void recordCustomCounter(String name, String... tags) {
        Counter.builder("intelliflow_custom_" + name + "_total")
                .description("Custom counter metric: " + name)
                .tag("service", getServiceName())
                .tags(tags)
                .register(meterRegistry)
                .increment();
    }

    public Timer.Sample startCustomTimer(String name, String... tags) {
        return Timer.start(meterRegistry);
    }

    public void stopCustomTimer(Timer.Sample sample, String name, String... tags) {
        sample.stop(Timer.builder("intelliflow_custom_" + name + "_duration_seconds")
                .description("Custom timer metric: " + name)
                .tag("service", getServiceName())
                .tags(tags)
                .register(meterRegistry));
    }

    // Utility Methods

    private String getServiceName() {
        return System.getProperty("spring.application.name", "intelliflow-service");
    }

    private String getScoreRange(double score) {
        if (score >= 0.8) return "high";
        if (score >= 0.5) return "medium";
        return "low";
    }

    private double calculateFraudRate() {
        // Calculate fraud rate based on recent transactions
        // This would typically query a repository or cache
        // For now, return a dummy value
        double totalTransactions = transactionProcessedCounter.count();
        double fraudTransactions = fraudDetectedCounter.count();
        
        if (totalTransactions == 0) return 0.0;
        return fraudTransactions / totalTransactions;
    }

    // Circuit Breaker Metrics
    public void recordCircuitBreakerState(String circuitBreakerName, String state) {
        Gauge.builder("intelliflow_circuit_breaker_state")
                .description("Circuit breaker state (0=closed, 1=open, 2=half-open)")
                .tag("service", getServiceName())
                .tag("circuit_breaker", circuitBreakerName)
                .register(meterRegistry, stateToNumber(state), val -> val);
    }

    private double stateToNumber(String state) {
        switch (state.toLowerCase()) {
            case "closed": return 0.0;
            case "open": return 1.0;
            case "half-open": return 2.0;
            default: return -1.0;
        }
    }

    // Queue Metrics
    public void recordQueueSize(String queueName, int size) {
        Gauge.builder("intelliflow_queue_size")
                .description("Current queue size")
                .tag("service", getServiceName())
                .tag("queue", queueName)
                .register(meterRegistry, size, val -> val);
    }

    public void recordQueueProcessingTime(String queueName, Duration duration) {
        Timer.builder("intelliflow_queue_processing_duration_seconds")
                .description("Time taken to process queue items")
                .tag("service", getServiceName())
                .tag("queue", queueName)
                .register(meterRegistry)
                .record(duration);
    }

    // Business KPI Metrics
    public void recordRevenue(double amount, String currency) {
        Counter.builder("intelliflow_revenue_total")
                .description("Total revenue processed")
                .tag("service", getServiceName())
                .tag("currency", currency)
                .register(meterRegistry)
                .increment(amount);
    }

    public void recordUserSessionDuration(Duration duration, String userType) {
        Timer.builder("intelliflow_user_session_duration_seconds")
                .description("User session duration")
                .tag("service", getServiceName())
                .tag("user_type", userType)
                .register(meterRegistry)
                .record(duration);
    }

    // ML Model Metrics
    public void recordModelPredictionCount(String modelName, String prediction) {
        Counter.builder("intelliflow_model_predictions_total")
                .description("Total model predictions")
                .tag("service", getServiceName())
                .tag("model_name", modelName)
                .tag("prediction", prediction)
                .register(meterRegistry)
                .increment();
    }

    public void recordFeatureComputationTime(String featureGroup, Duration duration) {
        Timer.builder("intelliflow_feature_computation_duration_seconds")
                .description("Time taken to compute features")
                .tag("service", getServiceName())
                .tag("feature_group", featureGroup)
                .register(meterRegistry)
                .record(duration);
    }

    // Security Metrics
    public void recordAuthenticationAttempt(String result, String method) {
        Counter.builder("intelliflow_authentication_attempts_total")
                .description("Authentication attempts")
                .tag("service", getServiceName())
                .tag("result", result) // success, failure, blocked
                .tag("method", method) // password, jwt, oauth
                .register(meterRegistry)
                .increment();
    }

    public void recordSecurityEvent(String eventType, String severity) {
        Counter.builder("intelliflow_security_events_total")
                .description("Security events")
                .tag("service", getServiceName())
                .tag("event_type", eventType)
                .tag("severity", severity)
                .register(meterRegistry)
                .increment();
    }
}