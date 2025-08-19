package com.intelliflow.gateway.config;

import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig.SlidingWindowType;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.timelimiter.TimeLimiterConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

/**
 * Circuit breaker configuration for API Gateway
 */
@Configuration
public class CircuitBreakerConfig {
    
    private static final Logger logger = LoggerFactory.getLogger(CircuitBreakerConfig.class);
    
    @Bean
    public CircuitBreakerRegistry circuitBreakerRegistry() {
        Map<String, io.github.resilience4j.circuitbreaker.CircuitBreakerConfig> configs = new HashMap<>();
        
        // User Service Circuit Breaker
        configs.put("user-service-cb", io.github.resilience4j.circuitbreaker.CircuitBreakerConfig.custom()
            .slidingWindowType(SlidingWindowType.COUNT_BASED)
            .slidingWindowSize(10)
            .minimumNumberOfCalls(5)
            .failureRateThreshold(50.0f)
            .waitDurationInOpenState(Duration.ofSeconds(30))
            .permittedNumberOfCallsInHalfOpenState(3)
            .slowCallRateThreshold(50.0f)
            .slowCallDurationThreshold(Duration.ofSeconds(2))
            .automaticTransitionFromOpenToHalfOpenEnabled(true)
            .recordExceptions(Exception.class)
            .ignoreExceptions(IllegalArgumentException.class)
            .build());
        
        // Transaction Service Circuit Breaker (More strict due to financial nature)
        configs.put("transaction-service-cb", io.github.resilience4j.circuitbreaker.CircuitBreakerConfig.custom()
            .slidingWindowType(SlidingWindowType.COUNT_BASED)
            .slidingWindowSize(8)
            .minimumNumberOfCalls(4)
            .failureRateThreshold(60.0f)
            .waitDurationInOpenState(Duration.ofSeconds(45))
            .permittedNumberOfCallsInHalfOpenState(2)
            .slowCallRateThreshold(60.0f)
            .slowCallDurationThreshold(Duration.ofSeconds(3))
            .automaticTransitionFromOpenToHalfOpenEnabled(true)
            .build());
        
        // Fraud Detection Service Circuit Breaker (Most strict)
        configs.put("fraud-detection-service-cb", io.github.resilience4j.circuitbreaker.CircuitBreakerConfig.custom()
            .slidingWindowType(SlidingWindowType.COUNT_BASED)
            .slidingWindowSize(6)
            .minimumNumberOfCalls(3)
            .failureRateThreshold(70.0f)
            .waitDurationInOpenState(Duration.ofSeconds(60))
            .permittedNumberOfCallsInHalfOpenState(2)
            .slowCallRateThreshold(70.0f)
            .slowCallDurationThreshold(Duration.ofSeconds(5))
            .automaticTransitionFromOpenToHalfOpenEnabled(true)
            .build());
        
        // Analytics Service Circuit Breaker (More lenient)
        configs.put("analytics-service-cb", io.github.resilience4j.circuitbreaker.CircuitBreakerConfig.custom()
            .slidingWindowType(SlidingWindowType.COUNT_BASED)
            .slidingWindowSize(8)
            .minimumNumberOfCalls(4)
            .failureRateThreshold(50.0f)
            .waitDurationInOpenState(Duration.ofSeconds(30))
            .permittedNumberOfCallsInHalfOpenState(3)
            .slowCallRateThreshold(50.0f)
            .slowCallDurationThreshold(Duration.ofSeconds(10))
            .automaticTransitionFromOpenToHalfOpenEnabled(true)
            .build());
        
        // Notification Service Circuit Breaker
        configs.put("notification-service-cb", io.github.resilience4j.circuitbreaker.CircuitBreakerConfig.custom()
            .slidingWindowType(SlidingWindowType.COUNT_BASED)
            .slidingWindowSize(10)
            .minimumNumberOfCalls(5)
            .failureRateThreshold(40.0f)
            .waitDurationInOpenState(Duration.ofSeconds(20))
            .permittedNumberOfCallsInHalfOpenState(3)
            .slowCallRateThreshold(40.0f)
            .slowCallDurationThreshold(Duration.ofSeconds(3))
            .automaticTransitionFromOpenToHalfOpenEnabled(true)
            .build());
        
        CircuitBreakerRegistry registry = CircuitBreakerRegistry.of(configs);
        
        // Add event listeners for monitoring
        registry.getAllCircuitBreakers().forEach(circuitBreaker -> {
            circuitBreaker.getEventPublisher()
                .onStateTransition(event -> 
                    logger.info("Circuit breaker {} state transition: {} -> {}", 
                        circuitBreaker.getName(), 
                        event.getStateTransition().getFromState(),
                        event.getStateTransition().getToState()))
                .onFailureRateExceeded(event ->
                    logger.warn("Circuit breaker {} failure rate exceeded: {}%", 
                        circuitBreaker.getName(), event.getFailureRate()))
                .onSlowCallRateExceeded(event ->
                    logger.warn("Circuit breaker {} slow call rate exceeded: {}%", 
                        circuitBreaker.getName(), event.getSlowCallRate()))
                .onCallNotPermitted(event ->
                    logger.warn("Circuit breaker {} call not permitted", circuitBreaker.getName()));
        });
        
        return registry;
    }
    
    @Bean
    public TimeLimiterConfig defaultTimeLimiterConfig() {
        return TimeLimiterConfig.custom()
            .timeoutDuration(Duration.ofSeconds(10))
            .cancelRunningFuture(true)
            .build();
    }
    
    @Bean
    public Map<String, TimeLimiterConfig> timeLimiterConfigs() {
        Map<String, TimeLimiterConfig> configs = new HashMap<>();
        
        configs.put("user-service-cb", TimeLimiterConfig.custom()
            .timeoutDuration(Duration.ofSeconds(5))
            .cancelRunningFuture(true)
            .build());
        
        configs.put("transaction-service-cb", TimeLimiterConfig.custom()
            .timeoutDuration(Duration.ofSeconds(10))
            .cancelRunningFuture(true)
            .build());
        
        configs.put("fraud-detection-service-cb", TimeLimiterConfig.custom()
            .timeoutDuration(Duration.ofSeconds(15))
            .cancelRunningFuture(true)
            .build());
        
        configs.put("analytics-service-cb", TimeLimiterConfig.custom()
            .timeoutDuration(Duration.ofSeconds(30))
            .cancelRunningFuture(true)
            .build());
        
        configs.put("notification-service-cb", TimeLimiterConfig.custom()
            .timeoutDuration(Duration.ofSeconds(8))
            .cancelRunningFuture(true)
            .build());
        
        return configs;
    }
    
    /**
     * Manual circuit breaker for critical operations
     */
    @Bean
    public CircuitBreaker criticalOperationsCircuitBreaker() {
        io.github.resilience4j.circuitbreaker.CircuitBreakerConfig config = 
            io.github.resilience4j.circuitbreaker.CircuitBreakerConfig.custom()
                .slidingWindowType(SlidingWindowType.TIME_BASED)
                .slidingWindowSize(60) // 60 seconds
                .minimumNumberOfCalls(10)
                .failureRateThreshold(30.0f) // Very strict for critical ops
                .waitDurationInOpenState(Duration.ofMinutes(2))
                .permittedNumberOfCallsInHalfOpenState(5)
                .slowCallRateThreshold(30.0f)
                .slowCallDurationThreshold(Duration.ofSeconds(1))
                .automaticTransitionFromOpenToHalfOpenEnabled(true)
                .build();
        
        CircuitBreaker circuitBreaker = CircuitBreaker.of("critical-operations-cb", config);
        
        // Add comprehensive monitoring
        circuitBreaker.getEventPublisher()
            .onStateTransition(event -> {
                logger.warn("CRITICAL: Circuit breaker state change: {} -> {}", 
                    event.getStateTransition().getFromState(),
                    event.getStateTransition().getToState());
                
                // In production, this could trigger alerts
                if (event.getStateTransition().getToState() == CircuitBreaker.State.OPEN) {
                    logger.error("ALERT: Critical operations circuit breaker is now OPEN - service degraded");
                }
            })
            .onFailureRateExceeded(event ->
                logger.error("ALERT: Critical operations failure rate exceeded: {}%", event.getFailureRate()))
            .onSlowCallRateExceeded(event ->
                logger.error("ALERT: Critical operations slow call rate exceeded: {}%", event.getSlowCallRate()));
        
        return circuitBreaker;
    }
}
