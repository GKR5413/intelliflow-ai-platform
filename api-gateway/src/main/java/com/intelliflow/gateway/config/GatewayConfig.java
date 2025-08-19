package com.intelliflow.gateway.config;

import com.intelliflow.gateway.filter.AuthenticationFilter;
import com.intelliflow.gateway.filter.LoggingFilter;
import com.intelliflow.gateway.filter.RateLimitingFilter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cloud.circuitbreaker.resilience4j.ReactiveResilience4JCircuitBreakerFactory;
import org.springframework.cloud.circuitbreaker.resilience4j.Resilience4JConfigBuilder;
import org.springframework.cloud.client.circuitbreaker.Customizer;
import org.springframework.cloud.gateway.route.RouteLocator;
import org.springframework.cloud.gateway.route.builder.RouteLocatorBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;

import java.time.Duration;

@Configuration
public class GatewayConfig {
    
    @Autowired
    private AuthenticationFilter authenticationFilter;
    
    @Autowired
    private LoggingFilter loggingFilter;
    
    @Autowired
    private RateLimitingFilter rateLimitingFilter;
    
    @Bean
    public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
        return builder.routes()
            // User Service Routes
            .route("user-service-auth", r -> r
                .path("/api/v1/auth/**")
                .and()
                .method(HttpMethod.POST)
                .filters(f -> f
                    .filter(loggingFilter.apply(new LoggingFilter.Config()))
                    .filter(rateLimitingFilter.apply(new RateLimitingFilter.Config(100, 1, Duration.ofMinutes(1))))
                    .circuitBreaker(config -> config
                        .setName("user-service-cb")
                        .setFallbackUri("forward:/fallback/user"))
                    .retry(config -> config
                        .setRetries(3)
                        .setMethods(HttpMethod.GET, HttpMethod.POST)
                        .setBackoff(Duration.ofMillis(100), Duration.ofMillis(1000), 2, false)))
                .uri("lb://user-service"))
            
            .route("user-service-profile", r -> r
                .path("/api/v1/users/**")
                .filters(f -> f
                    .filter(authenticationFilter.apply(new AuthenticationFilter.Config()))
                    .filter(loggingFilter.apply(new LoggingFilter.Config()))
                    .filter(rateLimitingFilter.apply(new RateLimitingFilter.Config(50, 1, Duration.ofMinutes(1))))
                    .circuitBreaker(config -> config
                        .setName("user-service-cb")
                        .setFallbackUri("forward:/fallback/user"))
                    .retry(config -> config
                        .setRetries(3)
                        .setMethods(HttpMethod.GET)
                        .setBackoff(Duration.ofMillis(100), Duration.ofMillis(1000), 2, false)))
                .uri("lb://user-service"))
            
            // Transaction Service Routes
            .route("transaction-service", r -> r
                .path("/api/v1/transactions/**")
                .filters(f -> f
                    .filter(authenticationFilter.apply(new AuthenticationFilter.Config()))
                    .filter(loggingFilter.apply(new LoggingFilter.Config()))
                    .filter(rateLimitingFilter.apply(new RateLimitingFilter.Config(30, 1, Duration.ofMinutes(1))))
                    .circuitBreaker(config -> config
                        .setName("transaction-service-cb")
                        .setFallbackUri("forward:/fallback/transaction"))
                    .retry(config -> config
                        .setRetries(2)
                        .setMethods(HttpMethod.GET)
                        .setBackoff(Duration.ofMillis(200), Duration.ofMillis(2000), 2, false)))
                .uri("lb://transaction-service"))
            
            // Fraud Detection Service Routes
            .route("fraud-detection-service", r -> r
                .path("/api/v1/fraud/**")
                .filters(f -> f
                    .filter(authenticationFilter.apply(new AuthenticationFilter.Config()))
                    .filter(loggingFilter.apply(new LoggingFilter.Config()))
                    .filter(rateLimitingFilter.apply(new RateLimitingFilter.Config(20, 1, Duration.ofMinutes(1))))
                    .circuitBreaker(config -> config
                        .setName("fraud-detection-service-cb")
                        .setFallbackUri("forward:/fallback/fraud"))
                    .retry(config -> config
                        .setRetries(1)
                        .setMethods(HttpMethod.GET)
                        .setBackoff(Duration.ofMillis(500), Duration.ofMillis(3000), 2, false)))
                .uri("lb://fraud-detection-service"))
            
            // Analytics Service Routes
            .route("analytics-service", r -> r
                .path("/api/v1/analytics/**")
                .filters(f -> f
                    .filter(authenticationFilter.apply(new AuthenticationFilter.Config()))
                    .filter(loggingFilter.apply(new LoggingFilter.Config()))
                    .filter(rateLimitingFilter.apply(new RateLimitingFilter.Config(10, 1, Duration.ofMinutes(1))))
                    .circuitBreaker(config -> config
                        .setName("analytics-service-cb")
                        .setFallbackUri("forward:/fallback/analytics"))
                    .retry(config -> config
                        .setRetries(2)
                        .setMethods(HttpMethod.GET)
                        .setBackoff(Duration.ofMillis(300), Duration.ofMillis(2000), 2, false)))
                .uri("lb://analytics-service"))
            
            // Notification Service Routes
            .route("notification-service", r -> r
                .path("/api/v1/notifications/**")
                .filters(f -> f
                    .filter(authenticationFilter.apply(new AuthenticationFilter.Config()))
                    .filter(loggingFilter.apply(new LoggingFilter.Config()))
                    .filter(rateLimitingFilter.apply(new RateLimitingFilter.Config(40, 1, Duration.ofMinutes(1))))
                    .circuitBreaker(config -> config
                        .setName("notification-service-cb")
                        .setFallbackUri("forward:/fallback/notification"))
                    .retry(config -> config
                        .setRetries(3)
                        .setMethods(HttpMethod.GET, HttpMethod.POST)
                        .setBackoff(Duration.ofMillis(100), Duration.ofMillis(1000), 2, false)))
                .uri("lb://notification-service"))
            
            // Health Check Routes (no authentication required)
            .route("health-checks", r -> r
                .path("/actuator/**")
                .filters(f -> f
                    .filter(loggingFilter.apply(new LoggingFilter.Config()))
                    .filter(rateLimitingFilter.apply(new RateLimitingFilter.Config(200, 1, Duration.ofMinutes(1)))))
                .uri("lb://user-service"))
            
            .build();
    }
    
    @Bean
    public Customizer<ReactiveResilience4JCircuitBreakerFactory> defaultCustomizer() {
        return factory -> factory.configureDefault(id -> new Resilience4JConfigBuilder(id)
            .circuitBreakerConfig(io.github.resilience4j.circuitbreaker.CircuitBreakerConfig.custom()
                .slidingWindowSize(10)
                .minimumNumberOfCalls(5)
                .failureRateThreshold(50.0f)
                .waitDurationInOpenState(Duration.ofSeconds(30))
                .permittedNumberOfCallsInHalfOpenState(3)
                .slowCallRateThreshold(50.0f)
                .slowCallDurationThreshold(Duration.ofSeconds(2))
                .build())
            .timeLimiterConfig(io.github.resilience4j.timelimiter.TimeLimiterConfig.custom()
                .timeoutDuration(Duration.ofSeconds(10))
                .build())
            .build());
    }
}
