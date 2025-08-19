package com.intelliflow.shared.cache;

import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.Cache;
import org.springframework.cache.CacheManager;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;

import jakarta.annotation.PostConstruct;

/**
 * Cache metrics for monitoring
 */
@Component
public class CacheMetrics {
    
    private final CacheManager cacheManager;
    private final MeterRegistry meterRegistry;
    
    @Autowired(required = false)
    private RedisTemplate<String, Object> redisTemplate;
    
    public CacheMetrics(CacheManager cacheManager) {
        this.cacheManager = cacheManager;
        this.meterRegistry = null; // Will be injected if available
    }
    
    @Autowired(required = false)
    public CacheMetrics(CacheManager cacheManager, MeterRegistry meterRegistry) {
        this.cacheManager = cacheManager;
        this.meterRegistry = meterRegistry;
    }
    
    @PostConstruct
    public void registerMetrics() {
        if (meterRegistry != null) {
            // Register cache hit/miss metrics for each cache
            cacheManager.getCacheNames().forEach(cacheName -> {
                Cache cache = cacheManager.getCache(cacheName);
                if (cache != null) {
                    Gauge.builder("cache.size")
                        .description("Cache size")
                        .tag("cache", cacheName)
                        .register(meterRegistry, cache, this::getCacheSize);
                }
            });
            
            // Register Redis connection metrics if available
            if (redisTemplate != null) {
                Gauge.builder("redis.connections.active")
                    .description("Active Redis connections")
                    .register(meterRegistry, this, CacheMetrics::getActiveConnections);
            }
        }
    }
    
    private double getCacheSize(Cache cache) {
        try {
            // This is a simplified implementation
            // In a real scenario, you might need to use cache-specific APIs
            return 0.0; // Placeholder
        } catch (Exception e) {
            return 0.0;
        }
    }
    
    private double getActiveConnections() {
        try {
            // Get Redis connection info if available
            return 1.0; // Placeholder
        } catch (Exception e) {
            return 0.0;
        }
    }
}
