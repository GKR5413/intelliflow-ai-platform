package com.intelliflow.shared.cache;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.Cache;
import org.springframework.cache.CacheManager;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.Set;
import java.util.concurrent.TimeUnit;

@Service
public class CacheService {
    
    private static final Logger logger = LoggerFactory.getLogger(CacheService.class);
    
    private final CacheManager cacheManager;
    private final RedisTemplate<String, Object> redisTemplate;
    
    @Autowired
    public CacheService(CacheManager cacheManager, RedisTemplate<String, Object> redisTemplate) {
        this.cacheManager = cacheManager;
        this.redisTemplate = redisTemplate;
    }
    
    /**
     * Cache user session with TTL
     */
    public void cacheUserSession(String sessionToken, Object sessionData, Duration ttl) {
        String key = "session:" + sessionToken;
        redisTemplate.opsForValue().set(key, sessionData, ttl.getSeconds(), TimeUnit.SECONDS);
        logger.debug("Cached user session: {} with TTL: {}", sessionToken, ttl);
    }
    
    /**
     * Get user session from cache
     */
    public Object getUserSession(String sessionToken) {
        String key = "session:" + sessionToken;
        Object session = redisTemplate.opsForValue().get(key);
        logger.debug("Retrieved user session: {} (found: {})", sessionToken, session != null);
        return session;
    }
    
    /**
     * Remove user session from cache
     */
    public void removeUserSession(String sessionToken) {
        String key = "session:" + sessionToken;
        redisTemplate.delete(key);
        logger.debug("Removed user session: {}", sessionToken);
    }
    
    /**
     * Cache transaction data with default TTL
     */
    public void cacheTransaction(String transactionId, Object transactionData) {
        Cache cache = cacheManager.getCache(CacheConfig.TRANSACTION_CACHE);
        if (cache != null) {
            cache.put(transactionId, transactionData);
            logger.debug("Cached transaction: {}", transactionId);
        }
    }
    
    /**
     * Get transaction from cache
     */
    public Object getTransaction(String transactionId) {
        Cache cache = cacheManager.getCache(CacheConfig.TRANSACTION_CACHE);
        if (cache != null) {
            Cache.ValueWrapper wrapper = cache.get(transactionId);
            if (wrapper != null) {
                logger.debug("Retrieved transaction from cache: {}", transactionId);
                return wrapper.get();
            }
        }
        logger.debug("Transaction not found in cache: {}", transactionId);
        return null;
    }
    
    /**
     * Cache fraud model results
     */
    public void cacheFraudModelResult(String modelKey, Object result, Duration ttl) {
        String key = "fraud_model:" + modelKey;
        redisTemplate.opsForValue().set(key, result, ttl.getSeconds(), TimeUnit.SECONDS);
        logger.debug("Cached fraud model result: {} with TTL: {}", modelKey, ttl);
    }
    
    /**
     * Get fraud model result from cache
     */
    public Object getFraudModelResult(String modelKey) {
        String key = "fraud_model:" + modelKey;
        Object result = redisTemplate.opsForValue().get(key);
        logger.debug("Retrieved fraud model result: {} (found: {})", modelKey, result != null);
        return result;
    }
    
    /**
     * Cache analytics data with custom TTL
     */
    public void cacheAnalyticsData(String analyticsKey, Object data, Duration ttl) {
        String key = "analytics:" + analyticsKey;
        redisTemplate.opsForValue().set(key, data, ttl.getSeconds(), TimeUnit.SECONDS);
        logger.debug("Cached analytics data: {} with TTL: {}", analyticsKey, ttl);
    }
    
    /**
     * Get analytics data from cache
     */
    public Object getAnalyticsData(String analyticsKey) {
        String key = "analytics:" + analyticsKey;
        Object data = redisTemplate.opsForValue().get(key);
        logger.debug("Retrieved analytics data: {} (found: {})", analyticsKey, data != null);
        return data;
    }
    
    /**
     * Cache user data
     */
    public void cacheUser(String userId, Object userData) {
        Cache cache = cacheManager.getCache(CacheConfig.USER_CACHE);
        if (cache != null) {
            cache.put(userId, userData);
            logger.debug("Cached user: {}", userId);
        }
    }
    
    /**
     * Get user from cache
     */
    public Object getUser(String userId) {
        Cache cache = cacheManager.getCache(CacheConfig.USER_CACHE);
        if (cache != null) {
            Cache.ValueWrapper wrapper = cache.get(userId);
            if (wrapper != null) {
                logger.debug("Retrieved user from cache: {}", userId);
                return wrapper.get();
            }
        }
        logger.debug("User not found in cache: {}", userId);
        return null;
    }
    
    /**
     * Remove user from cache
     */
    public void evictUser(String userId) {
        Cache cache = cacheManager.getCache(CacheConfig.USER_CACHE);
        if (cache != null) {
            cache.evict(userId);
            logger.debug("Evicted user from cache: {}", userId);
        }
    }
    
    /**
     * Cache notification template
     */
    public void cacheNotificationTemplate(String templateId, Object template) {
        Cache cache = cacheManager.getCache(CacheConfig.NOTIFICATION_TEMPLATE_CACHE);
        if (cache != null) {
            cache.put(templateId, template);
            logger.debug("Cached notification template: {}", templateId);
        }
    }
    
    /**
     * Get notification template from cache
     */
    public Object getNotificationTemplate(String templateId) {
        Cache cache = cacheManager.getCache(CacheConfig.NOTIFICATION_TEMPLATE_CACHE);
        if (cache != null) {
            Cache.ValueWrapper wrapper = cache.get(templateId);
            if (wrapper != null) {
                logger.debug("Retrieved notification template from cache: {}", templateId);
                return wrapper.get();
            }
        }
        logger.debug("Notification template not found in cache: {}", templateId);
        return null;
    }
    
    /**
     * Increment counter with expiration (for rate limiting)
     */
    public Long incrementCounter(String key, Duration expiration) {
        String counterKey = "counter:" + key;
        Long count = redisTemplate.opsForValue().increment(counterKey);
        if (count == 1) {
            // First increment, set expiration
            redisTemplate.expire(counterKey, expiration.getSeconds(), TimeUnit.SECONDS);
        }
        logger.debug("Incremented counter: {} to {}", key, count);
        return count;
    }
    
    /**
     * Get counter value
     */
    public Long getCounter(String key) {
        String counterKey = "counter:" + key;
        Object value = redisTemplate.opsForValue().get(counterKey);
        Long count = value != null ? Long.valueOf(value.toString()) : 0L;
        logger.debug("Retrieved counter: {} = {}", key, count);
        return count;
    }
    
    /**
     * Set with expiration using Redis native commands
     */
    public void setWithExpiration(String key, Object value, Duration ttl) {
        redisTemplate.opsForValue().set(key, value, ttl.getSeconds(), TimeUnit.SECONDS);
        logger.debug("Set key: {} with TTL: {}", key, ttl);
    }
    
    /**
     * Get value by key
     */
    public Object get(String key) {
        Object value = redisTemplate.opsForValue().get(key);
        logger.debug("Retrieved key: {} (found: {})", key, value != null);
        return value;
    }
    
    /**
     * Delete key
     */
    public void delete(String key) {
        redisTemplate.delete(key);
        logger.debug("Deleted key: {}", key);
    }
    
    /**
     * Delete keys by pattern
     */
    public void deleteByPattern(String pattern) {
        Set<String> keys = redisTemplate.keys(pattern);
        if (keys != null && !keys.isEmpty()) {
            redisTemplate.delete(keys);
            logger.debug("Deleted {} keys matching pattern: {}", keys.size(), pattern);
        }
    }
    
    /**
     * Check if key exists
     */
    public Boolean exists(String key) {
        Boolean exists = redisTemplate.hasKey(key);
        logger.debug("Key exists: {} = {}", key, exists);
        return exists;
    }
    
    /**
     * Get TTL of a key
     */
    public Duration getTtl(String key) {
        Long ttl = redisTemplate.getExpire(key, TimeUnit.SECONDS);
        Duration duration = ttl != null && ttl > 0 ? Duration.ofSeconds(ttl) : Duration.ZERO;
        logger.debug("TTL for key: {} = {}", key, duration);
        return duration;
    }
    
    /**
     * Clear all caches
     */
    public void clearAllCaches() {
        cacheManager.getCacheNames().forEach(cacheName -> {
            Cache cache = cacheManager.getCache(cacheName);
            if (cache != null) {
                cache.clear();
                logger.info("Cleared cache: {}", cacheName);
            }
        });
    }
    
    /**
     * Clear specific cache
     */
    public void clearCache(String cacheName) {
        Cache cache = cacheManager.getCache(cacheName);
        if (cache != null) {
            cache.clear();
            logger.info("Cleared cache: {}", cacheName);
        }
    }
    
    /**
     * Warm up caches with frequently accessed data
     */
    public void warmUpCaches() {
        logger.info("Starting cache warm-up process");
        // Implementation would load frequently accessed data into caches
        // This could be done asynchronously during application startup
    }
}
