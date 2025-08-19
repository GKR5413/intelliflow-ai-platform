package com.intelliflow.shared.cache;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.jsontype.DefaultBaseTypeLimitingValidator;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.CacheManager;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.data.redis.cache.RedisCacheConfiguration;
import org.springframework.data.redis.cache.RedisCacheManager;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.connection.RedisStandaloneConfiguration;
import org.springframework.data.redis.connection.lettuce.LettuceConnectionFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.serializer.GenericJackson2JsonRedisSerializer;
import org.springframework.data.redis.serializer.RedisSerializationContext;
import org.springframework.data.redis.serializer.StringRedisSerializer;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

@Configuration
@EnableCaching
public class CacheConfig {
    
    private static final Logger logger = LoggerFactory.getLogger(CacheConfig.class);
    
    @Value("${spring.data.redis.host:localhost}")
    private String redisHost;
    
    @Value("${spring.data.redis.port:6379}")
    private int redisPort;
    
    @Value("${spring.data.redis.password:}")
    private String redisPassword;
    
    @Value("${spring.data.redis.database:0}")
    private int redisDatabase;
    
    // Cache names and TTL configurations
    public static final String USER_CACHE = "users";
    public static final String USER_SESSION_CACHE = "user-sessions";
    public static final String TRANSACTION_CACHE = "transactions";
    public static final String FRAUD_MODEL_CACHE = "fraud-models";
    public static final String ANALYTICS_CACHE = "analytics";
    public static final String RATE_LIMIT_CACHE = "rate-limits";
    public static final String NOTIFICATION_TEMPLATE_CACHE = "notification-templates";
    
    @Bean
    public RedisConnectionFactory redisConnectionFactory() {
        RedisStandaloneConfiguration config = new RedisStandaloneConfiguration();
        config.setHostName(redisHost);
        config.setPort(redisPort);
        config.setDatabase(redisDatabase);
        
        if (redisPassword != null && !redisPassword.trim().isEmpty()) {
            config.setPassword(redisPassword);
        }
        
        LettuceConnectionFactory factory = new LettuceConnectionFactory(config);
        logger.info("Redis connection factory configured for {}:{}", redisHost, redisPort);
        return factory;
    }
    
    @Bean
    @Primary
    public CacheManager cacheManager(RedisConnectionFactory redisConnectionFactory) {
        RedisCacheConfiguration defaultCacheConfig = RedisCacheConfiguration.defaultCacheConfig()
            .entryTtl(Duration.ofMinutes(30))
            .serializeKeysWith(RedisSerializationContext.SerializationPair.fromSerializer(new StringRedisSerializer()))
            .serializeValuesWith(RedisSerializationContext.SerializationPair.fromSerializer(genericJackson2JsonRedisSerializer()))
            .disableCachingNullValues();
        
        Map<String, RedisCacheConfiguration> cacheConfigurations = new HashMap<>();
        
        // User cache - 1 hour TTL
        cacheConfigurations.put(USER_CACHE, defaultCacheConfig
            .entryTtl(Duration.ofHours(1)));
        
        // User session cache - 24 hours TTL
        cacheConfigurations.put(USER_SESSION_CACHE, defaultCacheConfig
            .entryTtl(Duration.ofHours(24)));
        
        // Transaction cache - 30 minutes TTL
        cacheConfigurations.put(TRANSACTION_CACHE, defaultCacheConfig
            .entryTtl(Duration.ofMinutes(30)));
        
        // Fraud model cache - 4 hours TTL (ML models don't change frequently)
        cacheConfigurations.put(FRAUD_MODEL_CACHE, defaultCacheConfig
            .entryTtl(Duration.ofHours(4)));
        
        // Analytics cache - 15 minutes TTL (for computed analytics)
        cacheConfigurations.put(ANALYTICS_CACHE, defaultCacheConfig
            .entryTtl(Duration.ofMinutes(15)));
        
        // Rate limit cache - 1 hour TTL
        cacheConfigurations.put(RATE_LIMIT_CACHE, defaultCacheConfig
            .entryTtl(Duration.ofHours(1)));
        
        // Notification template cache - 2 hours TTL
        cacheConfigurations.put(NOTIFICATION_TEMPLATE_CACHE, defaultCacheConfig
            .entryTtl(Duration.ofHours(2)));
        
        return RedisCacheManager.builder(redisConnectionFactory)
            .cacheDefaults(defaultCacheConfig)
            .withInitialCacheConfigurations(cacheConfigurations)
            .build();
    }
    
    @Bean
    public RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory redisConnectionFactory) {
        RedisTemplate<String, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(redisConnectionFactory);
        
        // Configure serializers
        template.setKeySerializer(new StringRedisSerializer());
        template.setHashKeySerializer(new StringRedisSerializer());
        template.setValueSerializer(genericJackson2JsonRedisSerializer());
        template.setHashValueSerializer(genericJackson2JsonRedisSerializer());
        
        template.setDefaultSerializer(genericJackson2JsonRedisSerializer());
        template.afterPropertiesSet();
        
        return template;
    }
    
    @Bean
    public GenericJackson2JsonRedisSerializer genericJackson2JsonRedisSerializer() {
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.ANY);
        objectMapper.activateDefaultTyping(
            new DefaultBaseTypeLimitingValidator(),
            ObjectMapper.DefaultTyping.NON_FINAL
        );
        objectMapper.registerModule(new JavaTimeModule());
        
        return new GenericJackson2JsonRedisSerializer(objectMapper);
    }
    
    @Bean
    public CacheMetrics cacheMetrics(CacheManager cacheManager) {
        return new CacheMetrics(cacheManager);
    }
}
