package com.intelliflow.gateway.filter;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cloud.gateway.filter.GatewayFilter;
import org.springframework.cloud.gateway.filter.factory.AbstractGatewayFilterFactory;
import org.springframework.data.redis.core.ReactiveStringRedisTemplate;
import org.springframework.data.redis.core.script.RedisScript;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Component
public class RateLimitingFilter extends AbstractGatewayFilterFactory<RateLimitingFilter.Config> {
    
    private static final Logger logger = LoggerFactory.getLogger(RateLimitingFilter.class);
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    @Autowired
    private ReactiveStringRedisTemplate redisTemplate;
    
    // Lua script for rate limiting using sliding window
    private static final String RATE_LIMIT_SCRIPT = 
        "local key = KEYS[1]\n" +
        "local window = tonumber(ARGV[1])\n" +
        "local limit = tonumber(ARGV[2])\n" +
        "local current_time = tonumber(ARGV[3])\n" +
        "\n" +
        "-- Remove old entries outside the window\n" +
        "redis.call('ZREMRANGEBYSCORE', key, 0, current_time - window * 1000)\n" +
        "\n" +
        "-- Count current requests in window\n" +
        "local current_requests = redis.call('ZCARD', key)\n" +
        "\n" +
        "if current_requests < limit then\n" +
        "    -- Add current request\n" +
        "    redis.call('ZADD', key, current_time, current_time)\n" +
        "    -- Set expiration\n" +
        "    redis.call('EXPIRE', key, window)\n" +
        "    return {1, limit - current_requests - 1}\n" +
        "else\n" +
        "    return {0, 0}\n" +
        "end";
    
    public RateLimitingFilter() {
        super(Config.class);
    }
    
    @Override
    public GatewayFilter apply(Config config) {
        return (exchange, chain) -> {
            ServerHttpRequest request = exchange.getRequest();
            
            // Generate rate limit key based on IP address and user ID
            String rateLimitKey = generateRateLimitKey(request);
            
            // Execute rate limiting check
            return executeRateLimitCheck(rateLimitKey, config)
                .flatMap(result -> {
                    boolean allowed = result.get(0) == 1;
                    long remaining = result.get(1);
                    
                    if (allowed) {
                        // Add rate limit headers
                        ServerHttpResponse response = exchange.getResponse();
                        response.getHeaders().add("X-RateLimit-Limit", String.valueOf(config.getLimit()));
                        response.getHeaders().add("X-RateLimit-Remaining", String.valueOf(remaining));
                        response.getHeaders().add("X-RateLimit-Window", String.valueOf(config.getWindow().getSeconds()));
                        
                        logger.debug("Rate limit check passed for key: {} (remaining: {})", rateLimitKey, remaining);
                        return chain.filter(exchange);
                    } else {
                        logger.warn("Rate limit exceeded for key: {}", rateLimitKey);
                        return onRateLimitExceeded(exchange, config);
                    }
                })
                .onErrorResume(throwable -> {
                    logger.error("Rate limiting check failed for key: {}", rateLimitKey, throwable);
                    // If Redis is down, allow the request to proceed
                    return chain.filter(exchange);
                });
        };
    }
    
    private String generateRateLimitKey(ServerHttpRequest request) {
        // Try to get user ID from headers (set by authentication filter)
        String userId = request.getHeaders().getFirst("X-User-Id");
        
        if (userId != null) {
            return "rate_limit:user:" + userId;
        } else {
            // Fall back to IP address
            String clientIp = getClientIpAddress(request);
            return "rate_limit:ip:" + clientIp;
        }
    }
    
    private String getClientIpAddress(ServerHttpRequest request) {
        String[] headers = {
            "X-Forwarded-For",
            "X-Real-IP",
            "Proxy-Client-IP",
            "WL-Proxy-Client-IP"
        };
        
        for (String header : headers) {
            String value = request.getHeaders().getFirst(header);
            if (value != null && !value.isEmpty() && !"unknown".equalsIgnoreCase(value)) {
                return value.split(",")[0].trim();
            }
        }
        
        return request.getRemoteAddress() != null ? 
            request.getRemoteAddress().getAddress().getHostAddress() : "unknown";
    }
    
    private Mono<List<Long>> executeRateLimitCheck(String key, Config config) {
        long currentTime = System.currentTimeMillis();
        
        return redisTemplate.execute(
            RedisScript.of(RATE_LIMIT_SCRIPT, List.class),
            List.of(key),
            String.valueOf(config.getWindow().getSeconds()),
            String.valueOf(config.getLimit()),
            String.valueOf(currentTime)
        ).cast(List.class);
    }
    
    private Mono<Void> onRateLimitExceeded(ServerWebExchange exchange, Config config) {
        ServerHttpResponse response = exchange.getResponse();
        response.setStatusCode(HttpStatus.TOO_MANY_REQUESTS);
        response.getHeaders().add(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE);
        response.getHeaders().add("X-RateLimit-Limit", String.valueOf(config.getLimit()));
        response.getHeaders().add("X-RateLimit-Remaining", "0");
        response.getHeaders().add("X-RateLimit-Window", String.valueOf(config.getWindow().getSeconds()));
        response.getHeaders().add("Retry-After", String.valueOf(config.getWindow().getSeconds()));
        
        Map<String, Object> errorResponse = new HashMap<>();
        errorResponse.put("error", "Rate Limit Exceeded");
        errorResponse.put("message", "Too many requests. Please try again later.");
        errorResponse.put("limit", config.getLimit());
        errorResponse.put("window", config.getWindow().getSeconds());
        errorResponse.put("status", HttpStatus.TOO_MANY_REQUESTS.value());
        errorResponse.put("timestamp", System.currentTimeMillis());
        
        try {
            String json = objectMapper.writeValueAsString(errorResponse);
            org.springframework.core.io.buffer.DataBuffer buffer = response.bufferFactory().wrap(json.getBytes());
            return response.writeWith(Mono.just(buffer));
        } catch (JsonProcessingException e) {
            logger.error("Error creating rate limit error response", e);
            return response.setComplete();
        }
    }
    
    public static class Config {
        private int limit;
        private int replenishRate;
        private Duration window;
        
        public Config() {
            this.limit = 100;
            this.replenishRate = 1;
            this.window = Duration.ofMinutes(1);
        }
        
        public Config(int limit, int replenishRate, Duration window) {
            this.limit = limit;
            this.replenishRate = replenishRate;
            this.window = window;
        }
        
        // Getters and Setters
        public int getLimit() { return limit; }
        public void setLimit(int limit) { this.limit = limit; }
        
        public int getReplenishRate() { return replenishRate; }
        public void setReplenishRate(int replenishRate) { this.replenishRate = replenishRate; }
        
        public Duration getWindow() { return window; }
        public void setWindow(Duration window) { this.window = window; }
    }
}
