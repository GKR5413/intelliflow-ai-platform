package com.intelliflow.gateway.filter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;
import org.springframework.cloud.gateway.filter.GatewayFilter;
import org.springframework.cloud.gateway.filter.factory.AbstractGatewayFilterFactory;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.util.UUID;

@Component
public class LoggingFilter extends AbstractGatewayFilterFactory<LoggingFilter.Config> {
    
    private static final Logger logger = LoggerFactory.getLogger(LoggingFilter.class);
    private static final String CORRELATION_ID_HEADER = "X-Correlation-ID";
    private static final String REQUEST_ID_HEADER = "X-Request-ID";
    
    public LoggingFilter() {
        super(Config.class);
    }
    
    @Override
    public GatewayFilter apply(Config config) {
        return (exchange, chain) -> {
            ServerHttpRequest request = exchange.getRequest();
            ServerHttpResponse response = exchange.getResponse();
            
            long startTime = System.currentTimeMillis();
            
            // Generate or extract correlation ID
            String correlationId = request.getHeaders().getFirst(CORRELATION_ID_HEADER);
            if (correlationId == null) {
                correlationId = UUID.randomUUID().toString();
            }
            
            // Generate request ID
            String requestId = UUID.randomUUID().toString();
            
            // Set MDC for logging context
            MDC.put("correlationId", correlationId);
            MDC.put("requestId", requestId);
            
            try {
                // Log incoming request
                logRequest(request, correlationId, requestId);
                
                // Add headers to request for downstream services
                ServerHttpRequest modifiedRequest = request.mutate()
                    .header(CORRELATION_ID_HEADER, correlationId)
                    .header(REQUEST_ID_HEADER, requestId)
                    .build();
                
                // Add headers to response
                response.getHeaders().add(CORRELATION_ID_HEADER, correlationId);
                response.getHeaders().add(REQUEST_ID_HEADER, requestId);
                
                return chain.filter(exchange.mutate().request(modifiedRequest).build())
                    .doOnSuccess(aVoid -> {
                        long duration = System.currentTimeMillis() - startTime;
                        logResponse(response, duration, correlationId, requestId);
                    })
                    .doOnError(throwable -> {
                        long duration = System.currentTimeMillis() - startTime;
                        logError(request, response, throwable, duration, correlationId, requestId);
                    });
                    
            } finally {
                // Clean up MDC
                MDC.clear();
            }
        };
    }
    
    private void logRequest(ServerHttpRequest request, String correlationId, String requestId) {
        String method = request.getMethod() != null ? request.getMethod().name() : "UNKNOWN";
        String path = request.getURI().getPath();
        String query = request.getURI().getQuery();
        String userAgent = request.getHeaders().getFirst("User-Agent");
        String clientIp = getClientIpAddress(request);
        String userId = request.getHeaders().getFirst("X-User-Id");
        
        logger.info("Incoming Request - Method: {}, Path: {}, Query: {}, ClientIP: {}, UserAgent: {}, UserId: {}, CorrelationId: {}, RequestId: {}",
            method, path, query, clientIp, userAgent, userId, correlationId, requestId);
        
        // Log headers if debug is enabled
        if (logger.isDebugEnabled()) {
            request.getHeaders().forEach((name, values) -> 
                logger.debug("Request Header - {}: {}", name, String.join(", ", values)));
        }
    }
    
    private void logResponse(ServerHttpResponse response, long duration, String correlationId, String requestId) {
        int statusCode = response.getStatusCode() != null ? response.getStatusCode().value() : 0;
        
        logger.info("Outgoing Response - Status: {}, Duration: {}ms, CorrelationId: {}, RequestId: {}",
            statusCode, duration, correlationId, requestId);
        
        // Log response headers if debug is enabled
        if (logger.isDebugEnabled()) {
            response.getHeaders().forEach((name, values) -> 
                logger.debug("Response Header - {}: {}", name, String.join(", ", values)));
        }
        
        // Log slow requests
        if (duration > 5000) { // 5 seconds
            logger.warn("Slow Request Detected - Duration: {}ms, CorrelationId: {}, RequestId: {}",
                duration, correlationId, requestId);
        }
    }
    
    private void logError(ServerHttpRequest request, ServerHttpResponse response, Throwable throwable,
                         long duration, String correlationId, String requestId) {
        String method = request.getMethod() != null ? request.getMethod().name() : "UNKNOWN";
        String path = request.getURI().getPath();
        int statusCode = response.getStatusCode() != null ? response.getStatusCode().value() : 500;
        
        logger.error("Request Error - Method: {}, Path: {}, Status: {}, Duration: {}ms, " +
                    "CorrelationId: {}, RequestId: {}, Error: {}",
            method, path, statusCode, duration, correlationId, requestId, throwable.getMessage(), throwable);
    }
    
    private String getClientIpAddress(ServerHttpRequest request) {
        String[] headers = {
            "X-Forwarded-For",
            "X-Real-IP",
            "Proxy-Client-IP",
            "WL-Proxy-Client-IP",
            "HTTP_X_FORWARDED_FOR",
            "HTTP_X_FORWARDED",
            "HTTP_X_CLUSTER_CLIENT_IP",
            "HTTP_CLIENT_IP",
            "HTTP_FORWARDED_FOR",
            "HTTP_FORWARDED",
            "HTTP_VIA",
            "REMOTE_ADDR"
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
    
    public static class Config {
        private boolean logHeaders = false;
        private boolean logBody = false;
        private long slowRequestThreshold = 5000; // 5 seconds
        
        // Getters and Setters
        public boolean isLogHeaders() { return logHeaders; }
        public void setLogHeaders(boolean logHeaders) { this.logHeaders = logHeaders; }
        
        public boolean isLogBody() { return logBody; }
        public void setLogBody(boolean logBody) { this.logBody = logBody; }
        
        public long getSlowRequestThreshold() { return slowRequestThreshold; }
        public void setSlowRequestThreshold(long slowRequestThreshold) { this.slowRequestThreshold = slowRequestThreshold; }
    }
}
