package com.intelliflow.gateway.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Fallback controller for circuit breaker patterns
 */
@RestController
@RequestMapping("/fallback")
public class FallbackController {
    
    private static final Logger logger = LoggerFactory.getLogger(FallbackController.class);
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    @GetMapping(value = "/user", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> userServiceFallback() {
        logger.warn("User service fallback triggered - service appears to be down");
        
        Map<String, Object> response = createFallbackResponse(
            "User Service Unavailable",
            "The user service is temporarily unavailable. Please try again later.",
            "USER_SERVICE_DOWN"
        );
        
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
            .contentType(MediaType.APPLICATION_JSON)
            .body(toJson(response));
    }
    
    @GetMapping(value = "/transaction", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> transactionServiceFallback() {
        logger.warn("Transaction service fallback triggered - service appears to be down");
        
        Map<String, Object> response = createFallbackResponse(
            "Transaction Service Unavailable",
            "The transaction service is temporarily unavailable. Your request is being queued for retry.",
            "TRANSACTION_SERVICE_DOWN"
        );
        
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
            .contentType(MediaType.APPLICATION_JSON)
            .body(toJson(response));
    }
    
    @GetMapping(value = "/fraud", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> fraudDetectionServiceFallback() {
        logger.warn("Fraud detection service fallback triggered - service appears to be down");
        
        Map<String, Object> response = createFallbackResponse(
            "Fraud Detection Service Unavailable",
            "Fraud detection is temporarily unavailable. Transactions are being processed with basic validation.",
            "FRAUD_SERVICE_DOWN"
        );
        
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
            .contentType(MediaType.APPLICATION_JSON)
            .body(toJson(response));
    }
    
    @GetMapping(value = "/analytics", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> analyticsServiceFallback() {
        logger.warn("Analytics service fallback triggered - service appears to be down");
        
        Map<String, Object> response = createFallbackResponse(
            "Analytics Service Unavailable",
            "Analytics data is temporarily unavailable. Please try again later.",
            "ANALYTICS_SERVICE_DOWN"
        );
        
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
            .contentType(MediaType.APPLICATION_JSON)
            .body(toJson(response));
    }
    
    @GetMapping(value = "/notification", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> notificationServiceFallback() {
        logger.warn("Notification service fallback triggered - service appears to be down");
        
        Map<String, Object> response = createFallbackResponse(
            "Notification Service Unavailable",
            "Notifications are temporarily unavailable. Your requests have been queued.",
            "NOTIFICATION_SERVICE_DOWN"
        );
        
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
            .contentType(MediaType.APPLICATION_JSON)
            .body(toJson(response));
    }
    
    @GetMapping(value = "/generic", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> genericFallback() {
        logger.warn("Generic fallback triggered - downstream service unavailable");
        
        Map<String, Object> response = createFallbackResponse(
            "Service Unavailable",
            "The requested service is temporarily unavailable. Please try again later.",
            "SERVICE_DOWN"
        );
        
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
            .contentType(MediaType.APPLICATION_JSON)
            .body(toJson(response));
    }
    
    @GetMapping(value = "/timeout", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> timeoutFallback() {
        logger.warn("Timeout fallback triggered - service response time exceeded");
        
        Map<String, Object> response = createFallbackResponse(
            "Request Timeout",
            "The service is taking longer than expected to respond. Please try again.",
            "SERVICE_TIMEOUT"
        );
        
        return ResponseEntity.status(HttpStatus.REQUEST_TIMEOUT)
            .contentType(MediaType.APPLICATION_JSON)
            .body(toJson(response));
    }
    
    private Map<String, Object> createFallbackResponse(String error, String message, String code) {
        Map<String, Object> response = new HashMap<>();
        response.put("success", false);
        response.put("error", error);
        response.put("message", message);
        response.put("errorCode", code);
        response.put("timestamp", LocalDateTime.now().toString());
        response.put("retryAfter", 30); // seconds
        
        // Add support information
        Map<String, String> support = new HashMap<>();
        support.put("contact", "support@intelliflow.com");
        support.put("documentation", "https://docs.intelliflow.com");
        response.put("support", support);
        
        return response;
    }
    
    private String toJson(Object object) {
        try {
            return objectMapper.writeValueAsString(object);
        } catch (JsonProcessingException e) {
            logger.error("Error converting fallback response to JSON", e);
            return "{\"error\":\"Internal Error\",\"message\":\"Service temporarily unavailable\"}";
        }
    }
}
