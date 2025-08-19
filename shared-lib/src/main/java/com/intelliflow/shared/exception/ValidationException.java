package com.intelliflow.shared.exception;

import java.util.Map;

/**
 * Exception thrown when validation fails
 */
public class ValidationException extends BusinessException {
    
    public ValidationException(String message) {
        super("VALIDATION_ERROR", message);
    }
    
    public ValidationException(String message, Map<String, String> fieldErrors) {
        super("VALIDATION_ERROR", message, fieldErrors);
    }
    
    public ValidationException(String message, Throwable cause) {
        super("VALIDATION_ERROR", message, cause);
    }
}
