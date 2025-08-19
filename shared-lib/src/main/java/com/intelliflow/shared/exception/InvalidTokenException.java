package com.intelliflow.shared.exception;

/**
 * Exception thrown when JWT token is invalid
 */
public class InvalidTokenException extends BusinessException {
    
    public InvalidTokenException(String message) {
        super("INVALID_TOKEN", message);
    }
    
    public InvalidTokenException(String message, Throwable cause) {
        super("INVALID_TOKEN", message, cause);
    }
}
