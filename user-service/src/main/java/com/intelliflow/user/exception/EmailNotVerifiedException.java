package com.intelliflow.user.exception;

public class EmailNotVerifiedException extends RuntimeException {
    
    public EmailNotVerifiedException(String message) {
        super(message);
    }
    
    public EmailNotVerifiedException(String message, Throwable cause) {
        super(message, cause);
    }
}
