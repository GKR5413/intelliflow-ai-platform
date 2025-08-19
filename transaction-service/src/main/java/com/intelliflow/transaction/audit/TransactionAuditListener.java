package com.intelliflow.transaction.audit;

import com.intelliflow.transaction.entity.Transaction;
import com.intelliflow.transaction.entity.TransactionAuditLog;
import com.intelliflow.transaction.repository.TransactionAuditLogRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;

import jakarta.persistence.*;
import java.time.LocalDateTime;

/**
 * JPA Entity Listener for Transaction audit logging
 */
@Component
public class TransactionAuditListener {
    
    private static final Logger logger = LoggerFactory.getLogger(TransactionAuditListener.class);
    
    @Autowired
    private TransactionAuditLogRepository auditLogRepository;
    
    @PrePersist
    public void prePersist(Transaction transaction) {
        logger.debug("Transaction entity about to be persisted: {}", transaction.getReferenceNumber());
        
        // Set correlation ID if not already set
        if (transaction.getCorrelationId() == null) {
            transaction.setCorrelationId(java.util.UUID.randomUUID().toString());
        }
        
        // Set initiated by if not already set
        if (transaction.getInitiatedBy() == null) {
            transaction.setInitiatedBy(getCurrentUser());
        }
        
        // Log the creation
        createAuditLog(transaction, "CREATED", null, transaction.getCurrentState().toString());
    }
    
    @PostPersist
    public void postPersist(Transaction transaction) {
        logger.info("Transaction created: {} with ID: {}", transaction.getReferenceNumber(), transaction.getId());
    }
    
    @PreUpdate
    public void preUpdate(Transaction transaction) {
        logger.debug("Transaction entity about to be updated: {}", transaction.getReferenceNumber());
        
        // This will be called before the update
        // We can capture the old state here if needed
    }
    
    @PostUpdate
    public void postUpdate(Transaction transaction) {
        logger.debug("Transaction updated: {}", transaction.getReferenceNumber());
        
        // Log state changes
        if (transaction.getPreviousState() != null && 
            !transaction.getPreviousState().equals(transaction.getCurrentState())) {
            createAuditLog(transaction, "STATE_CHANGED", 
                transaction.getPreviousState().toString(), 
                transaction.getCurrentState().toString());
        }
        
        // Log status changes
        createAuditLog(transaction, "UPDATED", null, transaction.getStatus().toString());
    }
    
    @PreRemove
    public void preRemove(Transaction transaction) {
        logger.warn("Transaction entity about to be removed: {}", transaction.getReferenceNumber());
        createAuditLog(transaction, "DELETED", null, null);
    }
    
    @PostRemove
    public void postRemove(Transaction transaction) {
        logger.warn("Transaction removed: {}", transaction.getReferenceNumber());
    }
    
    @PostLoad
    public void postLoad(Transaction transaction) {
        logger.debug("Transaction loaded: {}", transaction.getReferenceNumber());
    }
    
    private void createAuditLog(Transaction transaction, String action, String oldValue, String newValue) {
        try {
            TransactionAuditLog auditLog = new TransactionAuditLog();
            auditLog.setTransactionId(transaction.getId());
            auditLog.setReferenceNumber(transaction.getReferenceNumber());
            auditLog.setAction(action);
            auditLog.setOldValue(oldValue);
            auditLog.setNewValue(newValue);
            auditLog.setUserId(getCurrentUser());
            auditLog.setTimestamp(LocalDateTime.now());
            auditLog.setCorrelationId(transaction.getCorrelationId());
            
            // Add additional context
            auditLog.setUserAgent(getCurrentUserAgent());
            auditLog.setIpAddress(getCurrentIpAddress());
            
            if (auditLogRepository != null) {
                auditLogRepository.save(auditLog);
            }
        } catch (Exception e) {
            logger.error("Failed to create audit log for transaction: {}", transaction.getReferenceNumber(), e);
        }
    }
    
    private String getCurrentUser() {
        try {
            Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
            if (authentication != null && authentication.isAuthenticated()) {
                return authentication.getName();
            }
        } catch (Exception e) {
            logger.debug("Could not get current user", e);
        }
        return "SYSTEM";
    }
    
    private String getCurrentUserAgent() {
        // In a real implementation, this would be extracted from the current HTTP request
        // For now, return a default value
        return "N/A";
    }
    
    private String getCurrentIpAddress() {
        // In a real implementation, this would be extracted from the current HTTP request
        // For now, return a default value
        return "N/A";
    }
}
