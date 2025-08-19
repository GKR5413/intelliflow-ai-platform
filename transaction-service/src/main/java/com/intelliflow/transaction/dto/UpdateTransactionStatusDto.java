package com.intelliflow.transaction.dto;

import com.intelliflow.transaction.entity.TransactionStatus;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

public class UpdateTransactionStatusDto {
    
    @NotNull(message = "Status is required")
    private TransactionStatus status;
    
    @Size(max = 500, message = "Failure reason cannot exceed 500 characters")
    private String failureReason;
    
    @Size(max = 100, message = "Updated by cannot exceed 100 characters")
    private String updatedBy;
    
    // Constructors
    public UpdateTransactionStatusDto() {}
    
    public UpdateTransactionStatusDto(TransactionStatus status) {
        this.status = status;
    }
    
    public UpdateTransactionStatusDto(TransactionStatus status, String failureReason) {
        this.status = status;
        this.failureReason = failureReason;
    }
    
    // Getters and Setters
    public TransactionStatus getStatus() { return status; }
    public void setStatus(TransactionStatus status) { this.status = status; }
    
    public String getFailureReason() { return failureReason; }
    public void setFailureReason(String failureReason) { this.failureReason = failureReason; }
    
    public String getUpdatedBy() { return updatedBy; }
    public void setUpdatedBy(String updatedBy) { this.updatedBy = updatedBy; }
}
