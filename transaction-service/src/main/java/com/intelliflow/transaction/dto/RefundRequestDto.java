package com.intelliflow.transaction.dto;

import jakarta.validation.constraints.*;

import java.math.BigDecimal;

public class RefundRequestDto {
    
    @NotNull(message = "Refund amount is required")
    @DecimalMin(value = "0.01", message = "Refund amount must be greater than 0")
    private BigDecimal amount;
    
    @NotBlank(message = "Refund reason is required")
    @Size(max = 500, message = "Refund reason cannot exceed 500 characters")
    private String reason;
    
    @Size(max = 100, message = "Requested by cannot exceed 100 characters")
    private String requestedBy;
    
    @Size(max = 1000, message = "Additional notes cannot exceed 1000 characters")
    private String notes;
    
    // For partial refunds, specify if this is the final refund
    private Boolean isFinalRefund = false;
    
    // Constructors
    public RefundRequestDto() {}
    
    public RefundRequestDto(BigDecimal amount, String reason) {
        this.amount = amount;
        this.reason = reason;
    }
    
    // Getters and Setters
    public BigDecimal getAmount() { return amount; }
    public void setAmount(BigDecimal amount) { this.amount = amount; }
    
    public String getReason() { return reason; }
    public void setReason(String reason) { this.reason = reason; }
    
    public String getRequestedBy() { return requestedBy; }
    public void setRequestedBy(String requestedBy) { this.requestedBy = requestedBy; }
    
    public String getNotes() { return notes; }
    public void setNotes(String notes) { this.notes = notes; }
    
    public Boolean getIsFinalRefund() { return isFinalRefund; }
    public void setIsFinalRefund(Boolean isFinalRefund) { this.isFinalRefund = isFinalRefund; }
}
