package com.intelliflow.transaction.dto;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.intelliflow.transaction.entity.TransactionStatus;
import com.intelliflow.transaction.entity.TransactionType;
import com.intelliflow.transaction.entity.PaymentMethod;
import com.intelliflow.transaction.statemachine.TransactionState;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class TransactionSearchDto {
    
    private Long userId;
    private TransactionType type;
    private TransactionStatus status;
    private TransactionState currentState;
    private String currency;
    private String merchantId;
    private PaymentMethod paymentMethod;
    private String referenceNumber;
    private String correlationId;
    
    private BigDecimal minAmount;
    private BigDecimal maxAmount;
    
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime fromDate;
    
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime toDate;
    
    private String channel;
    private String source;
    private String initiatedBy;
    
    // For risk-based searches
    private Double minRiskScore;
    private Double maxRiskScore;
    
    // Constructors
    public TransactionSearchDto() {}
    
    // Getters and Setters
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    
    public TransactionType getType() { return type; }
    public void setType(TransactionType type) { this.type = type; }
    
    public TransactionStatus getStatus() { return status; }
    public void setStatus(TransactionStatus status) { this.status = status; }
    
    public TransactionState getCurrentState() { return currentState; }
    public void setCurrentState(TransactionState currentState) { this.currentState = currentState; }
    
    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }
    
    public String getMerchantId() { return merchantId; }
    public void setMerchantId(String merchantId) { this.merchantId = merchantId; }
    
    public PaymentMethod getPaymentMethod() { return paymentMethod; }
    public void setPaymentMethod(PaymentMethod paymentMethod) { this.paymentMethod = paymentMethod; }
    
    public String getReferenceNumber() { return referenceNumber; }
    public void setReferenceNumber(String referenceNumber) { this.referenceNumber = referenceNumber; }
    
    public String getCorrelationId() { return correlationId; }
    public void setCorrelationId(String correlationId) { this.correlationId = correlationId; }
    
    public BigDecimal getMinAmount() { return minAmount; }
    public void setMinAmount(BigDecimal minAmount) { this.minAmount = minAmount; }
    
    public BigDecimal getMaxAmount() { return maxAmount; }
    public void setMaxAmount(BigDecimal maxAmount) { this.maxAmount = maxAmount; }
    
    public LocalDateTime getFromDate() { return fromDate; }
    public void setFromDate(LocalDateTime fromDate) { this.fromDate = fromDate; }
    
    public LocalDateTime getToDate() { return toDate; }
    public void setToDate(LocalDateTime toDate) { this.toDate = toDate; }
    
    public String getChannel() { return channel; }
    public void setChannel(String channel) { this.channel = channel; }
    
    public String getSource() { return source; }
    public void setSource(String source) { this.source = source; }
    
    public String getInitiatedBy() { return initiatedBy; }
    public void setInitiatedBy(String initiatedBy) { this.initiatedBy = initiatedBy; }
    
    public Double getMinRiskScore() { return minRiskScore; }
    public void setMinRiskScore(Double minRiskScore) { this.minRiskScore = minRiskScore; }
    
    public Double getMaxRiskScore() { return maxRiskScore; }
    public void setMaxRiskScore(Double maxRiskScore) { this.maxRiskScore = maxRiskScore; }
}
