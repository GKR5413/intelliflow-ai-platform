package com.intelliflow.transaction.entity;

import com.intelliflow.shared.audit.Auditable;
import com.intelliflow.transaction.statemachine.TransactionState;
import jakarta.persistence.*;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotNull;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;
import org.hibernate.annotations.GenericGenerator;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.Currency;
import java.util.Objects;

@Entity
@Table(name = "transactions", indexes = {
    @Index(name = "idx_transaction_user_id", columnList = "user_id"),
    @Index(name = "idx_transaction_status", columnList = "status"),
    @Index(name = "idx_transaction_type", columnList = "type"),
    @Index(name = "idx_transaction_reference", columnList = "reference_number"),
    @Index(name = "idx_transaction_created_at", columnList = "created_at"),
    @Index(name = "idx_transaction_amount", columnList = "amount"),
    @Index(name = "idx_transaction_state", columnList = "current_state"),
    @Index(name = "idx_transaction_correlation", columnList = "correlation_id")
})
@EntityListeners(TransactionAuditListener.class)
public class Transaction extends Auditable {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(name = "user_id", nullable = false)
    @NotNull(message = "User ID is required")
    private Long userId;
    
    @Column(name = "reference_number", unique = true, nullable = false, length = 50)
    private String referenceNumber;
    
    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private TransactionType type;
    
    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private TransactionStatus status = TransactionStatus.PENDING;
    
    @Column(nullable = false, precision = 15, scale = 2)
    @DecimalMin(value = "0.01", message = "Amount must be greater than 0")
    private BigDecimal amount;
    
    @Column(nullable = false, length = 3)
    private String currency = "USD";
    
    @Column(length = 500)
    private String description;
    
    @Column(name = "merchant_id")
    private String merchantId;
    
    @Column(name = "merchant_name", length = 100)
    private String merchantName;
    
    @Column(name = "merchant_category", length = 50)
    private String merchantCategory;
    
    @Column(name = "payment_method")
    @Enumerated(EnumType.STRING)
    private PaymentMethod paymentMethod;
    
    @Column(name = "payment_instrument_id")
    private String paymentInstrumentId;
    
    @Column(name = "from_account_id")
    private String fromAccountId;
    
    @Column(name = "to_account_id")
    private String toAccountId;
    
    @Column(name = "external_transaction_id")
    private String externalTransactionId;
    
    @Column(name = "processing_fee", precision = 10, scale = 2)
    private BigDecimal processingFee = BigDecimal.ZERO;
    
    @Column(name = "exchange_rate", precision = 10, scale = 6)
    private BigDecimal exchangeRate;
    
    @Column(name = "original_amount", precision = 15, scale = 2)
    private BigDecimal originalAmount;
    
    @Column(name = "original_currency", length = 3)
    private String originalCurrency;
    
    @Column(name = "ip_address", length = 45)
    private String ipAddress;
    
    @Column(name = "user_agent", length = 500)
    private String userAgent;
    
    @Column(name = "device_fingerprint")
    private String deviceFingerprint;
    
    @Column(name = "location_data", length = 200)
    private String locationData;
    
    @Column(name = "risk_score", precision = 5, scale = 2)
    private BigDecimal riskScore;
    
    @Column(name = "fraud_check_status")
    @Enumerated(EnumType.STRING)
    private FraudCheckStatus fraudCheckStatus = FraudCheckStatus.PENDING;
    
    @Column(name = "fraud_check_details", columnDefinition = "TEXT")
    private String fraudCheckDetails;
    
    @Column(name = "authorization_code")
    private String authorizationCode;
    
    @Column(name = "settlement_date")
    private LocalDateTime settlementDate;
    
    @Column(name = "processed_at")
    private LocalDateTime processedAt;
    
    @Column(name = "failed_at")
    private LocalDateTime failedAt;
    
    @Column(name = "failure_reason", length = 500)
    private String failureReason;
    
    @Column(name = "retry_count")
    private Integer retryCount = 0;
    
    @Column(name = "max_retries")
    private Integer maxRetries = 3;
    
    @Column(name = "notification_sent")
    private Boolean notificationSent = false;
    
    @Column(columnDefinition = "TEXT")
    private String metadata;
    
    // State Management Fields
    @Enumerated(EnumType.STRING)
    @Column(name = "current_state", nullable = false)
    private TransactionState currentState = TransactionState.INITIATED;
    
    @Column(name = "previous_state")
    @Enumerated(EnumType.STRING)
    private TransactionState previousState;
    
    @Column(name = "state_changed_at")
    private LocalDateTime stateChangedAt;
    
    @Column(name = "correlation_id")
    private String correlationId;
    
    @Column(name = "parent_transaction_id")
    private Long parentTransactionId;
    
    @Column(name = "batch_id")
    private String batchId;
    
    // Audit Trail Enhancement
    @Column(name = "transaction_source", length = 50)
    private String transactionSource = "API";
    
    @Column(name = "channel", length = 50)
    private String channel;
    
    @Column(name = "initiated_by")
    private String initiatedBy;
    
    @Column(name = "approved_by")
    private String approvedBy;
    
    @Column(name = "approval_timestamp")
    private LocalDateTime approvalTimestamp;
    
    // Constructors
    public Transaction() {}
    
    public Transaction(Long userId, TransactionType type, BigDecimal amount, String currency) {
        this.userId = userId;
        this.type = type;
        this.amount = amount;
        this.currency = currency;
        this.generateReferenceNumber();
    }
    
    // Generate unique reference number
    public void generateReferenceNumber() {
        this.referenceNumber = "TXN" + System.currentTimeMillis() + 
                              String.format("%04d", (int)(Math.random() * 10000));
    }
    
    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    
    public String getReferenceNumber() { return referenceNumber; }
    public void setReferenceNumber(String referenceNumber) { this.referenceNumber = referenceNumber; }
    
    public TransactionType getType() { return type; }
    public void setType(TransactionType type) { this.type = type; }
    
    public TransactionStatus getStatus() { return status; }
    public void setStatus(TransactionStatus status) { this.status = status; }
    
    public BigDecimal getAmount() { return amount; }
    public void setAmount(BigDecimal amount) { this.amount = amount; }
    
    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }
    
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    
    public String getMerchantId() { return merchantId; }
    public void setMerchantId(String merchantId) { this.merchantId = merchantId; }
    
    public String getMerchantName() { return merchantName; }
    public void setMerchantName(String merchantName) { this.merchantName = merchantName; }
    
    public String getMerchantCategory() { return merchantCategory; }
    public void setMerchantCategory(String merchantCategory) { this.merchantCategory = merchantCategory; }
    
    public PaymentMethod getPaymentMethod() { return paymentMethod; }
    public void setPaymentMethod(PaymentMethod paymentMethod) { this.paymentMethod = paymentMethod; }
    
    public String getPaymentInstrumentId() { return paymentInstrumentId; }
    public void setPaymentInstrumentId(String paymentInstrumentId) { this.paymentInstrumentId = paymentInstrumentId; }
    
    public String getFromAccountId() { return fromAccountId; }
    public void setFromAccountId(String fromAccountId) { this.fromAccountId = fromAccountId; }
    
    public String getToAccountId() { return toAccountId; }
    public void setToAccountId(String toAccountId) { this.toAccountId = toAccountId; }
    
    public String getExternalTransactionId() { return externalTransactionId; }
    public void setExternalTransactionId(String externalTransactionId) { this.externalTransactionId = externalTransactionId; }
    
    public BigDecimal getProcessingFee() { return processingFee; }
    public void setProcessingFee(BigDecimal processingFee) { this.processingFee = processingFee; }
    
    public BigDecimal getExchangeRate() { return exchangeRate; }
    public void setExchangeRate(BigDecimal exchangeRate) { this.exchangeRate = exchangeRate; }
    
    public BigDecimal getOriginalAmount() { return originalAmount; }
    public void setOriginalAmount(BigDecimal originalAmount) { this.originalAmount = originalAmount; }
    
    public String getOriginalCurrency() { return originalCurrency; }
    public void setOriginalCurrency(String originalCurrency) { this.originalCurrency = originalCurrency; }
    
    public String getIpAddress() { return ipAddress; }
    public void setIpAddress(String ipAddress) { this.ipAddress = ipAddress; }
    
    public String getUserAgent() { return userAgent; }
    public void setUserAgent(String userAgent) { this.userAgent = userAgent; }
    
    public String getDeviceFingerprint() { return deviceFingerprint; }
    public void setDeviceFingerprint(String deviceFingerprint) { this.deviceFingerprint = deviceFingerprint; }
    
    public String getLocationData() { return locationData; }
    public void setLocationData(String locationData) { this.locationData = locationData; }
    
    public BigDecimal getRiskScore() { return riskScore; }
    public void setRiskScore(BigDecimal riskScore) { this.riskScore = riskScore; }
    
    public FraudCheckStatus getFraudCheckStatus() { return fraudCheckStatus; }
    public void setFraudCheckStatus(FraudCheckStatus fraudCheckStatus) { this.fraudCheckStatus = fraudCheckStatus; }
    
    public String getFraudCheckDetails() { return fraudCheckDetails; }
    public void setFraudCheckDetails(String fraudCheckDetails) { this.fraudCheckDetails = fraudCheckDetails; }
    
    public String getAuthorizationCode() { return authorizationCode; }
    public void setAuthorizationCode(String authorizationCode) { this.authorizationCode = authorizationCode; }
    
    public LocalDateTime getSettlementDate() { return settlementDate; }
    public void setSettlementDate(LocalDateTime settlementDate) { this.settlementDate = settlementDate; }
    
    public LocalDateTime getProcessedAt() { return processedAt; }
    public void setProcessedAt(LocalDateTime processedAt) { this.processedAt = processedAt; }
    
    public LocalDateTime getFailedAt() { return failedAt; }
    public void setFailedAt(LocalDateTime failedAt) { this.failedAt = failedAt; }
    
    public String getFailureReason() { return failureReason; }
    public void setFailureReason(String failureReason) { this.failureReason = failureReason; }
    
    public Integer getRetryCount() { return retryCount; }
    public void setRetryCount(Integer retryCount) { this.retryCount = retryCount; }
    
    public Integer getMaxRetries() { return maxRetries; }
    public void setMaxRetries(Integer maxRetries) { this.maxRetries = maxRetries; }
    
    public Boolean getNotificationSent() { return notificationSent; }
    public void setNotificationSent(Boolean notificationSent) { this.notificationSent = notificationSent; }
    
    public String getMetadata() { return metadata; }
    public void setMetadata(String metadata) { this.metadata = metadata; }
    
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    
    // New State Management Getters/Setters
    public TransactionState getCurrentState() { return currentState; }
    public void setCurrentState(TransactionState currentState) { 
        this.previousState = this.currentState;
        this.currentState = currentState; 
        this.stateChangedAt = LocalDateTime.now();
    }
    
    public TransactionState getPreviousState() { return previousState; }
    public void setPreviousState(TransactionState previousState) { this.previousState = previousState; }
    
    public LocalDateTime getStateChangedAt() { return stateChangedAt; }
    public void setStateChangedAt(LocalDateTime stateChangedAt) { this.stateChangedAt = stateChangedAt; }
    
    public String getCorrelationId() { return correlationId; }
    public void setCorrelationId(String correlationId) { this.correlationId = correlationId; }
    
    public Long getParentTransactionId() { return parentTransactionId; }
    public void setParentTransactionId(Long parentTransactionId) { this.parentTransactionId = parentTransactionId; }
    
    public String getBatchId() { return batchId; }
    public void setBatchId(String batchId) { this.batchId = batchId; }
    
    public String getTransactionSource() { return transactionSource; }
    public void setTransactionSource(String transactionSource) { this.transactionSource = transactionSource; }
    
    public String getChannel() { return channel; }
    public void setChannel(String channel) { this.channel = channel; }
    
    public String getInitiatedBy() { return initiatedBy; }
    public void setInitiatedBy(String initiatedBy) { this.initiatedBy = initiatedBy; }
    
    public String getApprovedBy() { return approvedBy; }
    public void setApprovedBy(String approvedBy) { this.approvedBy = approvedBy; }
    
    public LocalDateTime getApprovalTimestamp() { return approvalTimestamp; }
    public void setApprovalTimestamp(LocalDateTime approvalTimestamp) { this.approvalTimestamp = approvalTimestamp; }
    
    // State Management Utility Methods
    public void transitionToState(TransactionState newState) {
        if (canTransitionTo(newState)) {
            this.previousState = this.currentState;
            this.currentState = newState;
            this.stateChangedAt = LocalDateTime.now();
        } else {
            throw new IllegalStateException("Invalid state transition from " + currentState + " to " + newState);
        }
    }
    
    public boolean canTransitionTo(TransactionState targetState) {
        // Define valid state transitions
        return switch (currentState) {
            case INITIATED -> targetState == TransactionState.VALIDATED || targetState == TransactionState.FRAUD_DECLINED;
            case VALIDATED -> targetState == TransactionState.FRAUD_CHECKING;
            case FRAUD_CHECKING -> targetState == TransactionState.FRAUD_APPROVED || targetState == TransactionState.FRAUD_DECLINED;
            case FRAUD_APPROVED -> targetState == TransactionState.BALANCE_CHECKING;
            case BALANCE_CHECKING -> targetState == TransactionState.BALANCE_VERIFIED || targetState == TransactionState.BALANCE_INSUFFICIENT;
            case BALANCE_VERIFIED -> targetState == TransactionState.PROCESSING;
            case PROCESSING -> targetState == TransactionState.PAYMENT_AUTHORIZED || targetState == TransactionState.FAILED;
            case PAYMENT_AUTHORIZED -> targetState == TransactionState.PAYMENT_CAPTURED;
            case PAYMENT_CAPTURED -> targetState == TransactionState.COMPLETED;
            case COMPLETED -> targetState == TransactionState.REFUND_INITIATED || targetState == TransactionState.DISPUTED;
            case REFUND_INITIATED -> targetState == TransactionState.REFUND_COMPLETED;
            default -> false;
        };
    }
    
    public boolean isInFinalState() {
        return currentState == TransactionState.COMPLETED || 
               currentState == TransactionState.FAILED || 
               currentState == TransactionState.CANCELLED ||
               currentState == TransactionState.REFUND_COMPLETED ||
               currentState == TransactionState.DISPUTED;
    }
    
    // Utility methods
    public boolean isSuccessful() {
        return TransactionStatus.SUCCESS.equals(status);
    }
    
    public boolean isFailed() {
        return TransactionStatus.FAILED.equals(status);
    }
    
    public boolean isPending() {
        return TransactionStatus.PENDING.equals(status);
    }
    
    public boolean canRetry() {
        return retryCount < maxRetries && isFailed();
    }
    
    public void incrementRetryCount() {
        this.retryCount = (this.retryCount == null ? 0 : this.retryCount) + 1;
    }
    
    public BigDecimal getTotalAmount() {
        return amount.add(processingFee != null ? processingFee : BigDecimal.ZERO);
    }
    
    // Audit methods
    public void markAsApproved(String approver) {
        this.approvedBy = approver;
        this.approvalTimestamp = LocalDateTime.now();
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Transaction that = (Transaction) o;
        return Objects.equals(id, that.id) && Objects.equals(referenceNumber, that.referenceNumber);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(id, referenceNumber);
    }
    
    @Override
    public String toString() {
        return "Transaction{" +
               "id=" + id +
               ", referenceNumber='" + referenceNumber + '\'' +
               ", currentState=" + currentState +
               ", status=" + status +
               ", amount=" + amount +
               ", currency='" + currency + '\'' +
               '}';
    }
}
