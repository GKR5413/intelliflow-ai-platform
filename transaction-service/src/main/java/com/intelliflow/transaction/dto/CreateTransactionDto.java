package com.intelliflow.transaction.dto;

import com.intelliflow.transaction.entity.PaymentMethod;
import com.intelliflow.transaction.entity.TransactionType;
import jakarta.validation.constraints.*;

import java.math.BigDecimal;

public class CreateTransactionDto {
    
    @NotNull(message = "User ID is required")
    private Long userId;
    
    @NotNull(message = "Transaction type is required")
    private TransactionType type;
    
    @NotNull(message = "Amount is required")
    @DecimalMin(value = "0.01", message = "Amount must be greater than 0")
    @DecimalMax(value = "100000.00", message = "Amount cannot exceed 100,000")
    private BigDecimal amount;
    
    @NotBlank(message = "Currency is required")
    @Pattern(regexp = "^[A-Z]{3}$", message = "Currency must be a valid 3-letter ISO code")
    private String currency;
    
    @Size(max = 500, message = "Description cannot exceed 500 characters")
    private String description;
    
    @NotBlank(message = "Merchant ID is required")
    @Size(max = 100, message = "Merchant ID cannot exceed 100 characters")
    private String merchantId;
    
    @Size(max = 200, message = "Merchant name cannot exceed 200 characters")
    private String merchantName;
    
    @NotNull(message = "Payment method is required")
    private PaymentMethod paymentMethod;
    
    @Size(max = 100, message = "Payment instrument ID cannot exceed 100 characters")
    private String paymentInstrumentId;
    
    @Size(max = 50, message = "Source cannot exceed 50 characters")
    private String source;
    
    @Size(max = 50, message = "Channel cannot exceed 50 characters")
    private String channel;
    
    @Size(max = 100, message = "Initiated by cannot exceed 100 characters")
    private String initiatedBy;
    
    // For multi-currency transactions
    @DecimalMin(value = "0.0001", message = "Exchange rate must be positive")
    private BigDecimal exchangeRate;
    
    @Pattern(regexp = "^[A-Z]{3}$", message = "Original currency must be a valid 3-letter ISO code")
    private String originalCurrency;
    
    private BigDecimal originalAmount;
    
    // For transfers
    @Size(max = 100, message = "To account ID cannot exceed 100 characters")
    private String toAccountId;
    
    // Metadata
    @Size(max = 1000, message = "Metadata cannot exceed 1000 characters")
    private String metadata;
    
    // Constructors
    public CreateTransactionDto() {}
    
    public CreateTransactionDto(Long userId, TransactionType type, BigDecimal amount, String currency) {
        this.userId = userId;
        this.type = type;
        this.amount = amount;
        this.currency = currency;
    }
    
    // Getters and Setters
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    
    public TransactionType getType() { return type; }
    public void setType(TransactionType type) { this.type = type; }
    
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
    
    public PaymentMethod getPaymentMethod() { return paymentMethod; }
    public void setPaymentMethod(PaymentMethod paymentMethod) { this.paymentMethod = paymentMethod; }
    
    public String getPaymentInstrumentId() { return paymentInstrumentId; }
    public void setPaymentInstrumentId(String paymentInstrumentId) { this.paymentInstrumentId = paymentInstrumentId; }
    
    public String getSource() { return source; }
    public void setSource(String source) { this.source = source; }
    
    public String getChannel() { return channel; }
    public void setChannel(String channel) { this.channel = channel; }
    
    public String getInitiatedBy() { return initiatedBy; }
    public void setInitiatedBy(String initiatedBy) { this.initiatedBy = initiatedBy; }
    
    public BigDecimal getExchangeRate() { return exchangeRate; }
    public void setExchangeRate(BigDecimal exchangeRate) { this.exchangeRate = exchangeRate; }
    
    public String getOriginalCurrency() { return originalCurrency; }
    public void setOriginalCurrency(String originalCurrency) { this.originalCurrency = originalCurrency; }
    
    public BigDecimal getOriginalAmount() { return originalAmount; }
    public void setOriginalAmount(BigDecimal originalAmount) { this.originalAmount = originalAmount; }
    
    public String getToAccountId() { return toAccountId; }
    public void setToAccountId(String toAccountId) { this.toAccountId = toAccountId; }
    
    public String getMetadata() { return metadata; }
    public void setMetadata(String metadata) { this.metadata = metadata; }
}
