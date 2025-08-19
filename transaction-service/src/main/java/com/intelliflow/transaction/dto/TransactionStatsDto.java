package com.intelliflow.transaction.dto;

import com.intelliflow.transaction.entity.TransactionStatus;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

public class TransactionStatsDto {
    
    private Map<TransactionStatus, Long> statusCounts = new HashMap<>();
    private Map<String, BigDecimal> currencyAmounts = new HashMap<>();
    private Long totalTransactions = 0L;
    private BigDecimal totalVolume = BigDecimal.ZERO;
    private Double averageTransactionAmount = 0.0;
    private Double successRate = 0.0;
    
    // Constructors
    public TransactionStatsDto() {}
    
    // Helper methods
    public void addStatusCount(TransactionStatus status, Long count) {
        statusCounts.put(status, count);
        totalTransactions += count;
    }
    
    public void addCurrencyAmount(String currency, BigDecimal amount) {
        currencyAmounts.put(currency, amount);
        if ("USD".equals(currency)) { // Assume USD as base currency for total volume
            totalVolume = totalVolume.add(amount);
        }
    }
    
    public void calculateDerivedStats() {
        if (totalTransactions > 0) {
            averageTransactionAmount = totalVolume.doubleValue() / totalTransactions;
            
            Long successfulTransactions = statusCounts.getOrDefault(TransactionStatus.SUCCESS, 0L);
            successRate = (successfulTransactions.doubleValue() / totalTransactions) * 100;
        }
    }
    
    // Getters and Setters
    public Map<TransactionStatus, Long> getStatusCounts() { return statusCounts; }
    public void setStatusCounts(Map<TransactionStatus, Long> statusCounts) { this.statusCounts = statusCounts; }
    
    public Map<String, BigDecimal> getCurrencyAmounts() { return currencyAmounts; }
    public void setCurrencyAmounts(Map<String, BigDecimal> currencyAmounts) { this.currencyAmounts = currencyAmounts; }
    
    public Long getTotalTransactions() { return totalTransactions; }
    public void setTotalTransactions(Long totalTransactions) { this.totalTransactions = totalTransactions; }
    
    public BigDecimal getTotalVolume() { return totalVolume; }
    public void setTotalVolume(BigDecimal totalVolume) { this.totalVolume = totalVolume; }
    
    public Double getAverageTransactionAmount() { return averageTransactionAmount; }
    public void setAverageTransactionAmount(Double averageTransactionAmount) { this.averageTransactionAmount = averageTransactionAmount; }
    
    public Double getSuccessRate() { return successRate; }
    public void setSuccessRate(Double successRate) { this.successRate = successRate; }
}
