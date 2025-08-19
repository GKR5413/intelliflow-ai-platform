package com.intelliflow.transaction.service;

import com.intelliflow.transaction.dto.TransactionValidationDto;
import com.intelliflow.transaction.dto.ValidationResult;
import com.intelliflow.transaction.entity.Transaction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

@Service
public class TransactionValidationService {
    
    private static final Logger logger = LoggerFactory.getLogger(TransactionValidationService.class);
    
    private static final BigDecimal MAX_TRANSACTION_AMOUNT = new BigDecimal("100000.00");
    private static final BigDecimal MIN_TRANSACTION_AMOUNT = new BigDecimal("0.01");
    
    /**
     * Validate transaction data
     */
    public CompletableFuture<ValidationResult> validateTransaction(Transaction transaction) {
        return CompletableFuture.supplyAsync(() -> {
            logger.info("Validating transaction {}", transaction.getId());
            
            List<String> errors = new ArrayList<>();
            
            // Validate amount
            if (transaction.getAmount() == null) {
                errors.add("Transaction amount is required");
            } else {
                if (transaction.getAmount().compareTo(MIN_TRANSACTION_AMOUNT) < 0) {
                    errors.add("Transaction amount must be at least " + MIN_TRANSACTION_AMOUNT);
                }
                if (transaction.getAmount().compareTo(MAX_TRANSACTION_AMOUNT) > 0) {
                    errors.add("Transaction amount exceeds maximum limit of " + MAX_TRANSACTION_AMOUNT);
                }
            }
            
            // Validate currency
            if (transaction.getCurrency() == null || transaction.getCurrency().trim().isEmpty()) {
                errors.add("Currency is required");
            } else if (!isValidCurrency(transaction.getCurrency())) {
                errors.add("Invalid currency code: " + transaction.getCurrency());
            }
            
            // Validate user ID
            if (transaction.getUserId() == null) {
                errors.add("User ID is required");
            }
            
            // Validate payment method
            if (transaction.getPaymentMethod() == null) {
                errors.add("Payment method is required");
            }
            
            // Validate merchant information
            if (transaction.getMerchantId() == null || transaction.getMerchantId().trim().isEmpty()) {
                errors.add("Merchant ID is required");
            }
            
            // Additional business validations
            validateBusinessRules(transaction, errors);
            
            boolean isValid = errors.isEmpty();
            ValidationResult result = new ValidationResult(isValid, errors);
            
            logger.info("Transaction {} validation {}", transaction.getId(), 
                isValid ? "passed" : "failed with " + errors.size() + " errors");
            
            return result;
        });
    }
    
    /**
     * Validate business rules
     */
    private void validateBusinessRules(Transaction transaction, List<String> errors) {
        // Check for duplicate transaction (same amount, user, merchant within time window)
        validateDuplicateTransaction(transaction, errors);
        
        // Validate transaction type specific rules
        validateTransactionTypeRules(transaction, errors);
        
        // Validate payment method specific rules
        validatePaymentMethodRules(transaction, errors);
        
        // Validate processing fee
        validateProcessingFee(transaction, errors);
        
        // Validate exchange rate for multi-currency transactions
        validateCurrencyConversion(transaction, errors);
        
        // Validate risk-based rules
        validateRiskRules(transaction, errors);
        
        // Validate time-based rules
        validateTimeBasedRules(transaction, errors);
        
        // Validate merchant-specific rules
        validateMerchantRules(transaction, errors);
    }
    
    private void validateDuplicateTransaction(Transaction transaction, List<String> errors) {
        // Check for potential duplicate within last 5 minutes
        LocalDateTime fiveMinutesAgo = LocalDateTime.now().minusMinutes(5);
        
        // This would be implemented with actual database lookup
        // For now, just a placeholder check
        if (transaction.getAmount().compareTo(new BigDecimal("0.01")) == 0) {
            errors.add("Suspicious duplicate transaction pattern detected");
        }
    }
    
    private void validateTransactionTypeRules(Transaction transaction, List<String> errors) {
        switch (transaction.getType()) {
            case REFUND:
                if (transaction.getExternalTransactionId() == null) {
                    errors.add("Original transaction ID is required for refunds");
                }
                if (transaction.getAmount().compareTo(BigDecimal.ZERO) <= 0) {
                    errors.add("Refund amount must be positive");
                }
                break;
                
            case TRANSFER:
                if (transaction.getToAccountId() == null) {
                    errors.add("Destination account is required for transfers");
                }
                if (transaction.getUserId() != null && 
                    transaction.getToAccountId() != null && 
                    transaction.getUserId().toString().equals(transaction.getToAccountId())) {
                    errors.add("Cannot transfer to same account");
                }
                break;
                
            case PAYMENT:
                if (transaction.getMerchantId() == null) {
                    errors.add("Merchant ID is required for payments");
                }
                break;
                
            case DEPOSIT:
                // Deposits have different validation rules
                if (transaction.getAmount().compareTo(new BigDecimal("50000")) > 0) {
                    errors.add("Large deposits require additional verification");
                }
                break;
                
            case WITHDRAWAL:
                // Withdrawals have stricter limits
                if (transaction.getAmount().compareTo(new BigDecimal("10000")) > 0) {
                    errors.add("Withdrawal amount exceeds daily limit");
                }
                break;
                
            default:
                break;
        }
    }
    
    private void validatePaymentMethodRules(Transaction transaction, List<String> errors) {
        if (transaction.getPaymentMethod() == null) {
            errors.add("Payment method is required");
            return;
        }
        
        switch (transaction.getPaymentMethod()) {
            case CREDIT_CARD:
                if (transaction.getPaymentInstrumentId() == null) {
                    errors.add("Credit card ID is required for card payments");
                }
                // Check for card-specific limits
                if (transaction.getAmount().compareTo(new BigDecimal("5000")) > 0) {
                    errors.add("Credit card transaction exceeds single transaction limit");
                }
                break;
                
            case DEBIT_CARD:
                if (transaction.getPaymentInstrumentId() == null) {
                    errors.add("Debit card ID is required for card payments");
                }
                break;
                
            case BANK_TRANSFER:
                // Bank transfers have higher limits but longer processing
                if (transaction.getAmount().compareTo(new BigDecimal("100000")) > 0) {
                    errors.add("Bank transfer amount exceeds limit");
                }
                break;
                
            case DIGITAL_WALLET:
                // Digital wallets have specific validation
                if (transaction.getAmount().compareTo(new BigDecimal("2000")) > 0) {
                    errors.add("Digital wallet transaction exceeds limit");
                }
                break;
                
            case CRYPTOCURRENCY:
                // Crypto has special requirements
                if (transaction.getAmount().compareTo(new BigDecimal("10000")) > 0) {
                    errors.add("Cryptocurrency transaction requires enhanced verification");
                }
                break;
        }
    }
    
    private void validateProcessingFee(Transaction transaction, List<String> errors) {
        if (transaction.getProcessingFee() != null) {
            if (transaction.getProcessingFee().compareTo(BigDecimal.ZERO) < 0) {
                errors.add("Processing fee cannot be negative");
            }
            
            // Fee should not exceed 10% of transaction amount
            BigDecimal maxFee = transaction.getAmount().multiply(new BigDecimal("0.10"));
            if (transaction.getProcessingFee().compareTo(maxFee) > 0) {
                errors.add("Processing fee exceeds 10% of transaction amount");
            }
        }
    }
    
    private void validateCurrencyConversion(Transaction transaction, List<String> errors) {
        if (transaction.getOriginalCurrency() != null && 
            !transaction.getOriginalCurrency().equals(transaction.getCurrency())) {
            
            if (transaction.getExchangeRate() == null || 
                transaction.getExchangeRate().compareTo(BigDecimal.ZERO) <= 0) {
                errors.add("Valid exchange rate is required for currency conversion");
            }
            
            if (transaction.getOriginalAmount() == null) {
                errors.add("Original amount is required for currency conversion");
            }
            
            // Validate exchange rate is within reasonable bounds
            if (transaction.getExchangeRate() != null) {
                if (transaction.getExchangeRate().compareTo(new BigDecimal("0.001")) < 0 ||
                    transaction.getExchangeRate().compareTo(new BigDecimal("1000")) > 0) {
                    errors.add("Exchange rate is outside acceptable range");
                }
            }
        }
    }
    
    private void validateRiskRules(Transaction transaction, List<String> errors) {
        // High-value transaction validation
        if (transaction.getAmount().compareTo(new BigDecimal("25000")) > 0) {
            errors.add("High-value transaction requires manual approval");
        }
        
        // International transaction validation
        if (transaction.getOriginalCurrency() != null && 
            !transaction.getOriginalCurrency().equals("USD")) {
            if (transaction.getAmount().compareTo(new BigDecimal("10000")) > 0) {
                errors.add("International high-value transaction requires additional verification");
            }
        }
        
        // Validate against known risk patterns
        if (transaction.getMerchantId() != null && 
            isHighRiskMerchant(transaction.getMerchantId())) {
            errors.add("Transaction with high-risk merchant requires enhanced screening");
        }
    }
    
    private void validateTimeBasedRules(Transaction transaction, List<String> errors) {
        LocalDateTime now = LocalDateTime.now();
        
        // Night-time transaction limits (example: 10 PM to 6 AM)
        int hour = now.getHour();
        if ((hour >= 22 || hour < 6) && 
            transaction.getAmount().compareTo(new BigDecimal("5000")) > 0) {
            errors.add("Large transactions are restricted during night hours");
        }
        
        // Weekend transaction limits
        if ((now.getDayOfWeek().getValue() >= 6) && 
            transaction.getAmount().compareTo(new BigDecimal("15000")) > 0) {
            errors.add("Large transactions are restricted during weekends");
        }
    }
    
    private void validateMerchantRules(Transaction transaction, List<String> errors) {
        if (transaction.getMerchantId() == null) {
            return;
        }
        
        // Validate merchant category restrictions
        String merchantCategory = getMerchantCategory(transaction.getMerchantId());
        if ("GAMBLING".equals(merchantCategory) && 
            transaction.getAmount().compareTo(new BigDecimal("1000")) > 0) {
            errors.add("Gambling transactions have lower limits");
        }
        
        if ("CASH_ADVANCE".equals(merchantCategory)) {
            errors.add("Cash advance transactions require special approval");
        }
    }
    
    private boolean isHighRiskMerchant(String merchantId) {
        // This would typically check against a database of high-risk merchants
        // For demo purposes, using a simple check
        return merchantId.startsWith("RISK_") || merchantId.contains("SUSPICIOUS");
    }
    
    private String getMerchantCategory(String merchantId) {
        // This would typically lookup merchant category from database
        // For demo purposes, using simple pattern matching
        if (merchantId.contains("CASINO") || merchantId.contains("BET")) {
            return "GAMBLING";
        }
        if (merchantId.contains("CASH") || merchantId.contains("ATM")) {
            return "CASH_ADVANCE";
        }
        return "GENERAL";
    }
    
    /**
     * Check if currency code is valid
     */
    private boolean isValidCurrency(String currency) {
        // List of supported currencies
        String[] supportedCurrencies = {
            "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR", "BRL"
        };
        
        for (String supported : supportedCurrencies) {
            if (supported.equals(currency.toUpperCase())) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Validate transaction request DTO
     */
    public ValidationResult validateTransactionRequest(TransactionValidationDto request) {
        List<String> errors = new ArrayList<>();
        
        if (request.getAmount() == null || request.getAmount().compareTo(BigDecimal.ZERO) <= 0) {
            errors.add("Valid transaction amount is required");
        }
        
        if (request.getCurrency() == null || request.getCurrency().trim().isEmpty()) {
            errors.add("Currency is required");
        }
        
        if (request.getUserId() == null) {
            errors.add("User ID is required");
        }
        
        if (request.getMerchantId() == null || request.getMerchantId().trim().isEmpty()) {
            errors.add("Merchant ID is required");
        }
        
        return new ValidationResult(errors.isEmpty(), errors);
    }
}
