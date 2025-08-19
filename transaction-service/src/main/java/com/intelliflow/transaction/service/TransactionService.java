package com.intelliflow.transaction.service;

import com.intelliflow.shared.messaging.EventMessage;
import com.intelliflow.transaction.dto.*;
import com.intelliflow.transaction.entity.Transaction;
import com.intelliflow.transaction.entity.TransactionStatus;
import com.intelliflow.transaction.entity.TransactionType;
import com.intelliflow.transaction.repository.TransactionRepository;
import com.intelliflow.transaction.statemachine.TransactionEvent;
import com.intelliflow.transaction.statemachine.TransactionState;
import com.intelliflow.transaction.messaging.TransactionEventPublisher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.scheduling.annotation.Async;
import org.springframework.statemachine.StateMachine;
import org.springframework.statemachine.StateMachineFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

@Service
@Transactional
public class TransactionService {
    
    private static final Logger logger = LoggerFactory.getLogger(TransactionService.class);
    
    private final TransactionRepository transactionRepository;
    private final TransactionValidationService validationService;
    private final FraudCheckService fraudCheckService;
    private final BalanceVerificationService balanceService;
    private final PaymentProcessingService paymentService;
    private final TransactionEventPublisher eventPublisher;
    private final StateMachineFactory<TransactionState, TransactionEvent> stateMachineFactory;
    
    @Autowired
    public TransactionService(
            TransactionRepository transactionRepository,
            TransactionValidationService validationService,
            FraudCheckService fraudCheckService,
            BalanceVerificationService balanceService,
            PaymentProcessingService paymentService,
            TransactionEventPublisher eventPublisher,
            StateMachineFactory<TransactionState, TransactionEvent> stateMachineFactory) {
        this.transactionRepository = transactionRepository;
        this.validationService = validationService;
        this.fraudCheckService = fraudCheckService;
        this.balanceService = balanceService;
        this.paymentService = paymentService;
        this.eventPublisher = eventPublisher;
        this.stateMachineFactory = stateMachineFactory;
    }
    
    /**
     * Create a new transaction
     */
    public CompletableFuture<TransactionResponseDto> createTransaction(CreateTransactionDto createDto) {
        return CompletableFuture.supplyAsync(() -> {
            logger.info("Creating new transaction for user: {}, amount: {}", 
                createDto.getUserId(), createDto.getAmount());
            
            try {
                // Create transaction entity
                Transaction transaction = buildTransaction(createDto);
                transaction = transactionRepository.save(transaction);
                
                // Publish transaction created event
                eventPublisher.publishTransactionCreated(transaction);
                
                // Start async processing
                processTransactionAsync(transaction.getId());
                
                logger.info("Transaction created with ID: {} and reference: {}", 
                    transaction.getId(), transaction.getReferenceNumber());
                
                return mapToTransactionResponseDto(transaction);
                
            } catch (Exception e) {
                logger.error("Failed to create transaction for user: {}", createDto.getUserId(), e);
                throw new RuntimeException("Transaction creation failed", e);
            }
        });
    }
    
    /**
     * Process transaction asynchronously
     */
    @Async
    public CompletableFuture<Void> processTransactionAsync(Long transactionId) {
        return CompletableFuture.runAsync(() -> {
            logger.info("Starting async processing for transaction: {}", transactionId);
            
            try {
                Transaction transaction = getTransactionById(transactionId);
                
                // Create and start state machine
                StateMachine<TransactionState, TransactionEvent> stateMachine = 
                    stateMachineFactory.getStateMachine(transaction.getReferenceNumber());
                
                // Set transaction ID in state machine context
                stateMachine.getExtendedState().getVariables().put("transactionId", transactionId);
                
                // Start the state machine
                stateMachine.start();
                
                // Begin validation
                performValidation(transaction, stateMachine);
                
            } catch (Exception e) {
                logger.error("Error in async transaction processing for ID: {}", transactionId, e);
                handleTransactionError(transactionId, e.getMessage());
            }
        });
    }
    
    /**
     * Perform transaction validation
     */
    private void performValidation(Transaction transaction, 
                                 StateMachine<TransactionState, TransactionEvent> stateMachine) {
        validationService.validateTransaction(transaction)
            .thenAccept(validationResult -> {
                if (validationResult.isValid()) {
                    stateMachine.sendEvent(TransactionEvent.VALIDATION_SUCCESS);
                } else {
                    stateMachine.getExtendedState().getVariables()
                        .put("failureReason", String.join(", ", validationResult.getErrors()));
                    stateMachine.sendEvent(TransactionEvent.VALIDATION_FAILED);
                }
            })
            .exceptionally(throwable -> {
                logger.error("Validation failed for transaction: {}", transaction.getId(), throwable);
                stateMachine.getExtendedState().getVariables()
                    .put("failureReason", "Validation service error: " + throwable.getMessage());
                stateMachine.sendEvent(TransactionEvent.VALIDATION_FAILED);
                return null;
            });
    }
    
    /**
     * Get transaction by ID
     */
    @Transactional(readOnly = true)
    public TransactionResponseDto getTransaction(Long transactionId) {
        Transaction transaction = getTransactionById(transactionId);
        return mapToTransactionResponseDto(transaction);
    }
    
    /**
     * Get transaction by reference number
     */
    @Transactional(readOnly = true)
    public TransactionResponseDto getTransactionByReference(String referenceNumber) {
        Transaction transaction = transactionRepository.findByReferenceNumber(referenceNumber)
            .orElseThrow(() -> new RuntimeException("Transaction not found: " + referenceNumber));
        return mapToTransactionResponseDto(transaction);
    }
    
    /**
     * Get transactions for user
     */
    @Transactional(readOnly = true)
    public Page<TransactionResponseDto> getUserTransactions(Long userId, Pageable pageable) {
        Page<Transaction> transactions = transactionRepository.findByUserId(userId, pageable);
        return transactions.map(this::mapToTransactionResponseDto);
    }
    
    /**
     * Update transaction status
     */
    public CompletableFuture<TransactionResponseDto> updateTransactionStatus(
            Long transactionId, UpdateTransactionStatusDto updateDto) {
        
        return CompletableFuture.supplyAsync(() -> {
            logger.info("Updating transaction status for ID: {} to status: {}", 
                transactionId, updateDto.getStatus());
            
            Transaction transaction = getTransactionById(transactionId);
            
            // Validate status transition
            if (!canTransitionToStatus(transaction.getStatus(), updateDto.getStatus())) {
                throw new IllegalStateException("Invalid status transition from " + 
                    transaction.getStatus() + " to " + updateDto.getStatus());
            }
            
            // Update status and related fields
            transaction.setStatus(updateDto.getStatus());
            if (updateDto.getFailureReason() != null) {
                transaction.setFailureReason(updateDto.getFailureReason());
            }
            
            if (updateDto.getStatus() == TransactionStatus.SUCCESS) {
                transaction.setProcessedAt(LocalDateTime.now());
            } else if (updateDto.getStatus() == TransactionStatus.FAILED) {
                transaction.setFailedAt(LocalDateTime.now());
            }
            
            transaction = transactionRepository.save(transaction);
            
            // Publish status change event
            eventPublisher.publishTransactionStatusChanged(transaction);
            
            logger.info("Transaction status updated for ID: {}", transactionId);
            return mapToTransactionResponseDto(transaction);
        });
    }
    
    /**
     * Process refund
     */
    public CompletableFuture<TransactionResponseDto> processRefund(Long transactionId, RefundRequestDto refundDto) {
        return CompletableFuture.supplyAsync(() -> {
            logger.info("Processing refund for transaction: {}, amount: {}", 
                transactionId, refundDto.getAmount());
            
            Transaction originalTransaction = getTransactionById(transactionId);
            
            // Validate refund eligibility
            validateRefundEligibility(originalTransaction, refundDto.getAmount());
            
            // Create refund transaction
            Transaction refundTransaction = createRefundTransaction(originalTransaction, refundDto);
            refundTransaction = transactionRepository.save(refundTransaction);
            
            // Process refund asynchronously
            processRefundAsync(refundTransaction.getId(), originalTransaction.getId());
            
            logger.info("Refund transaction created with ID: {}", refundTransaction.getId());
            return mapToTransactionResponseDto(refundTransaction);
        });
    }
    
    /**
     * Process refund asynchronously
     */
    @Async
    public void processRefundAsync(Long refundTransactionId, Long originalTransactionId) {
        try {
            Transaction refundTransaction = getTransactionById(refundTransactionId);
            Transaction originalTransaction = getTransactionById(originalTransactionId);
            
            // Process refund with payment service
            paymentService.processRefund(refundTransaction, originalTransaction)
                .thenAccept(refundResult -> {
                    if (refundResult.isSuccess()) {
                        refundTransaction.setStatus(TransactionStatus.SUCCESS);
                        refundTransaction.setProcessedAt(LocalDateTime.now());
                        
                        // Update original transaction status
                        originalTransaction.setStatus(TransactionStatus.REFUNDED);
                        
                        transactionRepository.save(refundTransaction);
                        transactionRepository.save(originalTransaction);
                        
                        // Publish refund completed event
                        eventPublisher.publishRefundCompleted(refundTransaction);
                        
                    } else {
                        refundTransaction.setStatus(TransactionStatus.FAILED);
                        refundTransaction.setFailureReason(refundResult.getFailureReason());
                        refundTransaction.setFailedAt(LocalDateTime.now());
                        transactionRepository.save(refundTransaction);
                        
                        // Publish refund failed event
                        eventPublisher.publishRefundFailed(refundTransaction);
                    }
                })
                .exceptionally(throwable -> {
                    logger.error("Refund processing failed for transaction: {}", refundTransactionId, throwable);
                    handleTransactionError(refundTransactionId, "Refund processing error: " + throwable.getMessage());
                    return null;
                });
                
        } catch (Exception e) {
            logger.error("Error in async refund processing for ID: {}", refundTransactionId, e);
            handleTransactionError(refundTransactionId, e.getMessage());
        }
    }
    
    /**
     * Cancel transaction
     */
    public CompletableFuture<TransactionResponseDto> cancelTransaction(Long transactionId, String reason) {
        return CompletableFuture.supplyAsync(() -> {
            logger.info("Cancelling transaction: {} with reason: {}", transactionId, reason);
            
            Transaction transaction = getTransactionById(transactionId);
            
            // Validate cancellation eligibility
            if (transaction.isInFinalState()) {
                throw new IllegalStateException("Cannot cancel transaction in final state: " + 
                    transaction.getCurrentState());
            }
            
            // Cancel transaction
            transaction.setStatus(TransactionStatus.CANCELLED);
            transaction.setCurrentState(TransactionState.CANCELLED);
            transaction.setFailureReason(reason);
            transaction = transactionRepository.save(transaction);
            
            // Publish cancellation event
            eventPublisher.publishTransactionCancelled(transaction);
            
            logger.info("Transaction cancelled: {}", transactionId);
            return mapToTransactionResponseDto(transaction);
        });
    }
    
    /**
     * Search transactions with filters
     */
    @Transactional(readOnly = true)
    public Page<TransactionResponseDto> searchTransactions(TransactionSearchDto searchDto, Pageable pageable) {
        // This would typically use a specification or criteria API for complex searches
        // For now, implementing basic search by user ID and date range
        
        if (searchDto.getUserId() != null && searchDto.getFromDate() != null && searchDto.getToDate() != null) {
            Page<Transaction> transactions = transactionRepository.findByUserIdAndCreatedAtBetween(
                searchDto.getUserId(), searchDto.getFromDate(), searchDto.getToDate(), pageable);
            return transactions.map(this::mapToTransactionResponseDto);
        } else if (searchDto.getUserId() != null) {
            Page<Transaction> transactions = transactionRepository.findByUserId(searchDto.getUserId(), pageable);
            return transactions.map(this::mapToTransactionResponseDto);
        } else if (searchDto.getStatus() != null) {
            Page<Transaction> transactions = transactionRepository.findByStatus(searchDto.getStatus(), pageable);
            return transactions.map(this::mapToTransactionResponseDto);
        } else {
            // Return all transactions if no specific filters
            Page<Transaction> transactions = transactionRepository.findAll(pageable);
            return transactions.map(this::mapToTransactionResponseDto);
        }
    }
    
    /**
     * Get transaction statistics
     */
    @Transactional(readOnly = true)
    public TransactionStatsDto getTransactionStats(LocalDateTime fromDate) {
        TransactionStatsDto stats = new TransactionStatsDto();
        
        // Get counts by status
        List<Object[]> statusCounts = transactionRepository.getTransactionCountByStatus(fromDate);
        for (Object[] row : statusCounts) {
            TransactionStatus status = (TransactionStatus) row[0];
            Long count = (Long) row[1];
            stats.addStatusCount(status, count);
        }
        
        // Get total amounts by currency
        List<Object[]> amountsByCurrency = transactionRepository.getTotalAmountByCurrency(fromDate);
        for (Object[] row : amountsByCurrency) {
            String currency = (String) row[0];
            BigDecimal amount = (BigDecimal) row[1];
            stats.addCurrencyAmount(currency, amount);
        }
        
        return stats;
    }
    
    // Private helper methods
    
    private Transaction buildTransaction(CreateTransactionDto createDto) {
        Transaction transaction = new Transaction();
        transaction.setUserId(createDto.getUserId());
        transaction.setType(createDto.getType());
        transaction.setAmount(createDto.getAmount());
        transaction.setCurrency(createDto.getCurrency());
        transaction.setDescription(createDto.getDescription());
        transaction.setMerchantId(createDto.getMerchantId());
        transaction.setMerchantName(createDto.getMerchantName());
        transaction.setPaymentMethod(createDto.getPaymentMethod());
        transaction.setPaymentInstrumentId(createDto.getPaymentInstrumentId());
        transaction.setCorrelationId(UUID.randomUUID().toString());
        transaction.setTransactionSource(createDto.getSource() != null ? createDto.getSource() : "API");
        transaction.setChannel(createDto.getChannel());
        transaction.setInitiatedBy(createDto.getInitiatedBy());
        transaction.generateReferenceNumber();
        
        return transaction;
    }
    
    private Transaction getTransactionById(Long transactionId) {
        return transactionRepository.findById(transactionId)
            .orElseThrow(() -> new RuntimeException("Transaction not found: " + transactionId));
    }
    
    private boolean canTransitionToStatus(TransactionStatus currentStatus, TransactionStatus newStatus) {
        // Define valid status transitions
        return switch (currentStatus) {
            case PENDING -> newStatus == TransactionStatus.PROCESSING || 
                          newStatus == TransactionStatus.FAILED || 
                          newStatus == TransactionStatus.CANCELLED;
            case PROCESSING -> newStatus == TransactionStatus.SUCCESS || 
                             newStatus == TransactionStatus.FAILED;
            case SUCCESS -> newStatus == TransactionStatus.REFUNDED || 
                          newStatus == TransactionStatus.DISPUTED;
            default -> false;
        };
    }
    
    private void validateRefundEligibility(Transaction transaction, BigDecimal refundAmount) {
        if (transaction.getStatus() != TransactionStatus.SUCCESS) {
            throw new IllegalStateException("Can only refund successful transactions");
        }
        
        if (refundAmount.compareTo(transaction.getAmount()) > 0) {
            throw new IllegalArgumentException("Refund amount cannot exceed original transaction amount");
        }
        
        if (refundAmount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new IllegalArgumentException("Refund amount must be positive");
        }
    }
    
    private Transaction createRefundTransaction(Transaction originalTransaction, RefundRequestDto refundDto) {
        Transaction refundTransaction = new Transaction();
        refundTransaction.setUserId(originalTransaction.getUserId());
        refundTransaction.setType(TransactionType.REFUND);
        refundTransaction.setAmount(refundDto.getAmount());
        refundTransaction.setCurrency(originalTransaction.getCurrency());
        refundTransaction.setDescription("Refund for transaction: " + originalTransaction.getReferenceNumber());
        refundTransaction.setMerchantId(originalTransaction.getMerchantId());
        refundTransaction.setMerchantName(originalTransaction.getMerchantName());
        refundTransaction.setPaymentMethod(originalTransaction.getPaymentMethod());
        refundTransaction.setExternalTransactionId(originalTransaction.getReferenceNumber());
        refundTransaction.setParentTransactionId(originalTransaction.getId());
        refundTransaction.setCorrelationId(originalTransaction.getCorrelationId());
        refundTransaction.setTransactionSource("REFUND");
        refundTransaction.generateReferenceNumber();
        
        return refundTransaction;
    }
    
    private void handleTransactionError(Long transactionId, String errorMessage) {
        try {
            Transaction transaction = getTransactionById(transactionId);
            transaction.setStatus(TransactionStatus.FAILED);
            transaction.setFailureReason(errorMessage);
            transaction.setFailedAt(LocalDateTime.now());
            transactionRepository.save(transaction);
            
            // Publish error event
            eventPublisher.publishTransactionFailed(transaction);
            
        } catch (Exception e) {
            logger.error("Failed to handle transaction error for ID: {}", transactionId, e);
        }
    }
    
    private TransactionResponseDto mapToTransactionResponseDto(Transaction transaction) {
        TransactionResponseDto dto = new TransactionResponseDto();
        dto.setId(transaction.getId());
        dto.setReferenceNumber(transaction.getReferenceNumber());
        dto.setUserId(transaction.getUserId());
        dto.setType(transaction.getType());
        dto.setStatus(transaction.getStatus());
        dto.setCurrentState(transaction.getCurrentState());
        dto.setAmount(transaction.getAmount());
        dto.setCurrency(transaction.getCurrency());
        dto.setDescription(transaction.getDescription());
        dto.setMerchantName(transaction.getMerchantName());
        dto.setPaymentMethod(transaction.getPaymentMethod());
        dto.setProcessingFee(transaction.getProcessingFee());
        dto.setRiskScore(transaction.getRiskScore());
        dto.setFraudCheckStatus(transaction.getFraudCheckStatus());
        dto.setProcessedAt(transaction.getProcessedAt());
        dto.setFailureReason(transaction.getFailureReason());
        dto.setCorrelationId(transaction.getCorrelationId());
        dto.setCreatedAt(transaction.getCreatedAt());
        dto.setUpdatedAt(transaction.getUpdatedAt());
        
        return dto;
    }
}
