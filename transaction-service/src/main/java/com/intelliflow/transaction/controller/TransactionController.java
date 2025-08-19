package com.intelliflow.transaction.controller;

import com.intelliflow.shared.dto.ApiResponse;
import com.intelliflow.transaction.dto.*;
import com.intelliflow.transaction.service.TransactionService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.concurrent.CompletableFuture;

/**
 * REST Controller for transaction operations
 */
@RestController
@RequestMapping("/api/v1/transactions")
@Validated
@CrossOrigin(origins = "*", maxAge = 3600)
public class TransactionController {
    
    private static final Logger logger = LoggerFactory.getLogger(TransactionController.class);
    
    private final TransactionService transactionService;
    
    @Autowired
    public TransactionController(TransactionService transactionService) {
        this.transactionService = transactionService;
    }
    
    /**
     * Create a new transaction
     */
    @PostMapping
    @PreAuthorize("hasRole('USER') or hasRole('ADMIN')")
    public CompletableFuture<ResponseEntity<ApiResponse<TransactionResponseDto>>> createTransaction(
            @Valid @RequestBody CreateTransactionDto createDto,
            @RequestHeader(value = "X-User-Id", required = false) String userId,
            @RequestHeader(value = "X-Correlation-ID", required = false) String correlationId) {
        
        logger.info("Creating transaction for user: {}, amount: {}, correlation: {}", 
            createDto.getUserId(), createDto.getAmount(), correlationId);
        
        return transactionService.createTransaction(createDto)
            .thenApply(transactionDto -> {
                logger.info("Transaction created successfully: {}", transactionDto.getReferenceNumber());
                return ResponseEntity.status(HttpStatus.CREATED)
                    .body(ApiResponse.success(transactionDto, "Transaction created successfully"));
            })
            .exceptionally(throwable -> {
                logger.error("Failed to create transaction for user: {}", createDto.getUserId(), throwable);
                return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(ApiResponse.error("Transaction creation failed: " + throwable.getMessage()));
            });
    }
    
    /**
     * Get transaction by ID
     */
    @GetMapping("/{transactionId}")
    @PreAuthorize("hasRole('USER') or hasRole('ADMIN')")
    public ResponseEntity<ApiResponse<TransactionResponseDto>> getTransaction(
            @PathVariable @NotNull Long transactionId,
            @RequestHeader(value = "X-User-Id", required = false) String userId) {
        
        logger.debug("Retrieving transaction: {} for user: {}", transactionId, userId);
        
        try {
            TransactionResponseDto transaction = transactionService.getTransaction(transactionId);
            
            // Check if user has permission to view this transaction
            if (userId != null && !userId.equals(transaction.getUserId().toString()) && 
                !hasAdminRole()) {
                logger.warn("Unauthorized access attempt for transaction: {} by user: {}", 
                    transactionId, userId);
                return ResponseEntity.status(HttpStatus.FORBIDDEN)
                    .body(ApiResponse.error("Access denied"));
            }
            
            logger.debug("Transaction retrieved successfully: {}", transactionId);
            return ResponseEntity.ok(ApiResponse.success(transaction, "Transaction retrieved successfully"));
            
        } catch (RuntimeException e) {
            logger.error("Error retrieving transaction: {}", transactionId, e);
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                .body(ApiResponse.error("Transaction not found"));
        }
    }
    
    /**
     * Get transaction by reference number
     */
    @GetMapping("/reference/{referenceNumber}")
    @PreAuthorize("hasRole('USER') or hasRole('ADMIN')")
    public ResponseEntity<ApiResponse<TransactionResponseDto>> getTransactionByReference(
            @PathVariable String referenceNumber,
            @RequestHeader(value = "X-User-Id", required = false) String userId) {
        
        logger.debug("Retrieving transaction by reference: {} for user: {}", referenceNumber, userId);
        
        try {
            TransactionResponseDto transaction = transactionService.getTransactionByReference(referenceNumber);
            
            // Check user permission
            if (userId != null && !userId.equals(transaction.getUserId().toString()) && 
                !hasAdminRole()) {
                logger.warn("Unauthorized access attempt for transaction reference: {} by user: {}", 
                    referenceNumber, userId);
                return ResponseEntity.status(HttpStatus.FORBIDDEN)
                    .body(ApiResponse.error("Access denied"));
            }
            
            logger.debug("Transaction retrieved by reference successfully: {}", referenceNumber);
            return ResponseEntity.ok(ApiResponse.success(transaction, "Transaction retrieved successfully"));
            
        } catch (RuntimeException e) {
            logger.error("Error retrieving transaction by reference: {}", referenceNumber, e);
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                .body(ApiResponse.error("Transaction not found"));
        }
    }
    
    /**
     * Get user transactions with pagination
     */
    @GetMapping("/user/{userId}")
    @PreAuthorize("hasRole('USER') or hasRole('ADMIN')")
    public ResponseEntity<ApiResponse<Page<TransactionResponseDto>>> getUserTransactions(
            @PathVariable Long userId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(defaultValue = "createdAt") String sortBy,
            @RequestParam(defaultValue = "desc") String sortDir,
            @RequestHeader(value = "X-User-Id", required = false) String requestingUserId) {
        
        logger.debug("Retrieving transactions for user: {}, page: {}, size: {}", userId, page, size);
        
        // Check user permission
        if (requestingUserId != null && !requestingUserId.equals(userId.toString()) && 
            !hasAdminRole()) {
            logger.warn("Unauthorized access attempt for user transactions: {} by user: {}", 
                userId, requestingUserId);
            return ResponseEntity.status(HttpStatus.FORBIDDEN)
                .body(ApiResponse.error("Access denied"));
        }
        
        try {
            Sort sort = Sort.by(Sort.Direction.fromString(sortDir), sortBy);
            Pageable pageable = PageRequest.of(page, size, sort);
            
            Page<TransactionResponseDto> transactions = transactionService.getUserTransactions(userId, pageable);
            
            logger.debug("Retrieved {} transactions for user: {}", transactions.getNumberOfElements(), userId);
            return ResponseEntity.ok(ApiResponse.success(transactions, "Transactions retrieved successfully"));
            
        } catch (Exception e) {
            logger.error("Error retrieving transactions for user: {}", userId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(ApiResponse.error("Error retrieving transactions"));
        }
    }
    
    /**
     * Search transactions with filters
     */
    @PostMapping("/search")
    @PreAuthorize("hasRole('ADMIN') or hasRole('OPERATOR')")
    public ResponseEntity<ApiResponse<Page<TransactionResponseDto>>> searchTransactions(
            @RequestBody TransactionSearchDto searchDto,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(defaultValue = "createdAt") String sortBy,
            @RequestParam(defaultValue = "desc") String sortDir) {
        
        logger.debug("Searching transactions with filters: {}", searchDto);
        
        try {
            Sort sort = Sort.by(Sort.Direction.fromString(sortDir), sortBy);
            Pageable pageable = PageRequest.of(page, size, sort);
            
            Page<TransactionResponseDto> transactions = transactionService.searchTransactions(searchDto, pageable);
            
            logger.debug("Found {} transactions matching search criteria", transactions.getNumberOfElements());
            return ResponseEntity.ok(ApiResponse.success(transactions, "Search completed successfully"));
            
        } catch (Exception e) {
            logger.error("Error searching transactions", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(ApiResponse.error("Error searching transactions"));
        }
    }
    
    /**
     * Update transaction status
     */
    @PutMapping("/{transactionId}/status")
    @PreAuthorize("hasRole('ADMIN') or hasRole('OPERATOR')")
    public CompletableFuture<ResponseEntity<ApiResponse<TransactionResponseDto>>> updateTransactionStatus(
            @PathVariable Long transactionId,
            @Valid @RequestBody UpdateTransactionStatusDto updateDto,
            @RequestHeader(value = "X-User-Id", required = false) String userId) {
        
        logger.info("Updating transaction status for: {} to: {} by user: {}", 
            transactionId, updateDto.getStatus(), userId);
        
        return transactionService.updateTransactionStatus(transactionId, updateDto)
            .thenApply(transactionDto -> {
                logger.info("Transaction status updated successfully: {}", transactionId);
                return ResponseEntity.ok(ApiResponse.success(transactionDto, "Transaction status updated"));
            })
            .exceptionally(throwable -> {
                logger.error("Failed to update transaction status: {}", transactionId, throwable);
                if (throwable.getCause() instanceof IllegalStateException) {
                    return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                        .body(ApiResponse.error("Invalid status transition: " + throwable.getMessage()));
                }
                return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(ApiResponse.error("Status update failed: " + throwable.getMessage()));
            });
    }
    
    /**
     * Process refund
     */
    @PostMapping("/{transactionId}/refund")
    @PreAuthorize("hasRole('ADMIN') or hasRole('OPERATOR')")
    public CompletableFuture<ResponseEntity<ApiResponse<TransactionResponseDto>>> processRefund(
            @PathVariable Long transactionId,
            @Valid @RequestBody RefundRequestDto refundDto,
            @RequestHeader(value = "X-User-Id", required = false) String userId) {
        
        logger.info("Processing refund for transaction: {}, amount: {} by user: {}", 
            transactionId, refundDto.getAmount(), userId);
        
        return transactionService.processRefund(transactionId, refundDto)
            .thenApply(refundTransaction -> {
                logger.info("Refund processed successfully: {}", refundTransaction.getReferenceNumber());
                return ResponseEntity.status(HttpStatus.CREATED)
                    .body(ApiResponse.success(refundTransaction, "Refund processed successfully"));
            })
            .exceptionally(throwable -> {
                logger.error("Failed to process refund for transaction: {}", transactionId, throwable);
                if (throwable.getCause() instanceof IllegalStateException || 
                    throwable.getCause() instanceof IllegalArgumentException) {
                    return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                        .body(ApiResponse.error("Refund failed: " + throwable.getMessage()));
                }
                return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(ApiResponse.error("Refund processing failed: " + throwable.getMessage()));
            });
    }
    
    /**
     * Cancel transaction
     */
    @PostMapping("/{transactionId}/cancel")
    @PreAuthorize("hasRole('ADMIN') or hasRole('OPERATOR')")
    public CompletableFuture<ResponseEntity<ApiResponse<TransactionResponseDto>>> cancelTransaction(
            @PathVariable Long transactionId,
            @RequestParam String reason,
            @RequestHeader(value = "X-User-Id", required = false) String userId) {
        
        logger.info("Cancelling transaction: {} with reason: {} by user: {}", 
            transactionId, reason, userId);
        
        return transactionService.cancelTransaction(transactionId, reason)
            .thenApply(transactionDto -> {
                logger.info("Transaction cancelled successfully: {}", transactionId);
                return ResponseEntity.ok(ApiResponse.success(transactionDto, "Transaction cancelled"));
            })
            .exceptionally(throwable -> {
                logger.error("Failed to cancel transaction: {}", transactionId, throwable);
                if (throwable.getCause() instanceof IllegalStateException) {
                    return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                        .body(ApiResponse.error("Cannot cancel transaction: " + throwable.getMessage()));
                }
                return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(ApiResponse.error("Cancellation failed: " + throwable.getMessage()));
            });
    }
    
    /**
     * Get transaction statistics
     */
    @GetMapping("/stats")
    @PreAuthorize("hasRole('ADMIN') or hasRole('ANALYTICS')")
    public ResponseEntity<ApiResponse<TransactionStatsDto>> getTransactionStats(
            @RequestParam(required = false) String period) {
        
        logger.debug("Retrieving transaction statistics for period: {}", period);
        
        try {
            LocalDateTime fromDate = calculateFromDate(period);
            TransactionStatsDto stats = transactionService.getTransactionStats(fromDate);
            stats.calculateDerivedStats();
            
            logger.debug("Transaction statistics retrieved successfully");
            return ResponseEntity.ok(ApiResponse.success(stats, "Statistics retrieved successfully"));
            
        } catch (Exception e) {
            logger.error("Error retrieving transaction statistics", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(ApiResponse.error("Error retrieving statistics"));
        }
    }
    
    /**
     * Health check endpoint
     */
    @GetMapping("/health")
    public ResponseEntity<ApiResponse<String>> healthCheck() {
        logger.debug("Transaction service health check");
        return ResponseEntity.ok(ApiResponse.success("OK", "Transaction service is healthy"));
    }
    
    // Private helper methods
    
    private boolean hasAdminRole() {
        // In a real implementation, this would check the user's roles from Spring Security context
        // For now, returning false as a placeholder
        return false;
    }
    
    private LocalDateTime calculateFromDate(String period) {
        LocalDateTime now = LocalDateTime.now();
        
        if (period == null || period.isEmpty()) {
            period = "24h";
        }
        
        return switch (period.toLowerCase()) {
            case "1h" -> now.minusHours(1);
            case "24h" -> now.minusDays(1);
            case "7d" -> now.minusDays(7);
            case "30d" -> now.minusDays(30);
            case "90d" -> now.minusDays(90);
            default -> now.minusDays(1);
        };
    }
}
