package com.intelliflow.transaction.repository;

import com.intelliflow.transaction.entity.Transaction;
import com.intelliflow.transaction.entity.TransactionStatus;
import com.intelliflow.transaction.entity.TransactionType;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface TransactionRepository extends JpaRepository<Transaction, Long> {
    
    // Basic finder methods
    Optional<Transaction> findByReferenceNumber(String referenceNumber);
    
    List<Transaction> findByUserId(Long userId);
    
    Page<Transaction> findByUserId(Long userId, Pageable pageable);
    
    // Status-based queries
    List<Transaction> findByStatus(TransactionStatus status);
    
    Page<Transaction> findByStatus(TransactionStatus status, Pageable pageable);
    
    List<Transaction> findByUserIdAndStatus(Long userId, TransactionStatus status);
    
    // Type-based queries
    List<Transaction> findByType(TransactionType type);
    
    List<Transaction> findByUserIdAndType(Long userId, TransactionType type);
    
    // Amount-based queries
    @Query("SELECT t FROM Transaction t WHERE t.amount >= :minAmount AND t.amount <= :maxAmount")
    List<Transaction> findByAmountRange(@Param("minAmount") BigDecimal minAmount, 
                                       @Param("maxAmount") BigDecimal maxAmount);
    
    @Query("SELECT t FROM Transaction t WHERE t.userId = :userId AND t.amount >= :minAmount")
    List<Transaction> findByUserIdAndAmountGreaterThan(@Param("userId") Long userId, 
                                                      @Param("minAmount") BigDecimal minAmount);
    
    // Date-based queries
    @Query("SELECT t FROM Transaction t WHERE t.createdAt >= :fromDate AND t.createdAt <= :toDate")
    List<Transaction> findByCreatedAtBetween(@Param("fromDate") LocalDateTime fromDate, 
                                           @Param("toDate") LocalDateTime toDate);
    
    @Query("SELECT t FROM Transaction t WHERE t.userId = :userId AND t.createdAt >= :fromDate AND t.createdAt <= :toDate")
    Page<Transaction> findByUserIdAndCreatedAtBetween(@Param("userId") Long userId,
                                                     @Param("fromDate") LocalDateTime fromDate,
                                                     @Param("toDate") LocalDateTime toDate,
                                                     Pageable pageable);
    
    // Fraud-related queries
    @Query("SELECT t FROM Transaction t WHERE t.riskScore >= :threshold")
    List<Transaction> findHighRiskTransactions(@Param("threshold") BigDecimal threshold);
    
    @Query("SELECT t FROM Transaction t WHERE t.fraudCheckStatus = 'REVIEW_REQUIRED'")
    List<Transaction> findTransactionsRequiringReview();
    
    // Merchant-related queries
    List<Transaction> findByMerchantId(String merchantId);
    
    @Query("SELECT t FROM Transaction t WHERE t.merchantCategory = :category AND t.createdAt >= :fromDate")
    List<Transaction> findByMerchantCategoryAndDateAfter(@Param("category") String category,
                                                        @Param("fromDate") LocalDateTime fromDate);
    
    // Payment method queries
    @Query("SELECT t FROM Transaction t WHERE t.paymentMethod = :paymentMethod AND t.userId = :userId")
    List<Transaction> findByUserIdAndPaymentMethod(@Param("userId") Long userId,
                                                  @Param("paymentMethod") String paymentMethod);
    
    // Failed transaction queries
    @Query("SELECT t FROM Transaction t WHERE t.status = 'FAILED' AND t.retryCount < t.maxRetries")
    List<Transaction> findFailedTransactionsEligibleForRetry();
    
    @Query("SELECT t FROM Transaction t WHERE t.userId = :userId AND t.status = 'FAILED' AND t.failedAt >= :fromDate")
    List<Transaction> findRecentFailedTransactionsByUser(@Param("userId") Long userId,
                                                        @Param("fromDate") LocalDateTime fromDate);
    
    // Aggregation queries
    @Query("SELECT SUM(t.amount) FROM Transaction t WHERE t.userId = :userId AND t.status = 'SUCCESS'")
    BigDecimal calculateTotalSuccessfulAmountByUser(@Param("userId") Long userId);
    
    @Query("SELECT SUM(t.amount) FROM Transaction t WHERE t.userId = :userId AND t.status = 'SUCCESS' AND t.createdAt >= :fromDate")
    BigDecimal calculateTotalAmountByUserSinceDate(@Param("userId") Long userId,
                                                  @Param("fromDate") LocalDateTime fromDate);
    
    @Query("SELECT COUNT(t) FROM Transaction t WHERE t.userId = :userId AND t.createdAt >= :fromDate")
    long countTransactionsByUserSinceDate(@Param("userId") Long userId,
                                         @Param("fromDate") LocalDateTime fromDate);
    
    // Update operations
    @Modifying
    @Query("UPDATE Transaction t SET t.status = :status, t.processedAt = :processedAt WHERE t.id = :id")
    int updateTransactionStatus(@Param("id") Long id,
                               @Param("status") TransactionStatus status,
                               @Param("processedAt") LocalDateTime processedAt);
    
    @Modifying
    @Query("UPDATE Transaction t SET t.fraudCheckStatus = :fraudStatus, t.riskScore = :riskScore WHERE t.id = :id")
    int updateFraudCheckResult(@Param("id") Long id,
                              @Param("fraudStatus") String fraudStatus,
                              @Param("riskScore") BigDecimal riskScore);
    
    @Modifying
    @Query("UPDATE Transaction t SET t.retryCount = t.retryCount + 1 WHERE t.id = :id")
    int incrementRetryCount(@Param("id") Long id);
    
    @Modifying
    @Query("UPDATE Transaction t SET t.notificationSent = :sent WHERE t.id = :id")
    int updateNotificationSent(@Param("id") Long id, @Param("sent") Boolean sent);
    
    // Statistics queries
    @Query("SELECT t.status, COUNT(t) FROM Transaction t WHERE t.createdAt >= :fromDate GROUP BY t.status")
    List<Object[]> getTransactionCountByStatus(@Param("fromDate") LocalDateTime fromDate);
    
    @Query("SELECT t.type, COUNT(t) FROM Transaction t WHERE t.createdAt >= :fromDate GROUP BY t.type")
    List<Object[]> getTransactionCountByType(@Param("fromDate") LocalDateTime fromDate);
    
    @Query("SELECT t.currency, SUM(t.amount) FROM Transaction t WHERE t.status = 'SUCCESS' AND t.createdAt >= :fromDate GROUP BY t.currency")
    List<Object[]> getTotalAmountByCurrency(@Param("fromDate") LocalDateTime fromDate);
    
    // Settlement queries
    @Query("SELECT t FROM Transaction t WHERE t.status = 'SUCCESS' AND t.settlementDate IS NULL")
    List<Transaction> findUnsettledTransactions();
    
    @Query("SELECT t FROM Transaction t WHERE t.settlementDate >= :fromDate AND t.settlementDate <= :toDate")
    List<Transaction> findTransactionsSettledBetween(@Param("fromDate") LocalDateTime fromDate,
                                                    @Param("toDate") LocalDateTime toDate);
}
