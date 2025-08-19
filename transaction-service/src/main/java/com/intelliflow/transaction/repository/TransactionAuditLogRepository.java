package com.intelliflow.transaction.repository;

import com.intelliflow.transaction.entity.TransactionAuditLog;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface TransactionAuditLogRepository extends JpaRepository<TransactionAuditLog, Long> {
    
    List<TransactionAuditLog> findByTransactionIdOrderByTimestampDesc(Long transactionId);
    
    Page<TransactionAuditLog> findByTransactionIdOrderByTimestampDesc(Long transactionId, Pageable pageable);
    
    List<TransactionAuditLog> findByReferenceNumberOrderByTimestampDesc(String referenceNumber);
    
    List<TransactionAuditLog> findByCorrelationIdOrderByTimestampDesc(String correlationId);
    
    @Query("SELECT a FROM TransactionAuditLog a WHERE a.userId = :userId AND a.timestamp >= :fromDate ORDER BY a.timestamp DESC")
    List<TransactionAuditLog> findByUserIdAndTimestampAfter(@Param("userId") String userId, 
                                                           @Param("fromDate") LocalDateTime fromDate);
    
    @Query("SELECT a FROM TransactionAuditLog a WHERE a.action = :action AND a.timestamp >= :fromDate ORDER BY a.timestamp DESC")
    List<TransactionAuditLog> findByActionAndTimestampAfter(@Param("action") String action, 
                                                           @Param("fromDate") LocalDateTime fromDate);
    
    @Query("SELECT COUNT(a) FROM TransactionAuditLog a WHERE a.transactionId = :transactionId")
    long countByTransactionId(@Param("transactionId") Long transactionId);
}
