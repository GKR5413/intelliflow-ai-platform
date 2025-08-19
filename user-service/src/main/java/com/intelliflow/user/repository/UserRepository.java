package com.intelliflow.user.repository;

import com.intelliflow.user.entity.User;
import com.intelliflow.user.entity.UserStatus;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    
    // Basic finder methods
    Optional<User> findByUsername(String username);
    
    Optional<User> findByEmail(String email);
    
    Optional<User> findByUsernameOrEmail(String username, String email);
    
    boolean existsByUsername(String username);
    
    boolean existsByEmail(String email);
    
    // Status-based queries
    List<User> findByStatus(UserStatus status);
    
    Page<User> findByStatus(UserStatus status, Pageable pageable);
    
    @Query("SELECT u FROM User u WHERE u.status = :status AND u.lastLogin < :lastLoginBefore")
    List<User> findInactiveUsers(@Param("status") UserStatus status, 
                                @Param("lastLoginBefore") LocalDateTime lastLoginBefore);
    
    // Authentication-related queries
    @Query("SELECT u FROM User u WHERE u.email = :email AND u.status = :status")
    Optional<User> findByEmailAndStatus(@Param("email") String email, @Param("status") UserStatus status);
    
    @Query("SELECT u FROM User u WHERE u.username = :username AND u.status = :status")
    Optional<User> findByUsernameAndStatus(@Param("username") String username, @Param("status") UserStatus status);
    
    // Token-based queries
    Optional<User> findByPasswordResetToken(String token);
    
    Optional<User> findByEmailVerificationToken(String token);
    
    @Query("SELECT u FROM User u WHERE u.passwordResetToken = :token AND u.passwordResetExpires > :now")
    Optional<User> findByValidPasswordResetToken(@Param("token") String token, @Param("now") LocalDateTime now);
    
    @Query("SELECT u FROM User u WHERE u.emailVerificationToken = :token AND u.emailVerificationExpires > :now")
    Optional<User> findByValidEmailVerificationToken(@Param("token") String token, @Param("now") LocalDateTime now);
    
    // Security-related queries
    @Query("SELECT u FROM User u WHERE u.failedLoginAttempts >= :threshold")
    List<User> findUsersWithExcessiveFailedLogins(@Param("threshold") Integer threshold);
    
    // Update operations
    @Modifying
    @Query("UPDATE User u SET u.lastLogin = :lastLogin WHERE u.id = :userId")
    int updateLastLogin(@Param("userId") Long userId, @Param("lastLogin") LocalDateTime lastLogin);
    
    @Modifying
    @Query("UPDATE User u SET u.failedLoginAttempts = :attempts WHERE u.id = :userId")
    int updateFailedLoginAttempts(@Param("userId") Long userId, @Param("attempts") Integer attempts);
    
    @Modifying
    @Query("UPDATE User u SET u.status = :status WHERE u.id = :userId")
    int updateUserStatus(@Param("userId") Long userId, @Param("status") UserStatus status);
    
    @Modifying
    @Query("UPDATE User u SET u.emailVerified = :verified WHERE u.id = :userId")
    int updateEmailVerified(@Param("userId") Long userId, @Param("verified") Boolean verified);
    
    @Modifying
    @Query("UPDATE User u SET u.passwordResetToken = null, u.passwordResetExpires = null WHERE u.id = :userId")
    int clearPasswordResetToken(@Param("userId") Long userId);
    
    @Modifying
    @Query("UPDATE User u SET u.emailVerificationToken = null, u.emailVerificationExpires = null WHERE u.id = :userId")
    int clearEmailVerificationToken(@Param("userId") Long userId);
    
    // Search operations
    @Query("SELECT u FROM User u WHERE " +
           "LOWER(u.username) LIKE LOWER(CONCAT('%', :searchTerm, '%')) OR " +
           "LOWER(u.email) LIKE LOWER(CONCAT('%', :searchTerm, '%')) OR " +
           "LOWER(u.firstName) LIKE LOWER(CONCAT('%', :searchTerm, '%')) OR " +
           "LOWER(u.lastName) LIKE LOWER(CONCAT('%', :searchTerm, '%'))")
    Page<User> searchUsers(@Param("searchTerm") String searchTerm, Pageable pageable);
    
    // Statistical queries
    @Query("SELECT COUNT(u) FROM User u WHERE u.status = :status")
    long countByStatus(@Param("status") UserStatus status);
    
    @Query("SELECT COUNT(u) FROM User u WHERE u.createdAt >= :fromDate")
    long countNewUsersFromDate(@Param("fromDate") LocalDateTime fromDate);
    
    @Query("SELECT COUNT(u) FROM User u WHERE u.lastLogin >= :fromDate")
    long countActiveUsersFromDate(@Param("fromDate") LocalDateTime fromDate);
}
