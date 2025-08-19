package com.intelliflow.user.repository;

import com.intelliflow.user.entity.ActivityType;
import com.intelliflow.user.entity.UserActivity;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface UserActivityRepository extends JpaRepository<UserActivity, Long> {
    
    // Find activities by user
    Page<UserActivity> findByUserIdOrderByTimestampDesc(Long userId, Pageable pageable);
    
    // Find activities by type
    Page<UserActivity> findByActivityTypeOrderByTimestampDesc(ActivityType activityType, Pageable pageable);
    
    // Find activities by type and time range
    Page<UserActivity> findByActivityTypeAndTimestampAfterOrderByTimestampDesc(
        ActivityType activityType, LocalDateTime since, Pageable pageable);
    
    // Find recent activities for user
    @Query("SELECT a FROM UserActivity a WHERE a.user.id = :userId AND a.timestamp >= :since ORDER BY a.timestamp DESC")
    List<UserActivity> findRecentActivitiesByUser(@Param("userId") Long userId, @Param("since") LocalDateTime since);
    
    // Count activities by type for user in time range
    @Query("SELECT COUNT(a) FROM UserActivity a WHERE a.user.id = :userId AND a.activityType = :type AND a.timestamp >= :since")
    long countByUserAndTypeAndTimestampAfter(@Param("userId") Long userId, 
                                           @Param("type") ActivityType type, 
                                           @Param("since") LocalDateTime since);
    
    // Find activities by IP address
    List<UserActivity> findByIpAddressOrderByTimestampDesc(String ipAddress);
    
    // Find suspicious activities from IP
    @Query("SELECT a FROM UserActivity a WHERE a.ipAddress = :ipAddress AND a.activityType = 'SUSPICIOUS_ACTIVITY' AND a.timestamp >= :since")
    List<UserActivity> findSuspiciousActivitiesByIp(@Param("ipAddress") String ipAddress, @Param("since") LocalDateTime since);
}
