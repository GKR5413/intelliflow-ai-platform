package com.intelliflow.user.service;

import com.intelliflow.user.entity.ActivityType;
import com.intelliflow.user.entity.User;
import com.intelliflow.user.entity.UserActivity;
import com.intelliflow.user.repository.UserActivityRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;

@Service
@Transactional
public class UserActivityService {
    
    private static final Logger logger = LoggerFactory.getLogger(UserActivityService.class);
    
    private final UserActivityRepository userActivityRepository;
    
    @Autowired
    public UserActivityService(UserActivityRepository userActivityRepository) {
        this.userActivityRepository = userActivityRepository;
    }
    
    /**
     * Log user activity
     */
    public void logActivity(User user, ActivityType activityType, String description, 
                           String ipAddress, String userAgent) {
        try {
            UserActivity activity = new UserActivity();
            activity.setUser(user);
            activity.setActivityType(activityType);
            activity.setDescription(description);
            activity.setIpAddress(ipAddress);
            activity.setUserAgent(userAgent);
            
            userActivityRepository.save(activity);
            logger.debug("Logged activity for user {}: {}", user.getUsername(), activityType);
        } catch (Exception e) {
            logger.error("Failed to log activity for user {}", user.getUsername(), e);
        }
    }
    
    /**
     * Log suspicious activity
     */
    public void logSuspiciousActivity(String usernameOrEmail, String description, 
                                    String ipAddress, String userAgent) {
        try {
            UserActivity activity = new UserActivity();
            activity.setActivityType(ActivityType.SUSPICIOUS_ACTIVITY);
            activity.setDescription(description + " - " + usernameOrEmail);
            activity.setIpAddress(ipAddress);
            activity.setUserAgent(userAgent);
            
            userActivityRepository.save(activity);
            logger.warn("Logged suspicious activity: {}", description);
        } catch (Exception e) {
            logger.error("Failed to log suspicious activity", e);
        }
    }
    
    /**
     * Get user activities
     */
    @Transactional(readOnly = true)
    public Page<UserActivity> getUserActivities(Long userId, Pageable pageable) {
        return userActivityRepository.findByUserIdOrderByTimestampDesc(userId, pageable);
    }
    
    /**
     * Get recent suspicious activities
     */
    @Transactional(readOnly = true)
    public Page<UserActivity> getRecentSuspiciousActivities(LocalDateTime since, Pageable pageable) {
        return userActivityRepository.findByActivityTypeAndTimestampAfterOrderByTimestampDesc(
            ActivityType.SUSPICIOUS_ACTIVITY, since, pageable);
    }
}
