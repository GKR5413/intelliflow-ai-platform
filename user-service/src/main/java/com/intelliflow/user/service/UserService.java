package com.intelliflow.user.service;

import com.intelliflow.user.dto.UserRegistrationDto;
import com.intelliflow.user.dto.UserLoginDto;
import com.intelliflow.user.dto.UserProfileDto;
import com.intelliflow.user.dto.PasswordChangeDto;
import com.intelliflow.user.dto.PasswordResetDto;
import com.intelliflow.user.entity.User;
import com.intelliflow.user.entity.UserStatus;
import com.intelliflow.user.entity.ActivityType;
import com.intelliflow.user.repository.UserRepository;
import com.intelliflow.user.security.JwtTokenProvider;
import com.intelliflow.user.exception.UserNotFoundException;
import com.intelliflow.user.exception.UserAlreadyExistsException;
import com.intelliflow.user.exception.InvalidCredentialsException;
import com.intelliflow.user.exception.AccountLockedException;
import com.intelliflow.user.exception.EmailNotVerifiedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Optional;
import java.util.UUID;

@Service
@Transactional
public class UserService {
    
    private static final Logger logger = LoggerFactory.getLogger(UserService.class);
    private static final int MAX_FAILED_ATTEMPTS = 5;
    private static final int PASSWORD_RESET_EXPIRY_HOURS = 24;
    private static final int EMAIL_VERIFICATION_EXPIRY_HOURS = 48;
    
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtTokenProvider jwtTokenProvider;
    private final UserActivityService userActivityService;
    private final EmailService emailService;
    private final RoleService roleService;
    
    @Value("${app.frontend.url:http://localhost:3000}")
    private String frontendUrl;
    
    @Autowired
    public UserService(UserRepository userRepository,
                      PasswordEncoder passwordEncoder,
                      JwtTokenProvider jwtTokenProvider,
                      UserActivityService userActivityService,
                      EmailService emailService,
                      RoleService roleService) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.jwtTokenProvider = jwtTokenProvider;
        this.userActivityService = userActivityService;
        this.emailService = emailService;
        this.roleService = roleService;
    }
    
    /**
     * Register a new user
     */
    public User registerUser(UserRegistrationDto registrationDto) {
        logger.info("Attempting to register user with email: {}", registrationDto.getEmail());
        
        // Check if user already exists
        if (userRepository.existsByEmail(registrationDto.getEmail())) {
            throw new UserAlreadyExistsException("User with email " + registrationDto.getEmail() + " already exists");
        }
        
        if (userRepository.existsByUsername(registrationDto.getUsername())) {
            throw new UserAlreadyExistsException("User with username " + registrationDto.getUsername() + " already exists");
        }
        
        // Create new user
        User user = new User();
        user.setUsername(registrationDto.getUsername());
        user.setEmail(registrationDto.getEmail());
        user.setPassword(passwordEncoder.encode(registrationDto.getPassword()));
        user.setFirstName(registrationDto.getFirstName());
        user.setLastName(registrationDto.getLastName());
        user.setPhoneNumber(registrationDto.getPhoneNumber());
        user.setStatus(UserStatus.PENDING_VERIFICATION);
        
        // Generate email verification token
        generateEmailVerificationToken(user);
        
        // Save user
        user = userRepository.save(user);
        
        // Assign default role
        roleService.assignDefaultRole(user);
        
        // Log activity
        userActivityService.logActivity(user, ActivityType.REGISTRATION, 
            "User registered successfully", registrationDto.getIpAddress(), registrationDto.getUserAgent());
        
        // Send verification email
        emailService.sendEmailVerification(user);
        
        logger.info("User registered successfully with ID: {}", user.getId());
        return user;
    }
    
    /**
     * Authenticate user and generate JWT token
     */
    public String authenticateUser(UserLoginDto loginDto) {
        logger.info("Attempting authentication for user: {}", loginDto.getUsernameOrEmail());
        
        // Find user by username or email
        Optional<User> userOpt = userRepository.findByUsernameOrEmail(
            loginDto.getUsernameOrEmail(), loginDto.getUsernameOrEmail());
        
        if (!userOpt.isPresent()) {
            userActivityService.logSuspiciousActivity(loginDto.getUsernameOrEmail(), 
                "Failed login attempt - user not found", loginDto.getIpAddress(), loginDto.getUserAgent());
            throw new InvalidCredentialsException("Invalid credentials");
        }
        
        User user = userOpt.get();
        
        // Check if account is locked
        if (user.isLocked()) {
            userActivityService.logActivity(user, ActivityType.SUSPICIOUS_ACTIVITY, 
                "Login attempt on locked account", loginDto.getIpAddress(), loginDto.getUserAgent());
            throw new AccountLockedException("Account is locked due to multiple failed login attempts");
        }
        
        // Check if email is verified (unless it's admin)
        if (!user.getEmailVerified() && !hasAdminRole(user)) {
            throw new EmailNotVerifiedException("Email verification required");
        }
        
        // Verify password
        if (!passwordEncoder.matches(loginDto.getPassword(), user.getPassword())) {
            handleFailedLogin(user, loginDto.getIpAddress(), loginDto.getUserAgent());
            throw new InvalidCredentialsException("Invalid credentials");
        }
        
        // Reset failed attempts on successful login
        if (user.getFailedLoginAttempts() > 0) {
            user.resetFailedLoginAttempts();
            userRepository.save(user);
        }
        
        // Update last login
        user.setLastLogin(LocalDateTime.now());
        userRepository.save(user);
        
        // Log successful login
        userActivityService.logActivity(user, ActivityType.LOGIN, 
            "User logged in successfully", loginDto.getIpAddress(), loginDto.getUserAgent());
        
        // Generate JWT token
        String token = jwtTokenProvider.generateToken(user);
        
        logger.info("User authenticated successfully: {}", user.getUsername());
        return token;
    }
    
    /**
     * Get user ID from JWT token
     */
    public Long getUserIdFromToken(String token) {
        return jwtTokenProvider.getUserIdFromToken(token);
    }
    
    /**
     * Get user profile
     */
    @Transactional(readOnly = true)
    public UserProfileDto getUserProfile(Long userId) {
        User user = findUserById(userId);
        return mapToUserProfileDto(user);
    }
    
    /**
     * Update user profile
     */
    public UserProfileDto updateUserProfile(Long userId, UserProfileDto profileDto) {
        User user = findUserById(userId);
        
        // Update fields
        user.setFirstName(profileDto.getFirstName());
        user.setLastName(profileDto.getLastName());
        user.setPhoneNumber(profileDto.getPhoneNumber());
        user.setDateOfBirth(profileDto.getDateOfBirth());
        user.setAddress(profileDto.getAddress());
        
        // Save user
        user = userRepository.save(user);
        
        // Log activity
        userActivityService.logActivity(user, ActivityType.PROFILE_UPDATE, 
            "User profile updated", null, null);
        
        logger.info("User profile updated for user ID: {}", userId);
        return mapToUserProfileDto(user);
    }
    
    /**
     * Change user password
     */
    public void changePassword(Long userId, PasswordChangeDto passwordChangeDto) {
        User user = findUserById(userId);
        
        // Verify current password
        if (!passwordEncoder.matches(passwordChangeDto.getCurrentPassword(), user.getPassword())) {
            throw new InvalidCredentialsException("Current password is incorrect");
        }
        
        // Update password
        user.setPassword(passwordEncoder.encode(passwordChangeDto.getNewPassword()));
        userRepository.save(user);
        
        // Log activity
        userActivityService.logActivity(user, ActivityType.PASSWORD_CHANGE, 
            "Password changed successfully", null, null);
        
        logger.info("Password changed for user ID: {}", userId);
    }
    
    /**
     * Request password reset
     */
    public void requestPasswordReset(String email) {
        Optional<User> userOpt = userRepository.findByEmail(email);
        
        if (!userOpt.isPresent()) {
            // Don't reveal if email exists or not for security
            logger.warn("Password reset requested for non-existent email: {}", email);
            return;
        }
        
        User user = userOpt.get();
        
        // Generate reset token
        String resetToken = UUID.randomUUID().toString();
        user.setPasswordResetToken(resetToken);
        user.setPasswordResetExpires(LocalDateTime.now().plusHours(PASSWORD_RESET_EXPIRY_HOURS));
        
        userRepository.save(user);
        
        // Send reset email
        emailService.sendPasswordResetEmail(user, resetToken);
        
        // Log activity
        userActivityService.logActivity(user, ActivityType.PASSWORD_RESET, 
            "Password reset requested", null, null);
        
        logger.info("Password reset requested for user: {}", email);
    }
    
    /**
     * Reset password with token
     */
    public void resetPassword(PasswordResetDto resetDto) {
        Optional<User> userOpt = userRepository.findByValidPasswordResetToken(
            resetDto.getToken(), LocalDateTime.now());
        
        if (!userOpt.isPresent()) {
            throw new InvalidCredentialsException("Invalid or expired reset token");
        }
        
        User user = userOpt.get();
        
        // Update password
        user.setPassword(passwordEncoder.encode(resetDto.getNewPassword()));
        user.setPasswordResetToken(null);
        user.setPasswordResetExpires(null);
        
        userRepository.save(user);
        
        // Log activity
        userActivityService.logActivity(user, ActivityType.PASSWORD_RESET, 
            "Password reset completed", null, null);
        
        logger.info("Password reset completed for user ID: {}", user.getId());
    }
    
    /**
     * Verify email with token
     */
    public void verifyEmail(String token) {
        Optional<User> userOpt = userRepository.findByValidEmailVerificationToken(
            token, LocalDateTime.now());
        
        if (!userOpt.isPresent()) {
            throw new InvalidCredentialsException("Invalid or expired verification token");
        }
        
        User user = userOpt.get();
        
        // Mark email as verified
        user.setEmailVerified(true);
        user.setEmailVerificationToken(null);
        user.setEmailVerificationExpires(null);
        user.setStatus(UserStatus.ACTIVE);
        
        userRepository.save(user);
        
        // Log activity
        userActivityService.logActivity(user, ActivityType.EMAIL_VERIFICATION, 
            "Email verified successfully", null, null);
        
        logger.info("Email verified for user ID: {}", user.getId());
    }
    
    /**
     * Resend email verification
     */
    public void resendEmailVerification(String email) {
        Optional<User> userOpt = userRepository.findByEmail(email);
        
        if (!userOpt.isPresent()) {
            throw new UserNotFoundException("User not found with email: " + email);
        }
        
        User user = userOpt.get();
        
        if (user.getEmailVerified()) {
            throw new IllegalStateException("Email is already verified");
        }
        
        // Generate new verification token
        generateEmailVerificationToken(user);
        userRepository.save(user);
        
        // Send verification email
        emailService.sendEmailVerification(user);
        
        logger.info("Email verification resent for user: {}", email);
    }
    
    /**
     * Enable two-factor authentication
     */
    public void enableTwoFactorAuth(Long userId) {
        User user = findUserById(userId);
        user.setTwoFactorEnabled(true);
        userRepository.save(user);
        
        userActivityService.logActivity(user, ActivityType.TWO_FACTOR_ENABLE, 
            "Two-factor authentication enabled", null, null);
        
        logger.info("Two-factor authentication enabled for user ID: {}", userId);
    }
    
    /**
     * Disable two-factor authentication
     */
    public void disableTwoFactorAuth(Long userId) {
        User user = findUserById(userId);
        user.setTwoFactorEnabled(false);
        userRepository.save(user);
        
        userActivityService.logActivity(user, ActivityType.TWO_FACTOR_DISABLE, 
            "Two-factor authentication disabled", null, null);
        
        logger.info("Two-factor authentication disabled for user ID: {}", userId);
    }
    
    /**
     * Search users (admin only)
     */
    @Transactional(readOnly = true)
    public Page<UserProfileDto> searchUsers(String searchTerm, Pageable pageable) {
        Page<User> users = userRepository.searchUsers(searchTerm, pageable);
        return users.map(this::mapToUserProfileDto);
    }
    
    /**
     * Get user by ID
     */
    @Transactional(readOnly = true)
    public User findUserById(Long userId) {
        return userRepository.findById(userId)
            .orElseThrow(() -> new UserNotFoundException("User not found with ID: " + userId));
    }
    
    /**
     * Lock user account
     */
    public void lockUserAccount(Long userId, String reason) {
        User user = findUserById(userId);
        user.setStatus(UserStatus.LOCKED);
        userRepository.save(user);
        
        userActivityService.logActivity(user, ActivityType.ACCOUNT_LOCK, 
            "Account locked: " + reason, null, null);
        
        logger.info("User account locked for user ID: {} - Reason: {}", userId, reason);
    }
    
    /**
     * Unlock user account
     */
    public void unlockUserAccount(Long userId) {
        User user = findUserById(userId);
        user.setStatus(UserStatus.ACTIVE);
        user.resetFailedLoginAttempts();
        userRepository.save(user);
        
        userActivityService.logActivity(user, ActivityType.ACCOUNT_UNLOCK, 
            "Account unlocked", null, null);
        
        logger.info("User account unlocked for user ID: {}", userId);
    }
    
    // Private helper methods
    
    private void handleFailedLogin(User user, String ipAddress, String userAgent) {
        user.incrementFailedLoginAttempts();
        
        if (user.getFailedLoginAttempts() >= MAX_FAILED_ATTEMPTS) {
            user.setStatus(UserStatus.LOCKED);
            userActivityService.logActivity(user, ActivityType.ACCOUNT_LOCK, 
                "Account locked due to excessive failed login attempts", ipAddress, userAgent);
        } else {
            userActivityService.logActivity(user, ActivityType.SUSPICIOUS_ACTIVITY, 
                "Failed login attempt", ipAddress, userAgent);
        }
        
        userRepository.save(user);
    }
    
    private void generateEmailVerificationToken(User user) {
        String verificationToken = UUID.randomUUID().toString();
        user.setEmailVerificationToken(verificationToken);
        user.setEmailVerificationExpires(LocalDateTime.now().plusHours(EMAIL_VERIFICATION_EXPIRY_HOURS));
    }
    
    private boolean hasAdminRole(User user) {
        return user.getUserRoles().stream()
            .anyMatch(userRole -> "ADMIN".equals(userRole.getRole().getName()));
    }
    
    private UserProfileDto mapToUserProfileDto(User user) {
        UserProfileDto dto = new UserProfileDto();
        dto.setId(user.getId());
        dto.setUsername(user.getUsername());
        dto.setEmail(user.getEmail());
        dto.setFirstName(user.getFirstName());
        dto.setLastName(user.getLastName());
        dto.setPhoneNumber(user.getPhoneNumber());
        dto.setDateOfBirth(user.getDateOfBirth());
        dto.setAddress(user.getAddress());
        dto.setStatus(user.getStatus());
        dto.setEmailVerified(user.getEmailVerified());
        dto.setPhoneVerified(user.getPhoneVerified());
        dto.setTwoFactorEnabled(user.getTwoFactorEnabled());
        dto.setLastLogin(user.getLastLogin());
        dto.setCreatedAt(user.getCreatedAt());
        return dto;
    }
}
