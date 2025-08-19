package com.intelliflow.user.controller;

import com.intelliflow.user.dto.*;
import com.intelliflow.user.entity.User;
import com.intelliflow.user.service.UserService;
import com.intelliflow.user.dto.ApiResponse;
import com.intelliflow.user.dto.JwtResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
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
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;

import java.security.Principal;
import java.util.Map;
import java.util.HashMap;

@RestController
@RequestMapping("/api/v1")
@CrossOrigin(origins = "*", maxAge = 3600)
public class UserController {
    
    private static final Logger logger = LoggerFactory.getLogger(UserController.class);
    
    private final UserService userService;
    
    @Autowired
    public UserController(UserService userService) {
        this.userService = userService;
    }
    
    /**
     * User Registration
     */
    @PostMapping("/auth/register")
    public ResponseEntity<ApiResponse> registerUser(@Valid @RequestBody UserRegistrationDto registrationDto,
                                                   BindingResult bindingResult,
                                                   HttpServletRequest request) {
        logger.info("Registration attempt for email: {}", registrationDto.getEmail());
        
        // Check for validation errors
        if (bindingResult.hasErrors()) {
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, "Validation failed", bindingResult.getAllErrors()));
        }
        
        // Check if passwords match
        if (!registrationDto.isPasswordMatching()) {
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, "Passwords do not match"));
        }
        
        // Set request metadata
        registrationDto.setIpAddress(getClientIpAddress(request));
        registrationDto.setUserAgent(request.getHeader("User-Agent"));
        
        try {
            User user = userService.registerUser(registrationDto);
            return ResponseEntity.status(HttpStatus.CREATED)
                .body(new ApiResponse(true, "User registered successfully. Please check your email for verification."));
        } catch (Exception e) {
            logger.error("Registration failed for email: {}", registrationDto.getEmail(), e);
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, e.getMessage()));
        }
    }
    
    /**
     * User Login
     */
    @PostMapping("/auth/login")
    public ResponseEntity<?> authenticateUser(@Valid @RequestBody UserLoginDto loginDto,
                                            BindingResult bindingResult,
                                            HttpServletRequest request) {
        logger.info("Login attempt for: {}", loginDto.getUsernameOrEmail());
        
        // Check for validation errors
        if (bindingResult.hasErrors()) {
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, "Validation failed", bindingResult.getAllErrors()));
        }
        
        // Set request metadata
        loginDto.setIpAddress(getClientIpAddress(request));
        loginDto.setUserAgent(request.getHeader("User-Agent"));
        loginDto.setDeviceFingerprint(request.getHeader("X-Device-Fingerprint"));
        
        try {
            String token = userService.authenticateUser(loginDto);
            
            // Get user details for response
            Long userId = userService.getUserIdFromToken(token);
            UserProfileDto userProfile = userService.getUserProfile(userId);
            
            // Create response with token and user data
            Map<String, Object> response = new HashMap<>();
            response.put("accessToken", token);
            response.put("tokenType", "Bearer");
            response.put("user", userProfile);
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            logger.error("Login failed for: {}", loginDto.getUsernameOrEmail(), e);
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                .body(new ApiResponse(false, e.getMessage()));
        }
    }
    
    /**
     * User Logout
     */
    @PostMapping("/auth/logout")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<ApiResponse> logout(@AuthenticationPrincipal Principal principal) {
        // Note: JWT tokens are stateless, so logout is handled client-side
        // In a production system, you might want to maintain a blacklist of tokens
        logger.info("User logged out: {}", principal.getName());
        return ResponseEntity.ok(new ApiResponse(true, "User logged out successfully"));
    }
    
    /**
     * Get Current User Profile
     */
    @GetMapping("/users/profile")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<?> getCurrentUserProfile(@AuthenticationPrincipal Principal principal) {
        try {
            Long userId = Long.parseLong(principal.getName());
            UserProfileDto profile = userService.getUserProfile(userId);
            return ResponseEntity.ok(profile);
        } catch (Exception e) {
            logger.error("Failed to get user profile", e);
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, e.getMessage()));
        }
    }
    
    /**
     * Update User Profile
     */
    @PutMapping("/users/profile")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<?> updateUserProfile(@Valid @RequestBody UserProfileDto profileDto,
                                             BindingResult bindingResult,
                                             @AuthenticationPrincipal Principal principal) {
        // Check for validation errors
        if (bindingResult.hasErrors()) {
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, "Validation failed", bindingResult.getAllErrors()));
        }
        
        try {
            Long userId = Long.parseLong(principal.getName());
            UserProfileDto updatedProfile = userService.updateUserProfile(userId, profileDto);
            return ResponseEntity.ok(updatedProfile);
        } catch (Exception e) {
            logger.error("Failed to update user profile", e);
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, e.getMessage()));
        }
    }
    
    /**
     * Change Password
     */
    @PostMapping("/users/change-password")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<ApiResponse> changePassword(@Valid @RequestBody PasswordChangeDto passwordChangeDto,
                                                     BindingResult bindingResult,
                                                     @AuthenticationPrincipal Principal principal) {
        // Check for validation errors
        if (bindingResult.hasErrors()) {
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, "Validation failed", bindingResult.getAllErrors()));
        }
        
        // Check if new passwords match
        if (!passwordChangeDto.isNewPasswordMatching()) {
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, "New passwords do not match"));
        }
        
        try {
            Long userId = Long.parseLong(principal.getName());
            userService.changePassword(userId, passwordChangeDto);
            return ResponseEntity.ok(new ApiResponse(true, "Password changed successfully"));
        } catch (Exception e) {
            logger.error("Failed to change password", e);
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, e.getMessage()));
        }
    }
    
    /**
     * Request Password Reset
     */
    @PostMapping("/auth/forgot-password")
    public ResponseEntity<ApiResponse> requestPasswordReset(@RequestParam String email) {
        try {
            userService.requestPasswordReset(email);
            return ResponseEntity.ok(new ApiResponse(true, 
                "If an account with that email exists, a password reset link has been sent."));
        } catch (Exception e) {
            logger.error("Password reset request failed for email: {}", email, e);
            // Don't reveal if email exists or not for security
            return ResponseEntity.ok(new ApiResponse(true, 
                "If an account with that email exists, a password reset link has been sent."));
        }
    }
    
    /**
     * Reset Password with Token
     */
    @PostMapping("/auth/reset-password")
    public ResponseEntity<ApiResponse> resetPassword(@Valid @RequestBody PasswordResetDto resetDto,
                                                    BindingResult bindingResult) {
        // Check for validation errors
        if (bindingResult.hasErrors()) {
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, "Validation failed", bindingResult.getAllErrors()));
        }
        
        // Check if passwords match
        if (!resetDto.isPasswordMatching()) {
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, "Passwords do not match"));
        }
        
        try {
            userService.resetPassword(resetDto);
            return ResponseEntity.ok(new ApiResponse(true, "Password reset successfully"));
        } catch (Exception e) {
            logger.error("Password reset failed", e);
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, e.getMessage()));
        }
    }
    
    /**
     * Verify Email
     */
    @GetMapping("/auth/verify-email")
    public ResponseEntity<ApiResponse> verifyEmail(@RequestParam String token) {
        try {
            userService.verifyEmail(token);
            return ResponseEntity.ok(new ApiResponse(true, "Email verified successfully"));
        } catch (Exception e) {
            logger.error("Email verification failed", e);
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, e.getMessage()));
        }
    }
    
    /**
     * Resend Email Verification
     */
    @PostMapping("/auth/resend-verification")
    public ResponseEntity<ApiResponse> resendEmailVerification(@RequestParam String email) {
        try {
            userService.resendEmailVerification(email);
            return ResponseEntity.ok(new ApiResponse(true, "Verification email sent"));
        } catch (Exception e) {
            logger.error("Failed to resend verification email", e);
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, e.getMessage()));
        }
    }
    
    /**
     * Enable Two-Factor Authentication
     */
    @PostMapping("/users/enable-2fa")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<ApiResponse> enableTwoFactorAuth(@AuthenticationPrincipal Principal principal) {
        try {
            Long userId = Long.parseLong(principal.getName());
            userService.enableTwoFactorAuth(userId);
            return ResponseEntity.ok(new ApiResponse(true, "Two-factor authentication enabled"));
        } catch (Exception e) {
            logger.error("Failed to enable 2FA", e);
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, e.getMessage()));
        }
    }
    
    /**
     * Disable Two-Factor Authentication
     */
    @PostMapping("/users/disable-2fa")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<ApiResponse> disableTwoFactorAuth(@AuthenticationPrincipal Principal principal) {
        try {
            Long userId = Long.parseLong(principal.getName());
            userService.disableTwoFactorAuth(userId);
            return ResponseEntity.ok(new ApiResponse(true, "Two-factor authentication disabled"));
        } catch (Exception e) {
            logger.error("Failed to disable 2FA", e);
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, e.getMessage()));
        }
    }
    
    /**
     * Get User by ID (Admin only)
     */
    @GetMapping("/users/{userId}")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<?> getUserById(@PathVariable Long userId) {
        try {
            UserProfileDto profile = userService.getUserProfile(userId);
            return ResponseEntity.ok(profile);
        } catch (Exception e) {
            logger.error("Failed to get user by ID: {}", userId, e);
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, e.getMessage()));
        }
    }
    
    /**
     * Search Users (Admin only)
     */
    @GetMapping("/users/search")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<?> searchUsers(
            @RequestParam(required = false) String searchTerm,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(defaultValue = "createdAt") String sortBy,
            @RequestParam(defaultValue = "desc") String sortDir) {
        
        try {
            Sort sort = Sort.by(Sort.Direction.fromString(sortDir), sortBy);
            Pageable pageable = PageRequest.of(page, size, sort);
            
            Page<UserProfileDto> users = userService.searchUsers(searchTerm != null ? searchTerm : "", pageable);
            return ResponseEntity.ok(users);
        } catch (Exception e) {
            logger.error("Failed to search users", e);
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, e.getMessage()));
        }
    }
    
    /**
     * Lock User Account (Admin only)
     */
    @PostMapping("/users/{userId}/lock")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<ApiResponse> lockUser(@PathVariable Long userId,
                                               @RequestParam(required = false) String reason) {
        try {
            userService.lockUserAccount(userId, reason != null ? reason : "Administrative action");
            return ResponseEntity.ok(new ApiResponse(true, "User account locked successfully"));
        } catch (Exception e) {
            logger.error("Failed to lock user: {}", userId, e);
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, e.getMessage()));
        }
    }
    
    /**
     * Unlock User Account (Admin only)
     */
    @PostMapping("/users/{userId}/unlock")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<ApiResponse> unlockUser(@PathVariable Long userId) {
        try {
            userService.unlockUserAccount(userId);
            return ResponseEntity.ok(new ApiResponse(true, "User account unlocked successfully"));
        } catch (Exception e) {
            logger.error("Failed to unlock user: {}", userId, e);
            return ResponseEntity.badRequest()
                .body(new ApiResponse(false, e.getMessage()));
        }
    }
    
    /**
     * Get Client IP Address
     */
    private String getClientIpAddress(HttpServletRequest request) {
        String[] headers = {
            "X-Forwarded-For",
            "X-Real-IP",
            "Proxy-Client-IP",
            "WL-Proxy-Client-IP",
            "HTTP_X_FORWARDED_FOR",
            "HTTP_X_FORWARDED",
            "HTTP_X_CLUSTER_CLIENT_IP",
            "HTTP_CLIENT_IP",
            "HTTP_FORWARDED_FOR",
            "HTTP_FORWARDED",
            "HTTP_VIA",
            "REMOTE_ADDR"
        };
        
        for (String header : headers) {
            String value = request.getHeader(header);
            if (value != null && !value.isEmpty() && !"unknown".equalsIgnoreCase(value)) {
                return value.split(",")[0].trim();
            }
        }
        
        return request.getRemoteAddr();
    }
}
