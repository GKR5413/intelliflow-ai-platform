package com.intelliflow.user.service;

import com.intelliflow.user.entity.User;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

@Service
public class EmailService {
    
    private static final Logger logger = LoggerFactory.getLogger(EmailService.class);
    
    private final JavaMailSender mailSender;
    
    @Value("${app.mail.from:noreply@intelliflow.com}")
    private String fromEmail;
    
    @Value("${app.frontend.url:http://localhost:3000}")
    private String frontendUrl;
    
    @Autowired
    public EmailService(JavaMailSender mailSender) {
        this.mailSender = mailSender;
    }
    
    /**
     * Send email verification
     */
    public void sendEmailVerification(User user) {
        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setFrom(fromEmail);
            message.setTo(user.getEmail());
            message.setSubject("Email Verification - IntelliFlow");
            
            String verificationUrl = frontendUrl + "/verify-email?token=" + user.getEmailVerificationToken();
            String emailBody = String.format(
                "Hello %s,\n\n" +
                "Thank you for registering with IntelliFlow. Please click the link below to verify your email address:\n\n" +
                "%s\n\n" +
                "This link will expire in 48 hours.\n\n" +
                "If you did not create this account, please ignore this email.\n\n" +
                "Best regards,\n" +
                "IntelliFlow Team",
                user.getFirstName() != null ? user.getFirstName() : user.getUsername(),
                verificationUrl
            );
            
            message.setText(emailBody);
            mailSender.send(message);
            
            logger.info("Email verification sent to: {}", user.getEmail());
        } catch (Exception e) {
            logger.error("Failed to send email verification to: {}", user.getEmail(), e);
        }
    }
    
    /**
     * Send password reset email
     */
    public void sendPasswordResetEmail(User user, String resetToken) {
        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setFrom(fromEmail);
            message.setTo(user.getEmail());
            message.setSubject("Password Reset - IntelliFlow");
            
            String resetUrl = frontendUrl + "/reset-password?token=" + resetToken;
            String emailBody = String.format(
                "Hello %s,\n\n" +
                "You have requested to reset your password. Please click the link below to reset your password:\n\n" +
                "%s\n\n" +
                "This link will expire in 24 hours.\n\n" +
                "If you did not request this password reset, please ignore this email and your password will remain unchanged.\n\n" +
                "Best regards,\n" +
                "IntelliFlow Team",
                user.getFirstName() != null ? user.getFirstName() : user.getUsername(),
                resetUrl
            );
            
            message.setText(emailBody);
            mailSender.send(message);
            
            logger.info("Password reset email sent to: {}", user.getEmail());
        } catch (Exception e) {
            logger.error("Failed to send password reset email to: {}", user.getEmail(), e);
        }
    }
}
