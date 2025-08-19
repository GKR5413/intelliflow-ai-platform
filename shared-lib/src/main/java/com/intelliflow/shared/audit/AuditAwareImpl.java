package com.intelliflow.shared.audit;

import org.springframework.data.domain.AuditorAware;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;

import java.util.Optional;

/**
 * Implementation of AuditorAware to capture the current user for auditing
 */
@Component
public class AuditAwareImpl implements AuditorAware<String> {
    
    @Override
    public Optional<String> getCurrentAuditor() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        
        if (authentication == null || !authentication.isAuthenticated() || 
            "anonymousUser".equals(authentication.getPrincipal())) {
            return Optional.of("SYSTEM");
        }
        
        // If the principal is a UserPrincipal, get the username
        if (authentication.getPrincipal() instanceof com.intelliflow.shared.security.UserPrincipal) {
            com.intelliflow.shared.security.UserPrincipal userPrincipal = 
                (com.intelliflow.shared.security.UserPrincipal) authentication.getPrincipal();
            return Optional.of(userPrincipal.getUsername());
        }
        
        // Otherwise, use the name from authentication
        return Optional.of(authentication.getName());
    }
}
