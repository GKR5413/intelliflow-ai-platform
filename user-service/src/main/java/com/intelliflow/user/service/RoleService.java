package com.intelliflow.user.service;

import com.intelliflow.user.entity.Role;
import com.intelliflow.user.entity.User;
import com.intelliflow.user.entity.UserRole;
import com.intelliflow.user.repository.RoleRepository;
import com.intelliflow.user.repository.UserRoleRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Optional;

@Service
@Transactional
public class RoleService {
    
    private static final Logger logger = LoggerFactory.getLogger(RoleService.class);
    
    private final RoleRepository roleRepository;
    private final UserRoleRepository userRoleRepository;
    
    @Autowired
    public RoleService(RoleRepository roleRepository, UserRoleRepository userRoleRepository) {
        this.roleRepository = roleRepository;
        this.userRoleRepository = userRoleRepository;
    }
    
    /**
     * Assign default role to user
     */
    public void assignDefaultRole(User user) {
        Optional<Role> defaultRole = roleRepository.findByName("USER");
        
        if (defaultRole.isPresent()) {
            UserRole userRole = new UserRole(user, defaultRole.get());
            userRoleRepository.save(userRole);
            logger.info("Assigned default role USER to user: {}", user.getUsername());
        } else {
            logger.error("Default role USER not found in database");
        }
    }
    
    /**
     * Assign role to user
     */
    public void assignRole(User user, String roleName) {
        Optional<Role> role = roleRepository.findByName(roleName);
        
        if (role.isPresent()) {
            // Check if user already has this role
            boolean hasRole = userRoleRepository.existsByUserAndRole(user, role.get());
            
            if (!hasRole) {
                UserRole userRole = new UserRole(user, role.get());
                userRoleRepository.save(userRole);
                logger.info("Assigned role {} to user: {}", roleName, user.getUsername());
            } else {
                logger.info("User {} already has role {}", user.getUsername(), roleName);
            }
        } else {
            logger.error("Role {} not found in database", roleName);
        }
    }
    
    /**
     * Remove role from user
     */
    public void removeRole(User user, String roleName) {
        Optional<Role> role = roleRepository.findByName(roleName);
        
        if (role.isPresent()) {
            Optional<UserRole> userRole = userRoleRepository.findByUserAndRole(user, role.get());
            
            if (userRole.isPresent()) {
                userRoleRepository.delete(userRole.get());
                logger.info("Removed role {} from user: {}", roleName, user.getUsername());
            } else {
                logger.info("User {} does not have role {}", user.getUsername(), roleName);
            }
        } else {
            logger.error("Role {} not found in database", roleName);
        }
    }
}
