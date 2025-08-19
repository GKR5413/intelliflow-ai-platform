import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Divider,
  Chip,
  alpha,
} from '@mui/material';
import {
  Dashboard,
  Payment,
  Security,
  Analytics,
  Notifications,
  Person,
  Settings,
  Rocket,
  TrendingUp,
  AccountBalance,
  Shield,
  Assessment,
  NotificationsActive,
} from '@mui/icons-material';

interface SidebarProps {
  onClose?: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ onClose }) => {
  const location = useLocation();
  const navigate = useNavigate();

  const menuItems = [
    {
      text: 'Dashboard',
      icon: <Dashboard />,
      path: '/dashboard',
      description: 'Overview & metrics',
    },
    {
      text: 'Transactions',
      icon: <Payment />,
      path: '/transactions',
      description: 'Payment processing',
      badge: 'New',
    },
    {
      text: 'Fraud Detection',
      icon: <Security />,
      path: '/fraud-detection',
      description: 'ML-powered analysis',
      badge: 'AI',
    },
    {
      text: 'Analytics',
      icon: <Analytics />,
      path: '/analytics',
      description: 'Business intelligence',
    },
    {
      text: 'Notifications',
      icon: <Notifications />,
      path: '/notifications',
      description: 'Alerts & messages',
    },
  ];

  const userMenuItems = [
    {
      text: 'Profile',
      icon: <Person />,
      path: '/profile',
      description: 'Account settings',
    },
    {
      text: 'Settings',
      icon: <Settings />,
      path: '/settings',
      description: 'Preferences',
    },
  ];

  const handleNavigation = (path: string) => {
    navigate(path);
    if (onClose) {
      onClose();
    }
  };

  const isSelected = (path: string) => {
    if (path === '/transactions') {
      return location.pathname.startsWith('/transactions');
    }
    return location.pathname === path;
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Logo/Header */}
      <Box
        sx={{
          p: 3,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Rocket sx={{ mr: 1, fontSize: 28 }} />
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            IntelliFlow
          </Typography>
        </Box>
        <Typography variant="caption" sx={{ opacity: 0.9 }}>
          AI-Powered Financial Platform
        </Typography>
      </Box>

      {/* Main Navigation */}
      <Box sx={{ flex: 1, py: 2 }}>
        <List>
          {menuItems.map((item) => (
            <ListItem key={item.path} disablePadding sx={{ px: 2, mb: 0.5 }}>
              <ListItemButton
                selected={isSelected(item.path)}
                onClick={() => handleNavigation(item.path)}
                sx={{
                  borderRadius: 2,
                  '&.Mui-selected': {
                    bgcolor: alpha('#667eea', 0.12),
                    '&:hover': {
                      bgcolor: alpha('#667eea', 0.16),
                    },
                  },
                  '&:hover': {
                    bgcolor: alpha('#667eea', 0.08),
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    color: isSelected(item.path) ? 'primary.main' : 'text.secondary',
                    minWidth: 40,
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography
                        variant="body2"
                        sx={{
                          fontWeight: isSelected(item.path) ? 600 : 500,
                          color: isSelected(item.path) ? 'primary.main' : 'text.primary',
                        }}
                      >
                        {item.text}
                      </Typography>
                      {item.badge && (
                        <Chip
                          label={item.badge}
                          size="small"
                          color={item.badge === 'AI' ? 'secondary' : 'primary'}
                          sx={{
                            height: 18,
                            fontSize: '0.65rem',
                            fontWeight: 600,
                          }}
                        />
                      )}
                    </Box>
                  }
                  secondary={
                    <Typography
                      variant="caption"
                      sx={{
                        color: 'text.secondary',
                        fontSize: '0.7rem',
                      }}
                    >
                      {item.description}
                    </Typography>
                  }
                />
              </ListItemButton>
            </ListItem>
          ))}
        </List>

        <Divider sx={{ mx: 2, my: 2 }} />

        {/* User Menu */}
        <List>
          {userMenuItems.map((item) => (
            <ListItem key={item.path} disablePadding sx={{ px: 2, mb: 0.5 }}>
              <ListItemButton
                selected={isSelected(item.path)}
                onClick={() => handleNavigation(item.path)}
                sx={{
                  borderRadius: 2,
                  '&.Mui-selected': {
                    bgcolor: alpha('#667eea', 0.12),
                    '&:hover': {
                      bgcolor: alpha('#667eea', 0.16),
                    },
                  },
                  '&:hover': {
                    bgcolor: alpha('#667eea', 0.08),
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    color: isSelected(item.path) ? 'primary.main' : 'text.secondary',
                    minWidth: 40,
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Typography
                      variant="body2"
                      sx={{
                        fontWeight: isSelected(item.path) ? 600 : 500,
                        color: isSelected(item.path) ? 'primary.main' : 'text.primary',
                      }}
                    >
                      {item.text}
                    </Typography>
                  }
                  secondary={
                    <Typography
                      variant="caption"
                      sx={{
                        color: 'text.secondary',
                        fontSize: '0.7rem',
                      }}
                    >
                      {item.description}
                    </Typography>
                  }
                />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>

      {/* Footer */}
      <Box sx={{ p: 2, borderTop: '1px solid', borderColor: 'divider' }}>
        <Box
          sx={{
            p: 2,
            borderRadius: 2,
            bgcolor: alpha('#667eea', 0.08),
            textAlign: 'center',
          }}
        >
          <TrendingUp sx={{ color: 'primary.main', mb: 1, fontSize: 24 }} />
          <Typography variant="caption" display="block" sx={{ fontWeight: 600 }}>
            Platform Status
          </Typography>
          <Typography variant="caption" color="success.main" sx={{ fontWeight: 500 }}>
            All systems operational
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default Sidebar;
