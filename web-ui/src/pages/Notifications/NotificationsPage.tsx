import React from 'react';
import { Box, Typography, Card, CardContent } from '@mui/material';
import { Notifications } from '@mui/icons-material';

const NotificationsPage: React.FC = () => {
  return (
    <Box>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Notifications
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Manage alerts and communication preferences
        </Typography>
      </Box>

      <Card>
        <CardContent sx={{ textAlign: 'center', py: 8 }}>
          <Notifications sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Notification Center
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Configure notification settings, view message history, and manage communication preferences.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default NotificationsPage;
