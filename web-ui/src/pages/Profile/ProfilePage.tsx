import React from 'react';
import { Box, Typography, Card, CardContent } from '@mui/material';
import { Person } from '@mui/icons-material';

const ProfilePage: React.FC = () => {
  return (
    <Box>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Profile
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Manage your account information and preferences
        </Typography>
      </Box>

      <Card>
        <CardContent sx={{ textAlign: 'center', py: 8 }}>
          <Person sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            User Profile
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Update personal information, security settings, and account preferences.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ProfilePage;
