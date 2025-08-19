import React from 'react';
import { Box, Typography, Card, CardContent } from '@mui/material';
import { Analytics } from '@mui/icons-material';

const AnalyticsPage: React.FC = () => {
  return (
    <Box>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Analytics
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Business intelligence and reporting
        </Typography>
      </Box>

      <Card>
        <CardContent sx={{ textAlign: 'center', py: 8 }}>
          <Analytics sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Business Analytics
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Comprehensive analytics and reporting for transaction patterns, user behavior, and business insights.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default AnalyticsPage;
