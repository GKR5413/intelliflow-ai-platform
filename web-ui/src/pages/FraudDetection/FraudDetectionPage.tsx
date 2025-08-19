import React from 'react';
import { Box, Typography, Card, CardContent } from '@mui/material';
import { Security } from '@mui/icons-material';

const FraudDetectionPage: React.FC = () => {
  return (
    <Box>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Fraud Detection
        </Typography>
        <Typography variant="body1" color="text.secondary">
          AI-powered fraud detection and risk analysis
        </Typography>
      </Box>

      <Card>
        <CardContent sx={{ textAlign: 'center', py: 8 }}>
          <Security sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            ML-Powered Fraud Detection
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Advanced machine learning algorithms analyze transactions in real-time to detect fraudulent activity.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default FraudDetectionPage;
