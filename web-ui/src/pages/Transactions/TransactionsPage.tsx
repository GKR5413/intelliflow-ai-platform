import React from 'react';
import { Box, Typography, Button, Card, CardContent } from '@mui/material';
import { Add, Payment } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const TransactionsPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Box>
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Transactions
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Manage all your financial transactions
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => navigate('/transactions/create')}
        >
          New Transaction
        </Button>
      </Box>

      <Card>
        <CardContent sx={{ textAlign: 'center', py: 8 }}>
          <Payment sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Transaction Management
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            View, search, and manage all transactions. This page will show transaction history, filters, and detailed views.
          </Typography>
          <Button variant="contained" onClick={() => navigate('/transactions/create')}>
            Create Your First Transaction
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
};

export default TransactionsPage;
