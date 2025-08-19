import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Grid,
  MenuItem,
  InputAdornment,
  Alert,
  Stepper,
  Step,
  StepLabel,
  Paper,
  Divider,
  Chip,
} from '@mui/material';
import {
  Payment,
  Security,
  CheckCircle,
  AttachMoney,
  AccountBalance,
  CreditCard,
  Person,
  Business,
} from '@mui/icons-material';
import { useForm, Controller } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import { useSnackbar } from 'notistack';
import { apiService } from '../../services/apiService.ts';
import { useAuth } from '../../contexts/AuthContext.tsx';
import LoadingSpinner from '../../components/Common/LoadingSpinner.tsx';

const schema = yup.object({
  amount: yup
    .number()
    .required('Amount is required')
    .positive('Amount must be positive')
    .max(1000000, 'Amount cannot exceed $1,000,000'),
  currency: yup
    .string()
    .required('Currency is required'),
  transactionType: yup
    .string()
    .required('Transaction type is required'),
  merchantId: yup
    .string()
    .required('Merchant ID is required'),
  description: yup
    .string()
    .required('Description is required')
    .max(500, 'Description cannot exceed 500 characters'),
  paymentMethod: yup
    .string()
    .required('Payment method is required'),
  recipientName: yup
    .string()
    .when('transactionType', {
      is: 'TRANSFER',
      then: (schema) => schema.required('Recipient name is required for transfers'),
      otherwise: (schema) => schema.optional(),
    }),
  recipientAccount: yup
    .string()
    .when('transactionType', {
      is: 'TRANSFER',
      then: (schema) => schema.required('Recipient account is required for transfers'),
      otherwise: (schema) => schema.optional(),
    }),
});

interface TransactionFormData {
  amount: number;
  currency: string;
  transactionType: string;
  merchantId: string;
  description: string;
  paymentMethod: string;
  recipientName?: string;
  recipientAccount?: string;
}

const steps = ['Transaction Details', 'Review & Confirm', 'Processing'];

const CreateTransactionPage: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [transactionResult, setTransactionResult] = useState<any>(null);
  const [fraudResult, setFraudResult] = useState<any>(null);
  const [processing, setProcessing] = useState(false);
  const { user } = useAuth();
  const { enqueueSnackbar } = useSnackbar();
  const navigate = useNavigate();

  const {
    control,
    handleSubmit,
    watch,
    formState: { errors, isValid },
  } = useForm<TransactionFormData>({
    resolver: yupResolver(schema),
    mode: 'onChange',
    defaultValues: {
      amount: 0,
      currency: 'USD',
      transactionType: 'PAYMENT',
      merchantId: '',
      description: '',
      paymentMethod: 'CREDIT_CARD',
      recipientName: '',
      recipientAccount: '',
    },
  });

  const watchedValues = watch();

  const currencies = [
    { value: 'USD', label: 'US Dollar (USD)' },
    { value: 'EUR', label: 'Euro (EUR)' },
    { value: 'GBP', label: 'British Pound (GBP)' },
    { value: 'JPY', label: 'Japanese Yen (JPY)' },
  ];

  const transactionTypes = [
    { value: 'PAYMENT', label: 'Payment', icon: <Payment /> },
    { value: 'TRANSFER', label: 'Transfer', icon: <AccountBalance /> },
    { value: 'WITHDRAWAL', label: 'Withdrawal', icon: <AttachMoney /> },
    { value: 'DEPOSIT', label: 'Deposit', icon: <AccountBalance /> },
  ];

  const paymentMethods = [
    { value: 'CREDIT_CARD', label: 'Credit Card', icon: <CreditCard /> },
    { value: 'DEBIT_CARD', label: 'Debit Card', icon: <CreditCard /> },
    { value: 'BANK_TRANSFER', label: 'Bank Transfer', icon: <AccountBalance /> },
    { value: 'WIRE_TRANSFER', label: 'Wire Transfer', icon: <AccountBalance /> },
  ];

  const merchantSuggestions = [
    'amazon_store',
    'walmart_online',
    'target_retail',
    'starbucks_coffee',
    'gas_station_shell',
    'restaurant_local',
    'grocery_store',
    'online_marketplace',
  ];

  const handleNext = () => {
    if (activeStep === 0 && isValid) {
      setActiveStep(1);
    } else if (activeStep === 1) {
      handleSubmitTransaction();
    }
  };

  const handleBack = () => {
    setActiveStep(activeStep - 1);
  };

  const handleSubmitTransaction = async () => {
    try {
      setProcessing(true);
      setActiveStep(2);

      // Create transaction
      const transactionData = {
        ...watchedValues,
        userId: user?.id || 1,
        metadata: {
          ipAddress: '192.168.1.100',
          userAgent: navigator.userAgent,
          deviceId: 'web_browser_' + Date.now(),
        },
      };

      const transactionResponse = await apiService.createTransaction(transactionData);
      const transaction = transactionResponse.data;
      setTransactionResult(transaction);

      enqueueSnackbar('Transaction created successfully!', { variant: 'success' });

      // Run fraud detection
      setTimeout(async () => {
        try {
          const fraudData = {
            transactionId: transaction.id,
            userId: user?.id || 1,
            amount: watchedValues.amount,
            currency: watchedValues.currency,
            merchantId: watchedValues.merchantId,
            location: {
              country: 'US',
              city: 'San Francisco',
              ipAddress: '192.168.1.100',
            },
            deviceInfo: {
              deviceId: 'web_browser_' + Date.now(),
              userAgent: navigator.userAgent,
            },
            transactionFeatures: {
              isHighRisk: watchedValues.amount > 5000,
              velocityScore: Math.random() * 0.5,
              geolocationRisk: Math.random() * 0.3,
            },
          };

          const fraudResponse = await apiService.scoreFraud(fraudData);
          setFraudResult(fraudResponse.data);

          if (fraudResponse.data.fraudScore > 0.7) {
            enqueueSnackbar('High fraud risk detected!', { variant: 'warning' });
          } else {
            enqueueSnackbar('Transaction passed fraud checks', { variant: 'success' });
          }
        } catch (error) {
          console.error('Fraud detection failed:', error);
          enqueueSnackbar('Fraud detection unavailable', { variant: 'warning' });
        }
      }, 2000);

    } catch (error: any) {
      console.error('Transaction failed:', error);
      enqueueSnackbar(
        error.response?.data?.message || 'Transaction failed. Please try again.',
        { variant: 'error' }
      );
      setActiveStep(1); // Go back to review step
    } finally {
      setProcessing(false);
    }
  };

  const formatCurrency = (amount: number, currency: string) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency,
    }).format(amount);
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={3}>
            {/* Amount and Currency */}
            <Grid item xs={12} md={6}>
              <Controller
                name="amount"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    label="Amount"
                    type="number"
                    error={!!errors.amount}
                    helperText={errors.amount?.message}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <AttachMoney />
                        </InputAdornment>
                      ),
                    }}
                  />
                )}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Controller
                name="currency"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    select
                    label="Currency"
                    error={!!errors.currency}
                    helperText={errors.currency?.message}
                  >
                    {currencies.map((option) => (
                      <MenuItem key={option.value} value={option.value}>
                        {option.label}
                      </MenuItem>
                    ))}
                  </TextField>
                )}
              />
            </Grid>

            {/* Transaction Type */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Transaction Type
              </Typography>
              <Grid container spacing={2}>
                {transactionTypes.map((type) => (
                  <Grid item xs={6} md={3} key={type.value}>
                    <Controller
                      name="transactionType"
                      control={control}
                      render={({ field }) => (
                        <Card
                          sx={{
                            cursor: 'pointer',
                            border: field.value === type.value ? 2 : 1,
                            borderColor: field.value === type.value ? 'primary.main' : 'divider',
                            '&:hover': { borderColor: 'primary.main' },
                          }}
                          onClick={() => field.onChange(type.value)}
                        >
                          <CardContent sx={{ textAlign: 'center', py: 2 }}>
                            {type.icon}
                            <Typography variant="body2" sx={{ mt: 1 }}>
                              {type.label}
                            </Typography>
                          </CardContent>
                        </Card>
                      )}
                    />
                  </Grid>
                ))}
              </Grid>
            </Grid>

            {/* Payment Method */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Payment Method
              </Typography>
              <Grid container spacing={2}>
                {paymentMethods.map((method) => (
                  <Grid item xs={6} md={3} key={method.value}>
                    <Controller
                      name="paymentMethod"
                      control={control}
                      render={({ field }) => (
                        <Card
                          sx={{
                            cursor: 'pointer',
                            border: field.value === method.value ? 2 : 1,
                            borderColor: field.value === method.value ? 'primary.main' : 'divider',
                            '&:hover': { borderColor: 'primary.main' },
                          }}
                          onClick={() => field.onChange(method.value)}
                        >
                          <CardContent sx={{ textAlign: 'center', py: 2 }}>
                            {method.icon}
                            <Typography variant="body2" sx={{ mt: 1 }}>
                              {method.label}
                            </Typography>
                          </CardContent>
                        </Card>
                      )}
                    />
                  </Grid>
                ))}
              </Grid>
            </Grid>

            {/* Merchant ID */}
            <Grid item xs={12}>
              <Controller
                name="merchantId"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    select
                    label="Merchant"
                    error={!!errors.merchantId}
                    helperText={errors.merchantId?.message}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Business />
                        </InputAdornment>
                      ),
                    }}
                  >
                    {merchantSuggestions.map((merchant) => (
                      <MenuItem key={merchant} value={merchant}>
                        {merchant.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </MenuItem>
                    ))}
                  </TextField>
                )}
              />
            </Grid>

            {/* Description */}
            <Grid item xs={12}>
              <Controller
                name="description"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    multiline
                    rows={3}
                    label="Description"
                    error={!!errors.description}
                    helperText={errors.description?.message}
                    placeholder="Enter transaction description..."
                  />
                )}
              />
            </Grid>

            {/* Transfer-specific fields */}
            {watchedValues.transactionType === 'TRANSFER' && (
              <>
                <Grid item xs={12} md={6}>
                  <Controller
                    name="recipientName"
                    control={control}
                    render={({ field }) => (
                      <TextField
                        {...field}
                        fullWidth
                        label="Recipient Name"
                        error={!!errors.recipientName}
                        helperText={errors.recipientName?.message}
                        InputProps={{
                          startAdornment: (
                            <InputAdornment position="start">
                              <Person />
                            </InputAdornment>
                          ),
                        }}
                      />
                    )}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <Controller
                    name="recipientAccount"
                    control={control}
                    render={({ field }) => (
                      <TextField
                        {...field}
                        fullWidth
                        label="Recipient Account"
                        error={!!errors.recipientAccount}
                        helperText={errors.recipientAccount?.message}
                        InputProps={{
                          startAdornment: (
                            <InputAdornment position="start">
                              <AccountBalance />
                            </InputAdornment>
                          ),
                        }}
                      />
                    )}
                  />
                </Grid>
              </>
            )}
          </Grid>
        );

      case 1:
        return (
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Review Transaction Details
            </Typography>
            <Divider sx={{ mb: 3 }} />
            
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Amount:
                </Typography>
                <Typography variant="h6">
                  {formatCurrency(watchedValues.amount, watchedValues.currency)}
                </Typography>
              </Grid>
              
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Type:
                </Typography>
                <Typography variant="h6">
                  {watchedValues.transactionType}
                </Typography>
              </Grid>
              
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Payment Method:
                </Typography>
                <Typography variant="h6">
                  {watchedValues.paymentMethod?.replace(/_/g, ' ')}
                </Typography>
              </Grid>
              
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Merchant:
                </Typography>
                <Typography variant="h6">
                  {watchedValues.merchantId?.replace(/_/g, ' ')}
                </Typography>
              </Grid>
              
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary">
                  Description:
                </Typography>
                <Typography variant="body1">
                  {watchedValues.description}
                </Typography>
              </Grid>

              {watchedValues.transactionType === 'TRANSFER' && (
                <>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Recipient:
                    </Typography>
                    <Typography variant="h6">
                      {watchedValues.recipientName}
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Account:
                    </Typography>
                    <Typography variant="h6">
                      {watchedValues.recipientAccount}
                    </Typography>
                  </Grid>
                </>
              )}
            </Grid>

            {watchedValues.amount > 5000 && (
              <Alert severity="warning" sx={{ mt: 3 }}>
                This is a high-value transaction and will undergo additional fraud detection screening.
              </Alert>
            )}
          </Paper>
        );

      case 2:
        return (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            {processing ? (
              <>
                <LoadingSpinner size="large" />
                <Typography variant="h6" sx={{ mt: 2 }}>
                  Processing Transaction...
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Running fraud detection and validation
                </Typography>
              </>
            ) : (
              <>
                <CheckCircle sx={{ fontSize: 60, color: 'success.main', mb: 2 }} />
                <Typography variant="h5" gutterBottom>
                  Transaction Completed!
                </Typography>
                
                {transactionResult && (
                  <Box sx={{ mt: 3, textAlign: 'left' }}>
                    <Paper sx={{ p: 3 }}>
                      <Typography variant="h6" gutterBottom>
                        Transaction Details
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Transaction ID:
                          </Typography>
                          <Typography variant="body1" className="monospace">
                            {transactionResult.id}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Status:
                          </Typography>
                          <Chip 
                            label={transactionResult.status || 'PENDING'} 
                            color="primary" 
                            size="small" 
                          />
                        </Grid>
                      </Grid>
                    </Paper>
                  </Box>
                )}

                {fraudResult && (
                  <Box sx={{ mt: 3, textAlign: 'left' }}>
                    <Paper sx={{ p: 3 }}>
                      <Typography variant="h6" gutterBottom>
                        Fraud Detection Results
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Fraud Score:
                          </Typography>
                          <Typography variant="h6" color={fraudResult.fraudScore > 0.5 ? 'error.main' : 'success.main'}>
                            {(fraudResult.fraudScore * 100).toFixed(1)}%
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Risk Level:
                          </Typography>
                          <Chip 
                            label={fraudResult.fraudScore > 0.7 ? 'HIGH' : fraudResult.fraudScore > 0.3 ? 'MEDIUM' : 'LOW'}
                            color={fraudResult.fraudScore > 0.7 ? 'error' : fraudResult.fraudScore > 0.3 ? 'warning' : 'success'}
                            size="small"
                          />
                        </Grid>
                      </Grid>
                    </Paper>
                  </Box>
                )}

                <Box sx={{ mt: 4, display: 'flex', gap: 2, justifyContent: 'center' }}>
                  <Button
                    variant="contained"
                    onClick={() => navigate('/transactions')}
                  >
                    View All Transactions
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={() => window.location.reload()}
                  >
                    Create Another
                  </Button>
                </Box>
              </>
            )}
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Create New Transaction
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Process payments, transfers, and other financial transactions
        </Typography>
      </Box>

      {/* Stepper */}
      <Paper sx={{ p: 3, mb: 4 }}>
        <Stepper activeStep={activeStep} alternativeLabel>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
      </Paper>

      {/* Form Content */}
      <Card>
        <CardContent sx={{ p: 4 }}>
          {renderStepContent(activeStep)}

          {/* Navigation Buttons */}
          {activeStep < 2 && (
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
              <Button
                disabled={activeStep === 0}
                onClick={handleBack}
                size="large"
              >
                Back
              </Button>
              <Button
                variant="contained"
                onClick={handleNext}
                disabled={activeStep === 0 && !isValid}
                size="large"
              >
                {activeStep === 1 ? 'Create Transaction' : 'Next'}
              </Button>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default CreateTransactionPage;
