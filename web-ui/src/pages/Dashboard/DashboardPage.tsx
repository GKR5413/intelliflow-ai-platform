import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  LinearProgress,
  Avatar,
  IconButton,
  Divider,
  Alert,
  Skeleton,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Payment,
  Security,
  Analytics,
  Notifications,
  Refresh,
  Add,
  Warning,
  CheckCircle,
  Error,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { useAuth } from '../../contexts/AuthContext.tsx';
import { apiService } from '../../services/apiService.ts';
import LoadingSpinner from '../../components/Common/LoadingSpinner.tsx';

interface DashboardMetrics {
  totalTransactions: number;
  transactionVolume: number;
  fraudDetected: number;
  fraudRate: number;
  activeUsers: number;
  systemHealth: 'healthy' | 'warning' | 'error';
}

interface TransactionData {
  date: string;
  amount: number;
  count: number;
  fraudCount: number;
}

interface ServiceStatus {
  name: string;
  status: 'healthy' | 'warning' | 'error';
  responseTime: number;
  uptime: number;
}

const DashboardPage: React.FC = () => {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [transactionData, setTransactionData] = useState<TransactionData[]>([]);
  const [serviceStatuses, setServiceStatuses] = useState<ServiceStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const { user } = useAuth();
  const navigate = useNavigate();

  // Mock data for demo purposes
  const mockMetrics: DashboardMetrics = {
    totalTransactions: 15847,
    transactionVolume: 2468750.89,
    fraudDetected: 23,
    fraudRate: 0.145,
    activeUsers: 1245,
    systemHealth: 'healthy',
  };

  const mockTransactionData: TransactionData[] = [
    { date: '2024-01-01', amount: 125000, count: 450, fraudCount: 2 },
    { date: '2024-01-02', amount: 142000, count: 523, fraudCount: 3 },
    { date: '2024-01-03', amount: 118000, count: 412, fraudCount: 1 },
    { date: '2024-01-04', amount: 165000, count: 598, fraudCount: 4 },
    { date: '2024-01-05', amount: 138000, count: 487, fraudCount: 2 },
    { date: '2024-01-06', amount: 156000, count: 542, fraudCount: 3 },
    { date: '2024-01-07', amount: 172000, count: 612, fraudCount: 5 },
  ];

  const mockServiceStatuses: ServiceStatus[] = [
    { name: 'User Service', status: 'healthy', responseTime: 45, uptime: 99.9 },
    { name: 'Transaction Service', status: 'healthy', responseTime: 52, uptime: 99.8 },
    { name: 'Fraud Detection', status: 'healthy', responseTime: 123, uptime: 99.7 },
    { name: 'Analytics Service', status: 'warning', responseTime: 234, uptime: 98.5 },
    { name: 'Notification Service', status: 'healthy', responseTime: 67, uptime: 99.6 },
  ];

  const fraudTypeData = [
    { name: 'Card Fraud', value: 45, color: '#ff6b6b' },
    { name: 'Identity Theft', value: 30, color: '#4ecdc4' },
    { name: 'Account Takeover', value: 15, color: '#45b7d1' },
    { name: 'Money Laundering', value: 10, color: '#96ceb4' },
  ];

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      // In real implementation, these would be actual API calls
      // For demo, we're using mock data
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API delay
      
      setMetrics(mockMetrics);
      setTransactionData(mockTransactionData);
      setServiceStatuses(mockServiceStatuses);
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadDashboardData();
    setRefreshing(false);
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'warning':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle color="success" />;
      case 'warning':
        return <Warning color="warning" />;
      case 'error':
        return <Error color="error" />;
      default:
        return <CheckCircle />;
    }
  };

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <Grid container spacing={3}>
          {[...Array(8)].map((_, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card>
                <CardContent>
                  <Skeleton variant="text" width="60%" />
                  <Skeleton variant="text" width="80%" height={40} />
                  <Skeleton variant="text" width="40%" />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Welcome back, {user?.firstName}! 👋
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Here's what's happening with your platform today.
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={handleRefresh}
            disabled={refreshing}
          >
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </Button>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => navigate('/transactions/create')}
          >
            New Transaction
          </Button>
        </Box>
      </Box>

      {/* System Health Alert */}
      {metrics?.systemHealth === 'warning' && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Some services are experiencing issues. Check the service status below.
        </Alert>
      )}

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card className="hover-lift">
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Total Transactions
                  </Typography>
                  <Typography variant="h4" component="div">
                    {formatNumber(metrics?.totalTransactions || 0)}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <TrendingUp sx={{ color: 'success.main', mr: 0.5, fontSize: 16 }} />
                    <Typography variant="caption" color="success.main">
                      +12.5% from last month
                    </Typography>
                  </Box>
                </Box>
                <Avatar sx={{ bgcolor: 'primary.main', width: 56, height: 56 }}>
                  <Payment />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card className="hover-lift">
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Transaction Volume
                  </Typography>
                  <Typography variant="h4" component="div">
                    {formatCurrency(metrics?.transactionVolume || 0)}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <TrendingUp sx={{ color: 'success.main', mr: 0.5, fontSize: 16 }} />
                    <Typography variant="caption" color="success.main">
                      +8.2% from last month
                    </Typography>
                  </Box>
                </Box>
                <Avatar sx={{ bgcolor: 'success.main', width: 56, height: 56 }}>
                  <TrendingUp />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card className="hover-lift">
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Fraud Detected
                  </Typography>
                  <Typography variant="h4" component="div">
                    {formatNumber(metrics?.fraudDetected || 0)}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <TrendingDown sx={{ color: 'success.main', mr: 0.5, fontSize: 16 }} />
                    <Typography variant="caption" color="success.main">
                      -5.1% fraud rate
                    </Typography>
                  </Box>
                </Box>
                <Avatar sx={{ bgcolor: 'error.main', width: 56, height: 56 }}>
                  <Security />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card className="hover-lift">
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="body2">
                    Active Users
                  </Typography>
                  <Typography variant="h4" component="div">
                    {formatNumber(metrics?.activeUsers || 0)}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <TrendingUp sx={{ color: 'success.main', mr: 0.5, fontSize: 16 }} />
                    <Typography variant="caption" color="success.main">
                      +3.7% from last week
                    </Typography>
                  </Box>
                </Box>
                <Avatar sx={{ bgcolor: 'info.main', width: 56, height: 56 }}>
                  <Analytics />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts Section */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {/* Transaction Trends */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Transaction Trends
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Daily transaction volume and count over the last 7 days
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={transactionData}>
                  <defs>
                    <linearGradient id="colorAmount" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#667eea" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#667eea" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip 
                    formatter={(value, name) => [
                      name === 'amount' ? formatCurrency(Number(value)) : formatNumber(Number(value)),
                      name === 'amount' ? 'Volume' : 'Count'
                    ]}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="amount" 
                    stroke="#667eea" 
                    fillOpacity={1} 
                    fill="url(#colorAmount)" 
                  />
                  <Line type="monotone" dataKey="count" stroke="#764ba2" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Fraud Detection Breakdown */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Fraud Types
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Distribution of detected fraud types
              </Typography>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={fraudTypeData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {fraudTypeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Service Status & Recent Activity */}
      <Grid container spacing={3}>
        {/* Service Status */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Service Status
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Real-time status of all platform services
              </Typography>
              {serviceStatuses.map((service, index) => (
                <Box key={service.name}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', py: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {getStatusIcon(service.status)}
                      <Box sx={{ ml: 2 }}>
                        <Typography variant="body2" fontWeight={500}>
                          {service.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Response: {service.responseTime}ms | Uptime: {service.uptime}%
                        </Typography>
                      </Box>
                    </Box>
                    <Chip
                      label={service.status}
                      color={getStatusColor(service.status) as any}
                      size="small"
                    />
                  </Box>
                  {index < serviceStatuses.length - 1 && <Divider />}
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Common tasks and shortcuts
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<Payment />}
                    onClick={() => navigate('/transactions/create')}
                    sx={{ py: 2 }}
                  >
                    New Transaction
                  </Button>
                </Grid>
                <Grid item xs={6}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<Security />}
                    onClick={() => navigate('/fraud-detection')}
                    sx={{ py: 2 }}
                  >
                    Fraud Check
                  </Button>
                </Grid>
                <Grid item xs={6}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<Analytics />}
                    onClick={() => navigate('/analytics')}
                    sx={{ py: 2 }}
                  >
                    View Analytics
                  </Button>
                </Grid>
                <Grid item xs={6}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<Notifications />}
                    onClick={() => navigate('/notifications')}
                    sx={{ py: 2 }}
                  >
                    Notifications
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;
