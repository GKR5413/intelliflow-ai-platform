import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';

class ApiService {
  private api: AxiosInstance;

  constructor() {
    this.api = axios.create({
      baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8081/api/v1',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    // Log the API configuration for debugging
    console.log('🚀 API Service initialized with baseURL:', this.api.defaults.baseURL);

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        // Add timestamp to prevent caching
        config.params = {
          ...config.params,
          _t: Date.now(),
        };

        console.log(`🚀 ${config.method?.toUpperCase()} ${config.url}`, {
          data: config.data,
          params: config.params,
        });

        return config;
      },
      (error) => {
        console.error('❌ Request error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.api.interceptors.response.use(
      (response: AxiosResponse) => {
        console.log(`✅ ${response.config.method?.toUpperCase()} ${response.config.url}`, {
          status: response.status,
          data: response.data,
        });
        return response;
      },
      (error: AxiosError) => {
        console.error(`❌ ${error.config?.method?.toUpperCase()} ${error.config?.url}`, {
          status: error.response?.status,
          data: error.response?.data,
          message: error.message,
        });

        // Handle common error scenarios
        if (error.response?.status === 401) {
          // Token expired or unauthorized
          this.clearAuthToken();
          window.location.href = '/login';
        }

        return Promise.reject(error);
      }
    );
  }

  setAuthToken(token: string | null): void {
    if (token) {
      this.api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
      delete this.api.defaults.headers.common['Authorization'];
    }
  }

  clearAuthToken(): void {
    delete this.api.defaults.headers.common['Authorization'];
    localStorage.removeItem('intelliflow_token');
    localStorage.removeItem('intelliflow_user');
  }

  // Generic HTTP methods
  async get<T = any>(url: string, params?: any): Promise<AxiosResponse<T>> {
    return this.api.get(url, { params });
  }

  async post<T = any>(url: string, data?: any): Promise<AxiosResponse<T>> {
    return this.api.post(url, data);
  }

  async put<T = any>(url: string, data?: any): Promise<AxiosResponse<T>> {
    return this.api.put(url, data);
  }

  async patch<T = any>(url: string, data?: any): Promise<AxiosResponse<T>> {
    return this.api.patch(url, data);
  }

  async delete<T = any>(url: string): Promise<AxiosResponse<T>> {
    return this.api.delete(url);
  }

  // Service-specific methods
  async getServiceHealth(service: string): Promise<AxiosResponse> {
    const serviceUrls: Record<string, string> = {
      'user': 'http://localhost:8081/actuator/health',
      'transaction': 'http://localhost:8082/actuator/health',
      'fraud': 'http://localhost:8083/health',
      'analytics': 'http://localhost:8084/actuator/health',
      'notification': 'http://localhost:8085/actuator/health',
    };

    const url = serviceUrls[service];
    if (!url) {
      throw new Error(`Unknown service: ${service}`);
    }

    return axios.get(url, { timeout: 5000 });
  }

  // Transaction Service API
  async createTransaction(data: any): Promise<AxiosResponse> {
    return axios.post('http://localhost:8082/api/v1/transactions', data, {
      headers: this.api.defaults.headers.common,
    });
  }

  async getTransactions(params?: any): Promise<AxiosResponse> {
    return axios.get('http://localhost:8082/api/v1/transactions', {
      params,
      headers: this.api.defaults.headers.common,
    });
  }

  async getTransaction(id: number): Promise<AxiosResponse> {
    return axios.get(`http://localhost:8082/api/v1/transactions/${id}`, {
      headers: this.api.defaults.headers.common,
    });
  }

  async getUserTransactions(userId: number, params?: any): Promise<AxiosResponse> {
    return axios.get(`http://localhost:8082/api/v1/transactions/user/${userId}`, {
      params,
      headers: this.api.defaults.headers.common,
    });
  }

  // Fraud Detection Service API
  async scoreFraud(data: any): Promise<AxiosResponse> {
    return axios.post('http://localhost:8083/api/v1/fraud/score', data, {
      headers: this.api.defaults.headers.common,
    });
  }

  async getFraudResult(transactionId: number): Promise<AxiosResponse> {
    return axios.get(`http://localhost:8083/api/v1/fraud/transaction/${transactionId}/result`, {
      headers: this.api.defaults.headers.common,
    });
  }

  async getFraudStats(): Promise<AxiosResponse> {
    return axios.get('http://localhost:8083/api/v1/fraud/stats', {
      headers: this.api.defaults.headers.common,
    });
  }

  // Analytics Service API
  async getUserAnalytics(userId: number, params?: any): Promise<AxiosResponse> {
    return axios.get(`http://localhost:8084/api/v1/analytics/user/${userId}`, {
      params,
      headers: this.api.defaults.headers.common,
    });
  }

  async getTransactionAnalytics(params?: any): Promise<AxiosResponse> {
    return axios.get('http://localhost:8084/api/v1/analytics/transactions/summary', {
      params,
      headers: this.api.defaults.headers.common,
    });
  }

  async getDashboardMetrics(): Promise<AxiosResponse> {
    return axios.get('http://localhost:8084/api/v1/analytics/dashboard', {
      headers: this.api.defaults.headers.common,
    });
  }

  // Notification Service API
  async sendNotification(data: any): Promise<AxiosResponse> {
    return axios.post('http://localhost:8085/api/v1/notifications/send', data, {
      headers: this.api.defaults.headers.common,
    });
  }

  async getNotifications(params?: any): Promise<AxiosResponse> {
    return axios.get('http://localhost:8085/api/v1/notifications', {
      params,
      headers: this.api.defaults.headers.common,
    });
  }

  async markNotificationAsRead(id: number): Promise<AxiosResponse> {
    return axios.patch(`http://localhost:8085/api/v1/notifications/${id}/read`, {}, {
      headers: this.api.defaults.headers.common,
    });
  }

  // Utility methods
  async uploadFile(file: File, endpoint: string): Promise<AxiosResponse> {
    const formData = new FormData();
    formData.append('file', file);

    return this.api.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  }

  async downloadFile(url: string, filename: string): Promise<void> {
    try {
      const response = await this.api.get(url, {
        responseType: 'blob',
      });

      const blob = new Blob([response.data]);
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      console.error('Download failed:', error);
      throw error;
    }
  }
}

export const apiService = new ApiService();
