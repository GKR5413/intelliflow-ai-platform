import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useSnackbar } from 'notistack';
import { apiService } from '../services/apiService.ts';

export interface User {
  id: number;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  roles: string[];
  emailVerified: boolean;
  createdAt: string;
}

export interface LoginCredentials {
  usernameOrEmail: string;
  password: string;
}

export interface RegisterData {
  username: string;
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  phoneNumber?: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => void;
  loading: boolean;
  updateProfile: (data: Partial<User>) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const { enqueueSnackbar } = useSnackbar();

  useEffect(() => {
    // Check for stored token on app load
    const storedToken = localStorage.getItem('intelliflow_token');
    const storedUser = localStorage.getItem('intelliflow_user');
    
    if (storedToken && storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        setToken(storedToken);
        setUser(parsedUser);
        apiService.setAuthToken(storedToken);
      } catch (error) {
        console.error('Error parsing stored user data:', error);
        localStorage.removeItem('intelliflow_token');
        localStorage.removeItem('intelliflow_user');
      }
    }
    
    setLoading(false);
  }, []);

  const login = async (credentials: LoginCredentials): Promise<void> => {
    try {
      setLoading(true);
      console.log('🚀 Attempting login with credentials:', credentials);
      console.log('🔗 API endpoint:', '/auth/login');
      
      const response = await apiService.post('/auth/login', credentials);
      console.log('✅ Login response:', response);
      
      // Handle different response formats
      let accessToken: string;
      let userData: User;
      
      if (response.data.accessToken) {
        // New format
        accessToken = response.data.accessToken;
        userData = response.data.user;
      } else if (response.data.token) {
        // Current backend format
        accessToken = response.data.token;
        // For now, create a basic user object - we'll get profile after login
        userData = {
          id: 0, // Will be updated after getting profile
          username: credentials.usernameOrEmail,
          email: credentials.usernameOrEmail,
          firstName: '',
          lastName: '',
          roles: ['USER'],
          emailVerified: true,
          createdAt: new Date().toISOString()
        };
      } else {
        throw new Error('Invalid response format');
      }
      
      setToken(accessToken);
      setUser(userData);
      
      // Store in localStorage
      localStorage.setItem('intelliflow_token', accessToken);
      localStorage.setItem('intelliflow_user', JSON.stringify(userData));
      
      // Set token in API service
      apiService.setAuthToken(accessToken);
      
      // Try to get user profile if we have a token
      if (userData.id === 0) {
        try {
          const profileResponse = await apiService.get('/users/profile');
          const fullUserData = {
            ...userData,
            ...profileResponse.data,
            roles: userData.roles
          };
          setUser(fullUserData);
          localStorage.setItem('intelliflow_user', JSON.stringify(fullUserData));
        } catch (profileError) {
          console.warn('Could not fetch user profile:', profileError);
        }
      }
      
      enqueueSnackbar(`Welcome back, ${userData.firstName}!`, { 
        variant: 'success',
        autoHideDuration: 3000,
      });
    } catch (error: any) {
      console.error('❌ Login error:', error);
      console.error('❌ Error response:', error.response);
      const message = error.response?.data?.message || 'Login failed';
      enqueueSnackbar(message, { variant: 'error' });
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const register = async (data: RegisterData): Promise<void> => {
    try {
      setLoading(true);
      console.log('🚀 Attempting registration with data:', data);
      console.log('🔗 API endpoint:', '/auth/register');
      
      const response = await apiService.post('/auth/register', data);
      console.log('✅ Registration response:', response);
      
      enqueueSnackbar('Registration successful! Please log in.', { 
        variant: 'success',
        autoHideDuration: 5000,
      });
    } catch (error: any) {
      console.error('❌ Registration error:', error);
      console.error('❌ Error response:', error.response);
      const message = error.response?.data?.message || 'Registration failed';
      enqueueSnackbar(message, { variant: 'error' });
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const logout = (): void => {
    setUser(null);
    setToken(null);
    
    // Clear localStorage
    localStorage.removeItem('intelliflow_token');
    localStorage.removeItem('intelliflow_user');
    
    // Clear token from API service
    apiService.setAuthToken(null);
    
    enqueueSnackbar('Logged out successfully', { 
      variant: 'info',
      autoHideDuration: 2000,
    });
  };

  const updateProfile = async (data: Partial<User>): Promise<void> => {
    try {
      const response = await apiService.put('/users/profile', data);
      const updatedUser = response.data;
      
      setUser(updatedUser);
      localStorage.setItem('intelliflow_user', JSON.stringify(updatedUser));
      
      enqueueSnackbar('Profile updated successfully', { 
        variant: 'success',
        autoHideDuration: 3000,
      });
    } catch (error: any) {
      const message = error.response?.data?.message || 'Profile update failed';
      enqueueSnackbar(message, { variant: 'error' });
      throw error;
    }
  };

  const value: AuthContextType = {
    user,
    token,
    login,
    register,
    logout,
    loading,
    updateProfile,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
