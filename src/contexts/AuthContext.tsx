import { createContext, useContext, useState, useEffect, ReactNode, useMemo } from 'react';
import { User } from '../types';
import { supabase } from '../lib/supabase';

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string, role: string) => Promise<boolean>;
  signup: (email: string, password: string, role: string, fullName: string, contactPhone?: string) => Promise<boolean>;
  logout: () => void;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  readonly children: ReactNode;
}

// Demo users: allow access with any password
const DEMO_USERS: Record<string, User> = {
  'patient@demo.com': {
    id: 'demo-patient',
    email: 'patient@demo.com',
    fullName: 'Test',
    role: 'patient',
    contactPhone: '123-456-7890',
    assignedLabId: 'demo-lab-a',
  },
  'lab@demo.com': {
    id: 'demo-lab-a',
    email: 'lab@demo.com',
    fullName: 'Lab A',
    role: 'lab',
    contactPhone: '123-456-7891',
    labName: 'Lab A',
  },
  'lab2@demo.com': {
    id: 'demo-lab-b',
    email: 'lab2@demo.com',
    fullName: 'Lab B',
    role: 'lab',
    contactPhone: '123-456-7892',
    labName: 'Lab B',
  },
  'admin@demo.com': {
    id: 'demo-admin',
    email: 'admin@demo.com',
    fullName: 'Sujal',
    role: 'central_admin',
    contactPhone: '123-456-7893',
  },
};

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check for existing session
    const getSession = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (session?.user) {
        await fetchUserProfile(session.user.id);
      }
      setIsLoading(false);
    };

    getSession();

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(async (event, session) => {
      if (session?.user) {
        await fetchUserProfile(session.user.id);
      } else {
        setUser(null);
      }
      setIsLoading(false);
    });

    // Fallback timeout to clear loading state
    const timeout = setTimeout(() => {
      setIsLoading(false);
    }, 2000);

    return () => {
      subscription.unsubscribe();
      clearTimeout(timeout);
    };
  }, []);

  const fetchUserProfile = async (userId: string) => {
    try {
      const { data, error } = await supabase
        .from('users')
        .select('*')
        .eq('id', userId)
        .single();

      if (error) {
        console.error('Error fetching user profile:', error);
        return;
      }

      if (data) {
        const userData: User = {
          id: data.id,
          email: data.email,
          fullName: data.full_name || '',
          role: data.role,
          contactPhone: data.contact_phone,
          assignedLabId: data.assigned_lab_id,
          labName: data.lab_name,
        };
        setUser(userData);
      }
    } catch (error) {
      console.error('Error fetching user profile:', error);
    }
  };

  const login = async (email: string, password: string, _role: string): Promise<boolean> => {
    // Demo bypass: grant access immediately for known demo emails
    const demo = DEMO_USERS[email.toLowerCase()];
    if (demo) {
      setUser(demo);
      setIsLoading(false); // Clear loading state for demo users
      return true;
    }

    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) {
        console.error('Login error:', error);
        return false;
      }

      if (data.user) {
        // Fetch user profile and trust stored role
        const { data: profileData, error: profileError } = await supabase
          .from('users')
          .select('*')
          .eq('id', data.user.id)
          .single();

        if (profileError || !profileData) {
          // Auto-provision minimal profile if missing
          const insertRes = await supabase.from('users').insert({
            id: data.user.id,
            email: data.user.email,
            role: 'patient',
            full_name: data.user.email?.split('@')[0] || 'User',
          }).select('*').single();
          if (insertRes.error || !insertRes.data) {
            await supabase.auth.signOut();
            return false;
          }
          const u: User = {
            id: insertRes.data.id,
            email: insertRes.data.email,
            fullName: insertRes.data.full_name || '',
            role: insertRes.data.role,
            contactPhone: insertRes.data.contact_phone,
            assignedLabId: insertRes.data.assigned_lab_id,
            labName: insertRes.data.lab_name,
          };
          setUser(u);
          return true;
        }

        const u: User = {
          id: profileData.id,
          email: profileData.email,
          fullName: profileData.full_name || '',
          role: profileData.role,
          contactPhone: profileData.contact_phone,
          assignedLabId: profileData.assigned_lab_id,
          labName: profileData.lab_name,
        };
        setUser(u);
        return true;
      }

      return false;
    } catch (error) {
      console.error('Login error:', error);
      return false;
    }
  };

  const signup = async (email: string, password: string, role: string, fullName: string, contactPhone?: string): Promise<boolean> => {
    // Demo bypass: allow demo emails to "sign up" as login
    const demo = DEMO_USERS[email.toLowerCase()];
    if (demo) {
      setUser(demo);
      setIsLoading(false); // Clear loading state for demo users
      return true;
    }

    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
      });

      if (error) {
        console.error('Signup error:', error);
        return false;
      }

      if (data.user) {
        // Insert user profile into users table
        const { error: profileError } = await supabase
          .from('users')
          .insert({
            id: data.user.id,
            email: data.user.email,
            role,
            full_name: fullName,
            contact_phone: contactPhone || null,
            lab_name: role === 'lab' ? fullName : null,
          });

        if (profileError) {
          console.error('Profile creation error:', profileError);
          return false;
        }

        const userData: User = {
          id: data.user.id,
          email: data.user.email || email,
          fullName,
          role,
          contactPhone,
          labName: role === 'lab' ? fullName : undefined,
        };
        setUser(userData);
        return true;
      }

      return false;
    } catch (error) {
      console.error('Signup error:', error);
      return false;
    }
  };

  const logout = () => {
    supabase.auth.signOut();
    setUser(null);
  };

  const contextValue = useMemo(() => ({
    user,
    login,
    signup,
    logout,
    isLoading,
  }), [user, isLoading]);

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
