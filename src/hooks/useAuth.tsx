import { createContext, useContext, useEffect, useState, useCallback, useRef, ReactNode } from "react";
import { User, Session } from "@supabase/supabase-js";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";

interface AuthContextType {
  user: User | null;
  session: Session | null;
  loading: boolean;
  isAdmin: boolean;
  signUp: (email: string, password: string, fullName: string) => Promise<{ error: Error | null }>;
  signIn: (email: string, password: string) => Promise<{ error: Error | null }>;
  signOut: () => Promise<void>;
  resetPassword: (email: string) => Promise<{ error: Error | null }>;
  updatePassword: (newPassword: string) => Promise<{ error: Error | null }>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Cache for admin role check to avoid duplicate calls
const adminRoleCache = new Map<string, { value: boolean; timestamp: number }>();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAdmin, setIsAdmin] = useState(false);
  const { toast } = useToast();
  
  // Refs to track processed sessions and prevent duplicate calls
  const processedSessionRef = useRef<string | null>(null);
  const isInitializedRef = useRef(false);

  // Check admin role from database using has_role function with caching
  const checkAdminRole = useCallback(async (userId: string): Promise<boolean> => {
    // Check cache first
    const cached = adminRoleCache.get(userId);
    if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
      return cached.value;
    }

    try {
      const { data, error } = await supabase.rpc('has_role', {
        _user_id: userId,
        _role: 'admin'
      });
      
      if (error) {
        console.error('Error checking admin role:', error);
        return false;
      }
      
      const isAdminRole = data === true;
      // Cache the result
      adminRoleCache.set(userId, { value: isAdminRole, timestamp: Date.now() });
      return isAdminRole;
    } catch (error) {
      console.error('Error checking admin role:', error);
      return false;
    }
  }, []);

  // Ensure profile exists for user - only run once per session
  const ensureProfileExists = useCallback(async (userId: string, fullName?: string) => {
    try {
      const { data: existingProfile } = await supabase
        .from('profiles')
        .select('id')
        .eq('id', userId)
        .single();

      if (!existingProfile) {
        await supabase
          .from('profiles')
          .insert({ 
            id: userId,
            full_name: fullName || null
          });
      }
    } catch (error) {
      // Profile might already exist, ignore error
    }
  }, []);

  // Process session only once per unique session
  const processSession = useCallback(async (newSession: Session | null) => {
    const sessionId = newSession?.access_token || null;
    
    // Skip if we've already processed this session
    if (sessionId === processedSessionRef.current) {
      return;
    }
    
    processedSessionRef.current = sessionId;
    setSession(newSession);
    setUser(newSession?.user ?? null);
    
    if (newSession?.user) {
      const adminStatus = await checkAdminRole(newSession.user.id);
      setIsAdmin(adminStatus);
      
      // Only ensure profile on initial sign-in
      if (!isInitializedRef.current) {
        ensureProfileExists(newSession.user.id, newSession.user.user_metadata?.full_name);
      }
    } else {
      setIsAdmin(false);
    }
    
    setLoading(false);
    isInitializedRef.current = true;
  }, [checkAdminRole, ensureProfileExists]);

  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      processSession(session);
    });

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        processSession(session);
      }
    );

    return () => subscription.unsubscribe();
  }, [processSession]);

  const signUp = useCallback(async (email: string, password: string, fullName: string) => {
    try {
      const redirectUrl = `${window.location.origin}/dashboard`;
      
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          emailRedirectTo: redirectUrl,
          data: {
            full_name: fullName,
          },
        },
      });

      if (error) {
        return { error };
      }

      if (data.user && data.session) {
        toast({
          title: "Account created!",
          description: "Welcome to NaariCare!",
        });
      } else if (data.user && !data.session) {
        toast({
          title: "Account created!",
          description: "Please check your email to verify your account.",
        });
      }

      return { error: null };
    } catch (error) {
      return { error: error as Error };
    }
  }, [toast]);

  const signIn = useCallback(async (email: string, password: string) => {
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) {
        return { error };
      }

      return { error: null };
    } catch (error) {
      return { error: error as Error };
    }
  }, []);

  const signOut = useCallback(async () => {
    // Clear cache for current user
    if (user?.id) {
      adminRoleCache.delete(user.id);
    }
    processedSessionRef.current = null;
    
    await supabase.auth.signOut();
    setUser(null);
    setSession(null);
    setIsAdmin(false);
    
    window.history.replaceState(null, '', '/login');
  }, [user?.id]);

  const resetPassword = useCallback(async (email: string) => {
    try {
      const redirectUrl = `${window.location.origin}/reset-password`;
      
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: redirectUrl,
      });

      if (error) {
        return { error };
      }

      return { error: null };
    } catch (error) {
      return { error: error as Error };
    }
  }, []);

  const updatePassword = useCallback(async (newPassword: string) => {
    try {
      const { error } = await supabase.auth.updateUser({
        password: newPassword,
      });

      if (error) {
        return { error };
      }

      return { error: null };
    } catch (error) {
      return { error: error as Error };
    }
  }, []);

  return (
    <AuthContext.Provider value={{ 
      user, 
      session, 
      loading, 
      isAdmin, 
      signUp, 
      signIn, 
      signOut,
      resetPassword,
      updatePassword
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
