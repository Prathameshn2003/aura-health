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

// Cache for admin role check
const adminRoleCache = new Map<string, { value: boolean; timestamp: number }>();
const CACHE_DURATION = 5 * 60 * 1000;

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAdmin, setIsAdmin] = useState(false);
  const { toast } = useToast();
  const isMountedRef = useRef(true);

  const checkAdminRole = useCallback(async (userId: string): Promise<boolean> => {
    const cached = adminRoleCache.get(userId);
    if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
      return cached.value;
    }
    try {
      const { data, error } = await supabase.rpc('has_role', {
        _user_id: userId,
        _role: 'admin'
      });
      if (error) return false;
      const result = data === true;
      adminRoleCache.set(userId, { value: result, timestamp: Date.now() });
      return result;
    } catch {
      return false;
    }
  }, []);

  const ensureProfileExists = useCallback(async (userId: string, fullName?: string) => {
    try {
      const { data } = await supabase
        .from('profiles')
        .select('id')
        .eq('id', userId)
        .maybeSingle();
      if (!data) {
        await supabase.from('profiles').insert({ id: userId, full_name: fullName || null });
      }
    } catch {
      // Profile might already exist via trigger
    }
  }, []);

  useEffect(() => {
    isMountedRef.current = true;

    // Set up listener FIRST (before getSession) per Supabase best practice
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, newSession) => {
        if (!isMountedRef.current) return;
        setSession(newSession);
        setUser(newSession?.user ?? null);

        // Non-blocking admin check for ongoing changes
        if (newSession?.user) {
          checkAdminRole(newSession.user.id).then(admin => {
            if (isMountedRef.current) setIsAdmin(admin);
          });
        } else {
          setIsAdmin(false);
        }
      }
    );

    // Initial load — await admin check before setting loading=false
    const initializeAuth = async () => {
      try {
        const { data: { session: initialSession } } = await supabase.auth.getSession();
        if (!isMountedRef.current) return;

        setSession(initialSession);
        setUser(initialSession?.user ?? null);

        if (initialSession?.user) {
          const [adminStatus] = await Promise.all([
            checkAdminRole(initialSession.user.id),
            ensureProfileExists(initialSession.user.id, initialSession.user.user_metadata?.full_name),
          ]);
          if (isMountedRef.current) setIsAdmin(adminStatus);
        }
      } catch (error) {
        console.error("Auth init error:", error);
      } finally {
        if (isMountedRef.current) setLoading(false);
      }
    };

    initializeAuth();

    return () => {
      isMountedRef.current = false;
      subscription.unsubscribe();
    };
  }, [checkAdminRole, ensureProfileExists]);

  const signUp = useCallback(async (email: string, password: string, fullName: string) => {
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          emailRedirectTo: `${window.location.origin}/dashboard`,
          data: { full_name: fullName },
        },
      });

      if (error) return { error };

      if (data.user && data.session) {
        toast({ title: "Account created!", description: "Welcome to NaariCare!" });
      } else if (data.user && !data.session) {
        toast({ title: "Account created!", description: "Please check your email to verify your account." });
      }

      return { error: null };
    } catch (error) {
      return { error: error as Error };
    }
  }, [toast]);

  const signIn = useCallback(async (email: string, password: string) => {
    try {
      const { error } = await supabase.auth.signInWithPassword({ email, password });
      return { error: error || null };
    } catch (error) {
      return { error: error as Error };
    }
  }, []);

  const signOut = useCallback(async () => {
    if (user?.id) adminRoleCache.delete(user.id);
    await supabase.auth.signOut();
    setUser(null);
    setSession(null);
    setIsAdmin(false);
    window.history.replaceState(null, '', '/login');
  }, [user?.id]);

  const resetPassword = useCallback(async (email: string) => {
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/reset-password`,
      });
      return { error: error || null };
    } catch (error) {
      return { error: error as Error };
    }
  }, []);

  const updatePassword = useCallback(async (newPassword: string) => {
    try {
      const { error } = await supabase.auth.updateUser({ password: newPassword });
      return { error: error || null };
    } catch (error) {
      return { error: error as Error };
    }
  }, []);

  return (
    <AuthContext.Provider value={{ user, session, loading, isAdmin, signUp, signIn, signOut, resetPassword, updatePassword }}>
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
