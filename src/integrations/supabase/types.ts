export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "14.1"
  }
  public: {
    Tables: {
      cycle_logs: {
        Row: {
          acne: string | null
          actual_vs_predicted: number | null
          bloating: string | null
          breast_tenderness: string | null
          cramps: string | null
          created_at: string
          cycle_length: number | null
          end_date: string | null
          fatigue: string | null
          flow_intensity: string | null
          hair_growth: string | null
          hair_loss: string | null
          headache: string | null
          id: string
          is_period_end: boolean | null
          is_period_start: boolean | null
          mood: string | null
          notes: string | null
          period_length: number | null
          physical_activity: string | null
          predicted_start: string | null
          sleep_hours: number | null
          start_date: string
          stress_level: number | null
          symptoms: Json | null
          updated_at: string
          user_id: string
        }
        Insert: {
          acne?: string | null
          actual_vs_predicted?: number | null
          bloating?: string | null
          breast_tenderness?: string | null
          cramps?: string | null
          created_at?: string
          cycle_length?: number | null
          end_date?: string | null
          fatigue?: string | null
          flow_intensity?: string | null
          hair_growth?: string | null
          hair_loss?: string | null
          headache?: string | null
          id?: string
          is_period_end?: boolean | null
          is_period_start?: boolean | null
          mood?: string | null
          notes?: string | null
          period_length?: number | null
          physical_activity?: string | null
          predicted_start?: string | null
          sleep_hours?: number | null
          start_date: string
          stress_level?: number | null
          symptoms?: Json | null
          updated_at?: string
          user_id: string
        }
        Update: {
          acne?: string | null
          actual_vs_predicted?: number | null
          bloating?: string | null
          breast_tenderness?: string | null
          cramps?: string | null
          created_at?: string
          cycle_length?: number | null
          end_date?: string | null
          fatigue?: string | null
          flow_intensity?: string | null
          hair_growth?: string | null
          hair_loss?: string | null
          headache?: string | null
          id?: string
          is_period_end?: boolean | null
          is_period_start?: boolean | null
          mood?: string | null
          notes?: string | null
          period_length?: number | null
          physical_activity?: string | null
          predicted_start?: string | null
          sleep_hours?: number | null
          start_date?: string
          stress_level?: number | null
          symptoms?: Json | null
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      cycle_predictions: {
        Row: {
          actual_start_date: string | null
          based_on_cycles: number | null
          confidence_level: string | null
          created_at: string
          cycle_length_used: number | null
          deviation_days: number | null
          id: string
          is_active: boolean | null
          predicted_end_date: string | null
          predicted_start_date: string
          prediction_method: string | null
          user_id: string
          was_accurate: boolean | null
        }
        Insert: {
          actual_start_date?: string | null
          based_on_cycles?: number | null
          confidence_level?: string | null
          created_at?: string
          cycle_length_used?: number | null
          deviation_days?: number | null
          id?: string
          is_active?: boolean | null
          predicted_end_date?: string | null
          predicted_start_date: string
          prediction_method?: string | null
          user_id: string
          was_accurate?: boolean | null
        }
        Update: {
          actual_start_date?: string | null
          based_on_cycles?: number | null
          confidence_level?: string | null
          created_at?: string
          cycle_length_used?: number | null
          deviation_days?: number | null
          id?: string
          is_active?: boolean | null
          predicted_end_date?: string | null
          predicted_start_date?: string
          prediction_method?: string | null
          user_id?: string
          was_accurate?: boolean | null
        }
        Relationships: []
      }
      doctors: {
        Row: {
          created_at: string
          description: string | null
          email: string | null
          hospital: string | null
          id: string
          image_url: string | null
          is_active: boolean | null
          location: string | null
          name: string
          phone: string | null
          specialization: string
          updated_at: string
        }
        Insert: {
          created_at?: string
          description?: string | null
          email?: string | null
          hospital?: string | null
          id?: string
          image_url?: string | null
          is_active?: boolean | null
          location?: string | null
          name: string
          phone?: string | null
          specialization: string
          updated_at?: string
        }
        Update: {
          created_at?: string
          description?: string | null
          email?: string | null
          hospital?: string | null
          id?: string
          image_url?: string | null
          is_active?: boolean | null
          location?: string | null
          name?: string
          phone?: string | null
          specialization?: string
          updated_at?: string
        }
        Relationships: []
      }
      health_assessments: {
        Row: {
          assessment_type: string
          created_at: string
          id: string
          recommendations: Json | null
          responses: Json | null
          risk_category: string | null
          risk_score: number | null
          user_id: string
        }
        Insert: {
          assessment_type: string
          created_at?: string
          id?: string
          recommendations?: Json | null
          responses?: Json | null
          risk_category?: string | null
          risk_score?: number | null
          user_id: string
        }
        Update: {
          assessment_type?: string
          created_at?: string
          id?: string
          recommendations?: Json | null
          responses?: Json | null
          risk_category?: string | null
          risk_score?: number | null
          user_id?: string
        }
        Relationships: []
      }
      health_resources: {
        Row: {
          category: string
          created_at: string
          description: string | null
          external_link: string | null
          id: string
          is_active: boolean | null
          status: string
          title: string
          updated_at: string
        }
        Insert: {
          category: string
          created_at?: string
          description?: string | null
          external_link?: string | null
          id?: string
          is_active?: boolean | null
          status?: string
          title: string
          updated_at?: string
        }
        Update: {
          category?: string
          created_at?: string
          description?: string | null
          external_link?: string | null
          id?: string
          is_active?: boolean | null
          status?: string
          title?: string
          updated_at?: string
        }
        Relationships: []
      }
      ngos: {
        Row: {
          contact_email: string | null
          contact_phone: string | null
          created_at: string
          description: string | null
          focus_area: string | null
          id: string
          image_url: string | null
          is_active: boolean | null
          location: string | null
          name: string
          updated_at: string
          website: string | null
        }
        Insert: {
          contact_email?: string | null
          contact_phone?: string | null
          created_at?: string
          description?: string | null
          focus_area?: string | null
          id?: string
          image_url?: string | null
          is_active?: boolean | null
          location?: string | null
          name: string
          updated_at?: string
          website?: string | null
        }
        Update: {
          contact_email?: string | null
          contact_phone?: string | null
          created_at?: string
          description?: string | null
          focus_area?: string | null
          id?: string
          image_url?: string | null
          is_active?: boolean | null
          location?: string | null
          name?: string
          updated_at?: string
          website?: string | null
        }
        Relationships: []
      }
      profiles: {
        Row: {
          avatar_url: string | null
          created_at: string
          date_of_birth: string | null
          full_name: string | null
          id: string
          updated_at: string
        }
        Insert: {
          avatar_url?: string | null
          created_at?: string
          date_of_birth?: string | null
          full_name?: string | null
          id: string
          updated_at?: string
        }
        Update: {
          avatar_url?: string | null
          created_at?: string
          date_of_birth?: string | null
          full_name?: string | null
          id?: string
          updated_at?: string
        }
        Relationships: []
      }
      schemes: {
        Row: {
          benefits: string | null
          category: string | null
          created_at: string
          description: string | null
          eligibility: string | null
          how_to_apply: string | null
          id: string
          is_active: boolean | null
          name: string
          updated_at: string
          website: string | null
        }
        Insert: {
          benefits?: string | null
          category?: string | null
          created_at?: string
          description?: string | null
          eligibility?: string | null
          how_to_apply?: string | null
          id?: string
          is_active?: boolean | null
          name: string
          updated_at?: string
          website?: string | null
        }
        Update: {
          benefits?: string | null
          category?: string | null
          created_at?: string
          description?: string | null
          eligibility?: string | null
          how_to_apply?: string | null
          id?: string
          is_active?: boolean | null
          name?: string
          updated_at?: string
          website?: string | null
        }
        Relationships: []
      }
      user_cycle_settings: {
        Row: {
          allow_advanced_analysis: boolean | null
          average_cycle_length: number | null
          average_period_length: number | null
          created_at: string
          cycle_variability: number | null
          hide_notification_text: boolean | null
          id: string
          last_calculated_at: string | null
          notification_enabled: boolean | null
          notification_time: string | null
          pcos_risk_flag: boolean | null
          pcos_risk_score: number | null
          reminder_days: number[] | null
          timezone: string | null
          updated_at: string
          user_id: string
        }
        Insert: {
          allow_advanced_analysis?: boolean | null
          average_cycle_length?: number | null
          average_period_length?: number | null
          created_at?: string
          cycle_variability?: number | null
          hide_notification_text?: boolean | null
          id?: string
          last_calculated_at?: string | null
          notification_enabled?: boolean | null
          notification_time?: string | null
          pcos_risk_flag?: boolean | null
          pcos_risk_score?: number | null
          reminder_days?: number[] | null
          timezone?: string | null
          updated_at?: string
          user_id: string
        }
        Update: {
          allow_advanced_analysis?: boolean | null
          average_cycle_length?: number | null
          average_period_length?: number | null
          created_at?: string
          cycle_variability?: number | null
          hide_notification_text?: boolean | null
          id?: string
          last_calculated_at?: string | null
          notification_enabled?: boolean | null
          notification_time?: string | null
          pcos_risk_flag?: boolean | null
          pcos_risk_score?: number | null
          reminder_days?: number[] | null
          timezone?: string | null
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      user_roles: {
        Row: {
          created_at: string
          id: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          role?: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Update: {
          created_at?: string
          id?: string
          role?: Database["public"]["Enums"]["app_role"]
          user_id?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      has_role: {
        Args: {
          _role: Database["public"]["Enums"]["app_role"]
          _user_id: string
        }
        Returns: boolean
      }
    }
    Enums: {
      app_role: "admin" | "user"
      error_severity: "info" | "warning" | "error" | "critical"
      error_status:
        | "detected"
        | "analyzed"
        | "fix_applied"
        | "verified"
        | "closed"
      service_status: "healthy" | "warning" | "down"
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {
      app_role: ["admin", "user"],
      error_severity: ["info", "warning", "error", "critical"],
      error_status: [
        "detected",
        "analyzed",
        "fix_applied",
        "verified",
        "closed",
      ],
      service_status: ["healthy", "warning", "down"],
    },
  },
} as const
