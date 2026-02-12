import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Bell,
  BellOff,
  Clock,
  Eye,
  EyeOff,
  Shield,
  Trash2,
  Loader2,
  CheckCircle2,
} from "lucide-react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

interface NotificationSettingsProps {
  settings: {
    notification_enabled: boolean;
    reminder_days: number[];
    notification_time: string;
    hide_notification_text: boolean;
    allow_advanced_analysis: boolean;
  } | null;
  notificationSchedule: { date: string; message: string; daysBefor: number }[];
  isIrregular: boolean;
  onUpdate: (updates: Partial<{
    notification_enabled: boolean;
    reminder_days: number[];
    notification_time: string;
    hide_notification_text: boolean;
    allow_advanced_analysis: boolean;
  }>) => Promise<void>;
  onDeleteData: () => Promise<void>;
  saving: boolean;
}

export function NotificationSettings({
  settings,
  notificationSchedule,
  isIrregular,
  onUpdate,
  onDeleteData,
  saving,
}: NotificationSettingsProps) {
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  if (!settings) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const reminderDays = isIrregular ? [5, 3, 1] : [3, 2, 1];

  return (
    <div className="space-y-6">
      {/* Notifications Toggle */}
      <Card className="glass-card border-0">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            {settings.notification_enabled ? (
              <Bell className="w-5 h-5 text-primary" />
            ) : (
              <BellOff className="w-5 h-5 text-muted-foreground" />
            )}
            Period Reminders
          </CardTitle>
          <CardDescription>
            Get gentle reminders before your predicted period
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="notifications" className="flex items-center gap-2 cursor-pointer">
              <span>Enable Reminders</span>
            </Label>
            <Switch
              id="notifications"
              checked={settings.notification_enabled}
              onCheckedChange={(checked) => onUpdate({ notification_enabled: checked })}
              disabled={saving}
            />
          </div>

          {settings.notification_enabled && (
            <>
              <div className="border-t border-border pt-4">
                <Label className="text-sm font-medium mb-3 block">Reminder Schedule</Label>
                <div className="flex flex-wrap gap-2">
                  {reminderDays.map(day => (
                    <Badge key={day} variant="secondary" className="gap-1">
                      <Clock className="w-3 h-3" />
                      {day} day{day > 1 ? "s" : ""} before
                    </Badge>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  {isIrregular 
                    ? "Using 5-3-1 day schedule for irregular cycles"
                    : "Using 3-2-1 day schedule for regular cycles"}
                </p>
              </div>

              <div className="border-t border-border pt-4">
                <div className="flex items-center justify-between">
                  <Label htmlFor="hide-text" className="flex items-center gap-2 cursor-pointer">
                    {settings.hide_notification_text ? (
                      <EyeOff className="w-4 h-4 text-muted-foreground" />
                    ) : (
                      <Eye className="w-4 h-4 text-muted-foreground" />
                    )}
                    <span>Private Notifications</span>
                  </Label>
                  <Switch
                    id="hide-text"
                    checked={settings.hide_notification_text}
                    onCheckedChange={(checked) => onUpdate({ hide_notification_text: checked })}
                    disabled={saving}
                  />
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {settings.hide_notification_text 
                    ? "Notification text will be hidden on lock screen"
                    : "Full notification text will be visible"}
                </p>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Upcoming Notifications Preview */}
      {settings.notification_enabled && notificationSchedule.length > 0 && (
        <Card className="glass-card border-0">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <CheckCircle2 className="w-5 h-5 text-teal" />
              Upcoming Reminders
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {notificationSchedule.map((notification, i) => (
                <div key={i} className="flex items-start gap-3 p-3 rounded-lg bg-muted/50">
                  <Bell className="w-4 h-4 text-primary flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-foreground">{notification.date}</p>
                    <p className="text-xs text-muted-foreground">
                      {settings.hide_notification_text 
                        ? "🌸 NaariCare Reminder"
                        : notification.message}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Advanced Analysis */}
      <Card className="glass-card border-0">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Shield className="w-5 h-5 text-purple-500" />
            Advanced Analysis
          </CardTitle>
          <CardDescription>
            Enable AI-powered insights and pattern detection
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <Label htmlFor="advanced" className="flex items-center gap-2 cursor-pointer">
              <span>Enable Advanced Insights</span>
            </Label>
            <Switch
              id="advanced"
              checked={settings.allow_advanced_analysis}
              onCheckedChange={(checked) => onUpdate({ allow_advanced_analysis: checked })}
              disabled={saving}
            />
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            {settings.allow_advanced_analysis 
              ? "Stress/sleep correlations and PCOS risk flags are enabled"
              : "Only basic cycle tracking is active"}
          </p>
        </CardContent>
      </Card>

      {/* Privacy & Data */}
      <Card className="glass-card border-0 border-destructive/20">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base text-destructive">
            <Trash2 className="w-5 h-5" />
            Data Privacy
          </CardTitle>
          <CardDescription>
            Your health data is encrypted and stored securely
          </CardDescription>
        </CardHeader>
        <CardContent>
          <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
            <AlertDialogTrigger asChild>
              <Button variant="destructive" size="sm" className="w-full">
                Delete All My Cycle Data
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete All Cycle Data?</AlertDialogTitle>
                <AlertDialogDescription>
                  This will permanently delete all your cycle logs, predictions, and settings. 
                  This action cannot be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={() => {
                    onDeleteData();
                    setShowDeleteDialog(false);
                  }}
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                >
                  Delete Everything
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
          <p className="text-xs text-muted-foreground mt-3 text-center">
            Your data is never shared with third parties.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
