import { useState, useEffect } from "react";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { HealthDisclaimer } from "@/components/health/HealthDisclaimer";
import { SimpleCycleCalendar } from "@/components/cycle/SimpleCycleCalendar";
import { SymptomLogger, SymptomData } from "@/components/cycle/SymptomLogger";
import { CycleInsights } from "@/components/cycle/CycleInsights";
import { NotificationSettings } from "@/components/cycle/NotificationSettings";
import { NearbyDoctors } from "@/components/health/NearbyDoctors";
import { OnboardingGuide } from "@/components/cycle/OnboardingGuide";
import { QuickStartCard } from "@/components/cycle/QuickStartCard";
import { CyclePhaseIndicator } from "@/components/cycle/CyclePhaseIndicator";
import { useCycleTracking } from "@/hooks/useCycleTracking";
import { 
  Calendar, 
  Activity, 
  TrendingUp, 
  Bell, 
  Stethoscope,
  Loader2,
  Droplets,
  BookOpen,
  ArrowRight,
  CheckCircle2,
  Info,
  Sparkles,
  Heart,
} from "lucide-react";
import { format, isToday } from "date-fns";

type TabValue = "home" | "calendar" | "symptoms" | "insights" | "settings" | "doctors";

const MenstrualModule = () => {
  const {
    cycleLogs,
    settings,
    loading,
    saving,
    prediction,
    insights,
    notificationSchedule,
    logPeriod,
    endPeriod,
    logSymptoms,
    updateNotificationSettings,
    deleteAllData,
  } = useCycleTracking();

  const [activeTab, setActiveTab] = useState<TabValue>("home");
  const [selectedDate, setSelectedDate] = useState<Date>(new Date());
  const [showOnboarding, setShowOnboarding] = useState(false);

  // Check if user is new (no cycles logged)
  useEffect(() => {
    if (!loading && cycleLogs.length === 0) {
      const hasSeenOnboarding = localStorage.getItem("menstrual_onboarding_complete");
      if (!hasSeenOnboarding) {
        setShowOnboarding(true);
      }
    }
  }, [loading, cycleLogs.length]);

  const handleOnboardingComplete = () => {
    localStorage.setItem("menstrual_onboarding_complete", "true");
    setShowOnboarding(false);
    setActiveTab("calendar");
  };

  const handleLogPeriod = async (date: Date) => {
    await logPeriod(date);
    setActiveTab("symptoms");
    setSelectedDate(date);
  };

  const handleSaveSymptoms = async (symptoms: SymptomData) => {
    const symptomsRecord: Record<string, unknown> = { ...symptoms };
    await logSymptoms(selectedDate, symptomsRecord);
  };

  // Check if user logged symptoms today
  const hasLoggedToday = cycleLogs.some(log => 
    log.start_date === format(new Date(), "yyyy-MM-dd")
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <main className="pt-24 pb-16 flex items-center justify-center min-h-[60vh]">
          <div className="flex flex-col items-center gap-4">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
            <p className="text-muted-foreground">Loading your cycle data...</p>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  // Show onboarding for new users
  if (showOnboarding) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <main className="pt-20 sm:pt-24 pb-16">
          <div className="container mx-auto px-4">
            <OnboardingGuide 
              onComplete={handleOnboardingComplete}
              onSkip={handleOnboardingComplete}
            />
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="pt-20 sm:pt-24 pb-16">
        <div className="container mx-auto px-4 max-w-4xl">
          {/* Header */}
          <div className="text-center mb-6">
            <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-2xl bg-primary/20 flex items-center justify-center mx-auto mb-4 sm:mb-5">
              <Droplets className="w-8 h-8 sm:w-10 sm:h-10 text-primary" />
            </div>
            <h1 className="font-heading text-2xl sm:text-3xl md:text-4xl font-bold text-foreground mb-2">
              Your Cycle Tracker
            </h1>
            <p className="text-muted-foreground text-sm sm:text-base max-w-xl mx-auto">
              Track your period, understand your body, and get smart predictions
            </p>
          </div>

          {/* Main Tabs */}
          <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as TabValue)}>
            <TabsList className="grid grid-cols-3 sm:grid-cols-6 mb-6 h-auto p-1">
              <TabsTrigger value="home" className="gap-1.5 text-xs sm:text-sm py-2">
                <Heart className="w-4 h-4" />
                <span className="hidden sm:inline">Home</span>
              </TabsTrigger>
              <TabsTrigger value="calendar" className="gap-1.5 text-xs sm:text-sm py-2">
                <Calendar className="w-4 h-4" />
                <span className="hidden sm:inline">Calendar</span>
              </TabsTrigger>
              <TabsTrigger value="symptoms" className="gap-1.5 text-xs sm:text-sm py-2">
                <Activity className="w-4 h-4" />
                <span className="hidden sm:inline">Symptoms</span>
              </TabsTrigger>
              <TabsTrigger value="insights" className="gap-1.5 text-xs sm:text-sm py-2">
                <TrendingUp className="w-4 h-4" />
                <span className="hidden sm:inline">Insights</span>
              </TabsTrigger>
              <TabsTrigger value="settings" className="gap-1.5 text-xs sm:text-sm py-2">
                <Bell className="w-4 h-4" />
                <span className="hidden sm:inline">Reminders</span>
              </TabsTrigger>
              <TabsTrigger value="doctors" className="gap-1.5 text-xs sm:text-sm py-2">
                <Stethoscope className="w-4 h-4" />
                <span className="hidden sm:inline">Doctors</span>
              </TabsTrigger>
            </TabsList>

            {/* Home Tab - Dashboard View */}
            <TabsContent value="home" className="space-y-6 animate-fade-up">
              {/* Prediction Banner */}
              {prediction && prediction.days_until > 0 && (
                <div className="glass-card rounded-xl p-4 bg-gradient-to-r from-primary/10 to-accent/10 border border-primary/20">
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
                      <Sparkles className="w-7 h-7 text-primary" />
                    </div>
                    <div className="flex-1">
                      <p className="font-heading text-lg font-semibold text-foreground">
                        {prediction.days_until <= 3 
                          ? "Period Coming Soon!" 
                          : `~${prediction.days_until} days until period`}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Expected around {format(new Date(prediction.predicted_start_date), "MMMM d")}
                        {prediction.confidence_level === "high" && " 🎯"}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Cycle Phase Indicator */}
              <CyclePhaseIndicator
                lastPeriodStart={cycleLogs.length > 0 ? cycleLogs[0].start_date : null}
                averageCycleLength={insights.averageCycleLength}
                averagePeriodLength={insights.averagePeriodLength}
                prediction={prediction}
              />

              {/* Quick Start Card */}
              <QuickStartCard
                cycleCount={cycleLogs.length}
                hasLoggedToday={hasLoggedToday}
                onLogPeriod={() => setActiveTab("calendar")}
                onLogSymptoms={() => setActiveTab("symptoms")}
                onViewInsights={() => setActiveTab("insights")}
              />

              {/* Quick Stats */}
              {cycleLogs.length >= 2 && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="glass-card rounded-xl p-4 text-center">
                    <div className="text-2xl font-bold text-foreground">{insights.averageCycleLength}</div>
                    <div className="text-xs text-muted-foreground">Cycle Days</div>
                  </div>
                  <div className="glass-card rounded-xl p-4 text-center">
                    <div className="text-2xl font-bold text-foreground">{insights.averagePeriodLength}</div>
                    <div className="text-xs text-muted-foreground">Period Days</div>
                  </div>
                  <div className="glass-card rounded-xl p-4 text-center">
                    <div className={`text-2xl font-bold ${insights.isRegular ? "text-teal" : "text-accent"}`}>
                      {insights.isRegular ? "✓" : "~"}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {insights.isRegular ? "Regular" : "Variable"}
                    </div>
                  </div>
                  <div className="glass-card rounded-xl p-4 text-center">
                    <div className="text-2xl font-bold text-foreground">{cycleLogs.length}</div>
                    <div className="text-xs text-muted-foreground">Cycles Tracked</div>
                  </div>
                </div>
              )}

              {/* Help Section */}
              <div className="glass-card rounded-xl p-4 bg-muted/30">
                <div className="flex items-start gap-3">
                  <Info className="w-5 h-5 text-muted-foreground mt-0.5" />
                  <div>
                    <p className="font-medium text-foreground text-sm mb-1">Need Help?</p>
                    <p className="text-xs text-muted-foreground mb-3">
                      Understanding your cycle is a journey. The more you track, the better your predictions become!
                    </p>
                    <Button 
                      variant="link" 
                      className="p-0 h-auto text-xs text-primary"
                      onClick={() => {
                        localStorage.removeItem("menstrual_onboarding_complete");
                        setShowOnboarding(true);
                      }}
                    >
                      View Tutorial Again <ArrowRight className="w-3 h-3 ml-1" />
                    </Button>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Calendar Tab */}
            <TabsContent value="calendar" className="animate-fade-up">
              {cycleLogs.length === 0 && (
                <div className="text-center py-8 mb-6 glass-card rounded-xl">
                  <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-4">
                    <Calendar className="w-8 h-8 text-primary" />
                  </div>
                  <h3 className="font-heading text-xl font-semibold text-foreground mb-2">
                    Log Your First Period
                  </h3>
                  <p className="text-muted-foreground text-sm max-w-sm mx-auto mb-4">
                    Tap any date on the calendar below to mark when your period started. You can log past periods too!
                  </p>
                </div>
              )}
              
              <SimpleCycleCalendar
                cycleLogs={cycleLogs}
                prediction={prediction}
                averageCycleLength={insights.averageCycleLength}
                onLogPeriod={handleLogPeriod}
                onSelectDate={setSelectedDate}
              />
            </TabsContent>

            {/* Symptoms Tab */}
            <TabsContent value="symptoms" className="animate-fade-up">
              <SymptomLogger
                date={selectedDate}
                onSave={handleSaveSymptoms}
                saving={saving}
                initialData={cycleLogs.find(log => log.start_date === format(selectedDate, "yyyy-MM-dd")) as SymptomData | undefined}
              />
            </TabsContent>

            {/* Insights Tab */}
            <TabsContent value="insights" className="animate-fade-up">
              {cycleLogs.length < 2 ? (
                <div className="text-center py-12 glass-card rounded-xl">
                  <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
                    <TrendingUp className="w-10 h-10 text-muted-foreground" />
                  </div>
                  <h3 className="font-heading text-xl font-semibold text-foreground mb-2">
                    More Data Needed
                  </h3>
                  <p className="text-muted-foreground mb-2 max-w-md mx-auto">
                    Log at least <span className="font-semibold text-foreground">2 complete cycles</span> to see personalized insights.
                  </p>
                  <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground mb-6">
                    <div className="flex items-center gap-1">
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center ${cycleLogs.length >= 1 ? "bg-teal/20" : "bg-muted"}`}>
                        {cycleLogs.length >= 1 ? (
                          <CheckCircle2 className="w-4 h-4 text-teal" />
                        ) : (
                          <span className="text-xs">1</span>
                        )}
                      </div>
                    </div>
                    <div className="w-8 h-0.5 bg-muted" />
                    <div className="flex items-center gap-1">
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center ${cycleLogs.length >= 2 ? "bg-teal/20" : "bg-muted"}`}>
                        {cycleLogs.length >= 2 ? (
                          <CheckCircle2 className="w-4 h-4 text-teal" />
                        ) : (
                          <span className="text-xs">2</span>
                        )}
                      </div>
                    </div>
                  </div>
                  <Button onClick={() => setActiveTab("calendar")} variant="outline">
                    <Calendar className="w-4 h-4 mr-2" />
                    Go to Calendar
                  </Button>
                </div>
              ) : (
                <CycleInsights insights={insights} cycleCount={cycleLogs.length} />
              )}
            </TabsContent>

            {/* Settings Tab */}
            <TabsContent value="settings" className="animate-fade-up">
              <NotificationSettings
                settings={settings}
                notificationSchedule={notificationSchedule}
                isIrregular={!insights.isRegular}
                onUpdate={updateNotificationSettings}
                onDeleteData={deleteAllData}
                saving={saving}
              />
            </TabsContent>

            {/* Doctors Tab */}
            <TabsContent value="doctors" className="animate-fade-up">
              <div className="space-y-6">
                <div className="text-center mb-6">
                  <h2 className="font-heading text-xl font-semibold text-foreground mb-2">
                    Find Nearby Specialists
                  </h2>
                  <p className="text-muted-foreground text-sm">
                    Connect with qualified gynecologists for professional care
                  </p>
                </div>
                <NearbyDoctors specialty="gynecologist" />
              </div>
            </TabsContent>
          </Tabs>

          {/* Global Disclaimer */}
          <div className="mt-8">
            <HealthDisclaimer />
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default MenstrualModule;
