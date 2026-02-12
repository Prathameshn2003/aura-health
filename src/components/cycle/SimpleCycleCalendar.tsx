import { useState, useMemo } from "react";
import { Calendar } from "@/components/ui/calendar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { 
  Droplets, 
  Plus,
  HelpCircle,
  CheckCircle2,
  Calendar as CalendarIcon,
  ChevronRight,
} from "lucide-react";
import { format, isSameDay, parseISO, addDays, differenceInDays } from "date-fns";

interface CycleLog {
  id: string;
  start_date: string;
  end_date: string | null;
  period_length: number | null;
  flow_intensity: string | null;
}

interface CyclePrediction {
  predicted_start_date: string;
  predicted_end_date: string | null;
  confidence_level: "high" | "medium" | "low";
  days_until: number;
}

interface SimpleCycleCalendarProps {
  cycleLogs: CycleLog[];
  prediction: CyclePrediction | null;
  averageCycleLength: number;
  onLogPeriod: (date: Date) => void;
  onSelectDate: (date: Date) => void;
}

export function SimpleCycleCalendar({
  cycleLogs,
  prediction,
  averageCycleLength,
  onLogPeriod,
  onSelectDate,
}: SimpleCycleCalendarProps) {
  const [selectedDate, setSelectedDate] = useState<Date | undefined>(new Date());
  const [showLogDialog, setShowLogDialog] = useState(false);
  const [dateToLog, setDateToLog] = useState<Date | null>(null);
  const [showHelpDialog, setShowHelpDialog] = useState(false);

  // Calculate date modifiers
  const dateModifiers = useMemo(() => {
    const periodDays: Date[] = [];
    const ovulationDays: Date[] = [];
    const predictedDays: Date[] = [];

    cycleLogs.forEach(log => {
      const startDate = parseISO(log.start_date);
      const periodLength = log.period_length || 5;
      
      for (let i = 0; i < periodLength; i++) {
        periodDays.push(addDays(startDate, i));
      }

      if (averageCycleLength > 0) {
        const ovulationDay = Math.round(averageCycleLength / 2) - 2;
        for (let i = 0; i < 5; i++) {
          ovulationDays.push(addDays(startDate, ovulationDay + i));
        }
      }
    });

    if (prediction) {
      const predictedStart = parseISO(prediction.predicted_start_date);
      const predictedEnd = prediction.predicted_end_date 
        ? parseISO(prediction.predicted_end_date)
        : addDays(predictedStart, 4);

      const predictedLength = differenceInDays(predictedEnd, predictedStart) + 1;
      for (let i = 0; i < predictedLength; i++) {
        predictedDays.push(addDays(predictedStart, i));
      }
    }

    return { periodDays, ovulationDays, predictedDays };
  }, [cycleLogs, prediction, averageCycleLength]);

  const handleDateClick = (date: Date | undefined) => {
    if (!date) return;
    setSelectedDate(date);
    
    const isLoggedPeriod = dateModifiers.periodDays.some(d => isSameDay(d, date));
    
    if (!isLoggedPeriod) {
      setDateToLog(date);
      setShowLogDialog(true);
    } else {
      onSelectDate(date);
    }
  };

  const confirmLogPeriod = () => {
    if (dateToLog) {
      onLogPeriod(dateToLog);
      setShowLogDialog(false);
      setDateToLog(null);
    }
  };

  return (
    <div className="space-y-4">
      {/* Instructions Card */}
      <Card className="glass-card border-0 bg-primary/5 border-l-4 border-l-primary">
        <CardContent className="p-4 flex items-start gap-3">
          <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
            <HelpCircle className="w-5 h-5 text-primary" />
          </div>
          <div className="flex-1">
            <p className="font-medium text-foreground text-sm mb-1">
              How to use the calendar
            </p>
            <p className="text-xs text-muted-foreground">
              Tap any date to log your period start. Colored dates show your tracked cycles and predictions.
            </p>
          </div>
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={() => setShowHelpDialog(true)}
            className="text-xs"
          >
            Learn More
          </Button>
        </CardContent>
      </Card>

      {/* Calendar */}
      <Card className="glass-card border-0">
        <CardContent className="p-4 md:p-6">
          <Calendar
            mode="single"
            selected={selectedDate}
            onSelect={handleDateClick}
            className="w-full"
            modifiers={{
              period: dateModifiers.periodDays,
              ovulation: dateModifiers.ovulationDays,
              predicted: dateModifiers.predictedDays,
            }}
            modifiersStyles={{
              period: {
                backgroundColor: "hsl(var(--primary) / 0.3)",
                borderRadius: "50%",
                color: "hsl(var(--foreground))",
              },
              ovulation: {
                backgroundColor: "hsl(var(--teal) / 0.25)",
                borderRadius: "50%",
              },
              predicted: {
                backgroundColor: "hsl(var(--primary) / 0.15)",
                borderRadius: "50%",
                border: "2px dashed hsl(var(--primary) / 0.5)",
              },
            }}
          />
        </CardContent>
      </Card>

      {/* Legend - More Visual */}
      <Card className="glass-card border-0">
        <CardContent className="p-4">
          <p className="text-xs font-medium text-muted-foreground mb-3 uppercase tracking-wide">
            Calendar Legend
          </p>
          <div className="grid grid-cols-2 gap-3">
            <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/30">
              <div className="w-6 h-6 rounded-full bg-primary/30" />
              <div>
                <p className="text-xs font-medium text-foreground">Period</p>
                <p className="text-[10px] text-muted-foreground">Your logged days</p>
              </div>
            </div>
            <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/30">
              <div className="w-6 h-6 rounded-full bg-teal/25" />
              <div>
                <p className="text-xs font-medium text-foreground">Ovulation</p>
                <p className="text-[10px] text-muted-foreground">Fertile window</p>
              </div>
            </div>
            <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/30">
              <div className="w-6 h-6 rounded-full border-2 border-dashed border-primary/50 bg-primary/15" />
              <div>
                <p className="text-xs font-medium text-foreground">Predicted</p>
                <p className="text-[10px] text-muted-foreground">Expected period</p>
              </div>
            </div>
            <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/30">
              <div className="w-6 h-6 rounded-full bg-accent" />
              <div>
                <p className="text-xs font-medium text-foreground">Today</p>
                <p className="text-[10px] text-muted-foreground">Current date</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Log Period Dialog */}
      <Dialog open={showLogDialog} onOpenChange={setShowLogDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-4">
              <Droplets className="w-8 h-8 text-primary" />
            </div>
            <DialogTitle className="text-center text-xl">
              Log Period Start?
            </DialogTitle>
            <DialogDescription className="text-center">
              {dateToLog && (
                <>
                  Did your period start on{" "}
                  <span className="font-medium text-foreground">
                    {format(dateToLog, "EEEE, MMMM d, yyyy")}
                  </span>
                  ?
                </>
              )}
            </DialogDescription>
          </DialogHeader>
          
          <div className="py-4 space-y-3">
            <div className="flex items-start gap-3 p-3 rounded-xl bg-muted/50">
              <CheckCircle2 className="w-5 h-5 text-teal mt-0.5" />
              <div className="text-sm">
                <p className="font-medium text-foreground">What happens next?</p>
                <p className="text-muted-foreground text-xs">
                  You'll be able to log symptoms and track this cycle's details.
                </p>
              </div>
            </div>
          </div>

          <DialogFooter className="flex gap-2 sm:gap-2">
            <Button 
              variant="outline" 
              onClick={() => setShowLogDialog(false)}
              className="flex-1"
            >
              Not Today
            </Button>
            <Button 
              onClick={confirmLogPeriod} 
              className="flex-1 gap-2"
            >
              <Plus className="w-4 h-4" />
              Yes, Log It
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Help Dialog */}
      <Dialog open={showHelpDialog} onOpenChange={setShowHelpDialog}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <CalendarIcon className="w-5 h-5 text-primary" />
              How to Use the Calendar
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-sm font-bold text-primary">1</span>
              </div>
              <div>
                <p className="font-medium text-foreground">Tap to Log</p>
                <p className="text-sm text-muted-foreground">
                  Tap any date to mark when your period started. You can log past periods too!
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-sm font-bold text-primary">2</span>
              </div>
              <div>
                <p className="font-medium text-foreground">View Patterns</p>
                <p className="text-sm text-muted-foreground">
                  Colored dates show your cycle history. Pink = period, teal = ovulation window.
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-sm font-bold text-primary">3</span>
              </div>
              <div>
                <p className="font-medium text-foreground">See Predictions</p>
                <p className="text-sm text-muted-foreground">
                  After 2+ cycles, dashed circles show when your next period is expected.
                </p>
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button onClick={() => setShowHelpDialog(false)} className="w-full">
              Got It!
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
