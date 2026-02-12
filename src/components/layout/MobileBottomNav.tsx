import { NavLink, useLocation } from "react-router-dom";
import { 
  LayoutDashboard, 
  BookOpen, 
  MessageCircle, 
  User,
  Heart
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { icon: LayoutDashboard, label: "Home", path: "/dashboard" },
  { icon: BookOpen, label: "Modules", path: "/modules" },
  { icon: MessageCircle, label: "AI Chat", path: "/chatbot" },
  { icon: Heart, label: "Wellness", path: "/hygiene" },
  { icon: User, label: "Profile", path: "/profile" },
];

export const MobileBottomNav = () => {
  const location = useLocation();

  const isActive = (path: string) => {
    if (path === "/dashboard") {
      return location.pathname === "/dashboard";
    }
    return location.pathname.startsWith(path);
  };

  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 md:hidden bg-card/95 backdrop-blur-xl border-t border-border safe-bottom">
      <div className="flex items-center justify-around h-16 px-2">
        {navItems.map((item) => {
          const active = isActive(item.path);
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={cn(
                "flex flex-col items-center justify-center gap-1 px-3 py-2 rounded-xl transition-all duration-200 min-w-[60px]",
                active 
                  ? "text-primary" 
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <div className={cn(
                "relative flex items-center justify-center w-10 h-7 rounded-full transition-all duration-200",
                active && "bg-primary/15"
              )}>
                <item.icon className={cn(
                  "w-5 h-5 transition-transform duration-200",
                  active && "scale-110"
                )} />
                {active && (
                  <span className="absolute -top-0.5 right-1 w-1.5 h-1.5 bg-primary rounded-full" />
                )}
              </div>
              <span className={cn(
                "text-2xs font-medium transition-all duration-200",
                active && "font-semibold"
              )}>
                {item.label}
              </span>
            </NavLink>
          );
        })}
      </div>
    </nav>
  );
};