import { Outlet, Link, useLocation, useNavigate } from "react-router";
import { Recycle, LayoutDashboard, Map, Settings, Brain, LogOut, User, Menu, X } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { useState } from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "./ui/alert-dialog";
import { AnimatedBackground } from "./AnimatedBackground";

export default function Layout() {
  const location = useLocation();
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [showLogoutDialog, setShowLogoutDialog] = useState(false);

  const getNavItems = () => {
    if (user?.role === 'admin') {
      return [
        { path: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
        { path: "/map", label: "Map", icon: Map },
        { path: "/admin", label: "Admin", icon: Settings },
        { path: "/ai-insights", label: "AI Insights", icon: Brain },
      ];
    }
    return [];
  };

  const navItems = getNavItems();

  const handleLogout = () => {
    setShowLogoutDialog(true);
  };

  const confirmLogout = () => {
    logout();
    navigate('/login');
    setShowLogoutDialog(false);
  };

  const isActive = (path: string) => {
    if (path === "/") {
      return location.pathname === "/";
    }
    return location.pathname.startsWith(path);
  };

  return (
    <div className="min-h-screen relative font-sans text-foreground bg-background">
      <AnimatedBackground />
      {/* Top Navigation Bar */}
      <nav className="glass-header sticky top-0 z-50 bg-background/80 backdrop-blur-xl border-b border-border shadow-sm">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="bg-primary p-2.5 rounded-2xl shadow-lg shadow-primary/20">
                <Recycle className="h-6 w-6 text-primary-foreground" />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-xl font-bold text-foreground tracking-tight">Smart Waste AI</h1>
                <p className="text-xs text-muted-foreground font-medium">Municipal Management System</p>
              </div>
            </div>

            {/* Desktop Navigation Menu */}
            <div className="hidden md:flex items-center gap-3">
              <div className="flex items-center gap-1 bg-muted/50 rounded-xl p-1 border border-border/50">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const active = isActive(item.path);
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-300 ${
                        active
                          ? "bg-primary text-primary-foreground shadow-md shadow-primary/20"
                          : "text-muted-foreground hover:bg-background hover:text-foreground hover:shadow-sm"
                      }`}
                    >
                      <Icon className="h-4 w-4" />
                      <span className="font-medium text-sm">{item.label}</span>
                    </Link>
                  );
                })}
              </div>
              
              {/* User Info & Logout */}
              {user && (
                <div className="flex items-center gap-3 ml-2 pl-4 border-l border-border">
                  <div className="flex items-center gap-3 px-3 py-2 bg-muted/30 rounded-xl border border-border/50">
                    <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center border border-primary/20">
                      <User className="h-4 w-4 text-primary" />
                    </div>
                    <div className="hidden lg:block text-left">
                      <p className="text-sm font-semibold text-foreground leading-none mb-1">{user.username}</p>
                      <p className="text-xs text-muted-foreground capitalize leading-none">{user.role}</p>
                    </div>
                  </div>
                  <button
                    onClick={handleLogout}
                    className="flex flex-col items-center justify-center w-10 h-10 bg-muted/30 hover:bg-destructive/10 hover:text-destructive rounded-xl transition-all text-muted-foreground border border-border/50 hover:border-destructive/20"
                    title="Logout"
                  >
                    <LogOut className="h-4 w-4" />
                  </button>
                </div>
              )}
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-lg bg-muted/50 hover:bg-muted text-foreground transition-colors border border-border/50"
            >
              {mobileMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile Sidebar Drawer */}
      {mobileMenuOpen && (
        <>
          {/* Backdrop/Overlay */}
          <div
            className="md:hidden fixed inset-0 bg-background/80 backdrop-blur-sm z-40 transition-opacity"
            onClick={() => setMobileMenuOpen(false)}
          />

          {/* Sidebar Drawer */}
          <div className="md:hidden fixed top-0 left-0 bottom-0 w-80 bg-background z-50 shadow-2xl border-r border-border transform transition-transform duration-300 ease-in-out overflow-y-auto">
            {/* Sidebar Header */}
            <div className="p-6 border-b border-border">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="bg-primary p-2.5 rounded-2xl shadow-lg shadow-primary/20">
                    <Recycle className="h-6 w-6 text-primary-foreground" />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold text-foreground">Smart Waste AI</h2>
                    <p className="text-xs text-muted-foreground">Municipal System</p>
                  </div>
                </div>
                <button
                  onClick={() => setMobileMenuOpen(false)}
                  className="p-2 rounded-lg hover:bg-muted transition-colors border border-transparent hover:border-border"
                >
                  <X className="h-5 w-5 text-muted-foreground" />
                </button>
              </div>

              {/* User Info in Sidebar */}
              {user && (
                <div className="flex items-center gap-3 p-4 bg-primary/10 border border-primary/20 rounded-xl">
                  <div className="w-12 h-12 bg-background rounded-full flex items-center justify-center border border-border shadow-sm">
                    <User className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <p className="font-semibold text-foreground">{user.username}</p>
                    <p className="text-xs text-muted-foreground capitalize mt-0.5">{user.role}</p>
                  </div>
                </div>
              )}
            </div>

            {/* Navigation Links */}
            <div className="p-6">
              <nav className="space-y-2">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const active = isActive(item.path);
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      onClick={() => setMobileMenuOpen(false)}
                      className={`flex items-center gap-4 px-4 py-3 rounded-xl transition-all duration-300 ${
                        active
                          ? "bg-primary text-primary-foreground shadow-md shadow-primary/20"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground border border-transparent hover:border-border/50"
                      }`}
                    >
                      <Icon className="h-5 w-5" />
                      <span className="font-medium">{item.label}</span>
                    </Link>
                  );
                })}
              </nav>
            </div>

            {/* Logout Button at Bottom */}
            {user && (
              <div className="absolute bottom-0 left-0 right-0 p-6 border-t border-border bg-background/50 backdrop-blur-md">
                <button
                  onClick={() => {
                    setMobileMenuOpen(false);
                    handleLogout();
                  }}
                  className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-destructive/10 text-destructive border border-destructive/20 hover:bg-destructive hover:text-destructive-foreground rounded-xl transition-all duration-300 font-medium shadow-sm"
                >
                  <LogOut className="h-5 w-5" />
                  Logout
                </button>
              </div>
            )}
          </div>
        </>
      )}

      {/* Main Content */}
      <main className="max-w-[1600px] mx-auto relative z-10 w-full animate-fade-in">
        <Outlet />
      </main>

      {/* Logout Confirmation Dialog */}
      <AlertDialog open={showLogoutDialog} onOpenChange={setShowLogoutDialog}>
        <AlertDialogContent className="border-border shadow-2xl rounded-2xl">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-xl">Logout Confirmation</AlertDialogTitle>
            <AlertDialogDescription className="text-muted-foreground">
              Are you sure you want to logout? This will end your current session.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="border-border hover:bg-muted">Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmLogout} className="bg-destructive text-destructive-foreground hover:bg-destructive/90 shadow-md">Logout</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}