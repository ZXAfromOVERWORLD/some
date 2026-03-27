import { Outlet, Link, useLocation, useNavigate } from 'react-router';
import { Trash2, MapPin, AlertCircle, User, LogOut, Menu, X, Mail, ShieldAlert, Zap } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useState } from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '../ui/alert-dialog';

export default function UserLayout() {
  const location = useLocation();
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [showLogoutDialog, setShowLogoutDialog] = useState(false);

  const navItems = [
    { path: '/user-collection', label: 'Collection Request', icon: Trash2, color: 'text-emerald-500', bg: 'bg-emerald-50' },
    { path: '/user-complaints', label: 'Complaints', icon: ShieldAlert, color: 'text-rose-500', bg: 'bg-rose-50' },
  ];

  const handleLogout = () => setShowLogoutDialog(true);

  const confirmLogout = () => {
    logout();
    navigate('/login');
    setShowLogoutDialog(false);
  };

  const isActive = (path: string) => location.pathname === path;

  return (
    <div className="min-h-screen bg-slate-50 flex font-sans text-slate-900 selection:bg-emerald-200">
      
      {/* Decorative Background Elements */}
      <div className="fixed top-0 left-0 w-full h-[300px] bg-gradient-to-b from-blue-100/40 to-transparent pointer-events-none" />
      <div className="fixed top-20 right-20 w-[500px] h-[500px] bg-emerald-100/30 rounded-full blur-[100px] pointer-events-none" />

      {/* Desktop Sidebar (Floating Style) */}
      <aside className="hidden lg:flex lg:flex-col lg:w-72 lg:fixed lg:inset-y-4 lg:left-4 z-40">
        <div className="flex-1 bg-white/80 backdrop-blur-3xl border border-white/40 shadow-[0_8px_30px_rgb(0,0,0,0.04)] rounded-[2rem] flex flex-col overflow-hidden">
          
          {/* Logo */}
          <div className="p-8 pb-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-tr from-emerald-500 to-teal-400 rounded-[1.25rem] flex items-center justify-center shadow-lg shadow-emerald-500/25">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-black text-slate-800 tracking-tight">SmartWaste</h1>
                <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wider">Citizen Portal</p>
              </div>
            </div>
          </div>

          {/* User Profile Card */}
          <div className="px-6 mb-6">
            <div className="relative overflow-hidden bg-gradient-to-br from-slate-900 to-slate-800 rounded-3xl p-5 shadow-xl shadow-slate-900/10 border border-slate-700">
              <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/20 rounded-full blur-2xl transform translate-x-1/2 -translate-y-1/2" />
              
              <div className="relative z-10">
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-12 h-12 bg-white/10 rounded-2xl flex items-center justify-center border border-white/10 backdrop-blur-md">
                    <User className="w-6 h-6 text-emerald-400" />
                  </div>
                  <div>
                    <p className="font-bold text-white text-lg leading-tight">{user?.username || 'Resident'}</p>
                    <div className="inline-flex items-center gap-1.5 mt-1 px-2.5 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-500/20">
                      <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                      <span className="text-[10px] font-bold tracking-wide text-emerald-300 uppercase">Verified</span>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-2.5">
                  <div className="flex items-center gap-3 text-slate-300">
                    <div className="w-7 h-7 rounded-xl bg-white/5 flex items-center justify-center">
                      <Mail className="w-3.5 h-3.5" />
                    </div>
                    <span className="text-xs font-medium truncate">{user?.username}@wastesmart.co</span>
                  </div>
                  <div className="flex items-center gap-3 text-slate-300">
                    <div className="w-7 h-7 rounded-xl bg-white/5 flex items-center justify-center">
                      <MapPin className="w-3.5 h-3.5" />
                    </div>
                    <span className="text-xs font-medium">Smart City Zone A</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 space-y-2 overflow-y-auto">
            <p className="px-4 text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2 mt-2">Services</p>
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.path);
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center gap-4 px-4 py-3.5 rounded-2xl transition-all duration-300 group ${
                    active
                      ? 'bg-white shadow-[0_2px_15px_-3px_rgba(0,0,0,0.07),0_10px_20px_-2px_rgba(0,0,0,0.04)] border border-slate-100'
                      : 'hover:bg-white/60 hover:shadow-sm border border-transparent'
                  }`}
                >
                  <div className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors duration-300 ${
                    active ? `bg-gradient-to-br ${item.bg.replace('bg-', 'from-').replace('-50', '-500')} to-${item.bg.replace('bg-', '').replace('-50', '-600')}` : item.bg + ' group-hover:scale-110'
                  }`}>
                    <Icon className={`w-5 h-5 ${active ? 'text-white' : item.color}`} />
                  </div>
                  <span className={`font-semibold ${active ? 'text-slate-800' : 'text-slate-600'}`}>
                    {item.label}
                  </span>
                </Link>
              );
            })}
          </nav>

          {/* Logout Button */}
          <div className="p-4 mt-auto">
            <button
              onClick={handleLogout}
              className="w-full flex items-center justify-between px-5 py-4 bg-slate-100 hover:bg-rose-50 border border-slate-200 hover:border-rose-200 rounded-2xl transition-all duration-300 text-slate-600 hover:text-rose-600 font-semibold group"
            >
              <div className="flex items-center gap-3">
                <LogOut className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
                <span>Logout</span>
              </div>
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 lg:pl-[20rem] flex flex-col min-h-screen relative z-10 w-full">
        
        {/* Mobile Header */}
        <header className="lg:hidden bg-white/80 backdrop-blur-xl border-b border-slate-200 sticky top-0 z-40">
          <div className="px-4 h-16 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-tr from-emerald-500 to-teal-400 rounded-xl flex items-center justify-center shadow-md">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-black text-slate-800 tracking-tight">SmartWaste</h1>
              </div>
            </div>
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2.5 bg-slate-100 hover:bg-slate-200 rounded-xl transition-colors"
            >
              {mobileMenuOpen ? <X className="w-5 h-5 text-slate-700" /> : <Menu className="w-5 h-5 text-slate-700" />}
            </button>
          </div>
        </header>

        {/* Mobile Sidebar Overlay */}
        {mobileMenuOpen && (
          <div className="lg:hidden fixed inset-0 z-50 flex">
            <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm transition-opacity" onClick={() => setMobileMenuOpen(false)} />
            <div className="relative w-4/5 max-w-sm bg-white h-full shadow-2xl flex flex-col">
              <div className="p-6 border-b border-slate-100 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-gradient-to-tr from-emerald-500 to-teal-400 rounded-xl flex items-center justify-center">
                    <Zap className="w-5 h-5 text-white" />
                  </div>
                  <h1 className="text-lg font-black text-slate-800 tracking-tight">SmartWaste</h1>
                </div>
                <button onClick={() => setMobileMenuOpen(false)} className="p-2 bg-slate-100 rounded-lg"><X className="w-5 h-5" /></button>
              </div>
              
              <div className="p-4 border-b border-slate-100">
                <div className="bg-slate-900 rounded-2xl p-4 flex items-center gap-3">
                  <div className="w-10 h-10 bg-white/10 rounded-xl flex items-center justify-center">
                    <User className="w-5 h-5 text-emerald-400" />
                  </div>
                  <div>
                    <p className="font-bold text-white">{user?.username || 'Resident'}</p>
                    <p className="text-xs text-emerald-400">Verified Citizen</p>
                  </div>
                </div>
              </div>

              <div className="p-4 flex-1">
                <p className="px-2 text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2">Services</p>
                <div className="space-y-2">
                  {navItems.map((item) => {
                    const Icon = item.icon;
                    const active = isActive(item.path);
                    return (
                      <Link
                        key={item.path}
                        to={item.path}
                        onClick={() => setMobileMenuOpen(false)}
                        className={`flex items-center gap-4 px-4 py-3 rounded-xl transition-all ${
                          active ? 'bg-slate-50 border border-slate-100 shadow-sm' : ''
                        }`}
                      >
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${active ? `bg-gradient-to-br from-emerald-500 to-teal-600` : item.bg}`}>
                          <Icon className={`w-5 h-5 ${active ? 'text-white' : item.color}`} />
                        </div>
                        <span className={`font-semibold ${active ? 'text-slate-900' : 'text-slate-600'}`}>{item.label}</span>
                      </Link>
                    );
                  })}
                </div>
              </div>

              <div className="p-4 border-t border-slate-100">
                <button onClick={handleLogout} className="w-full flex items-center justify-center gap-2 py-3 bg-rose-50 text-rose-600 rounded-xl font-semibold">
                  <LogOut className="w-5 h-5" /> Logout
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Page Content */}
        <main className="flex-1 w-full mx-auto relative p-4 sm:p-6 lg:p-8">
          <Outlet />
        </main>
      </div>

      {/* Logout Confirmation Dialog */}
      <AlertDialog open={showLogoutDialog} onOpenChange={setShowLogoutDialog}>
        <AlertDialogContent className="rounded-[2rem] border-0 shadow-2xl p-6 sm:p-8 max-w-md">
          <AlertDialogHeader>
            <div className="w-16 h-16 bg-rose-100 text-rose-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <LogOut className="w-8 h-8" />
            </div>
            <AlertDialogTitle className="text-center text-2xl font-bold text-slate-800">Ready to leave?</AlertDialogTitle>
            <AlertDialogDescription className="text-center text-slate-500 text-base mt-2">
              You are about to log out from your SmartWaste citizen account. You will need your credentials to log back in.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter className="mt-8 flex gap-3 sm:space-x-0">
            <AlertDialogCancel className="w-full border-2 border-slate-200 text-slate-600 font-bold hover:bg-slate-50 hover:text-slate-800 rounded-xl py-6 rounded-2xl">Stay logged in</AlertDialogCancel>
            <AlertDialogAction onClick={confirmLogout} className="w-full bg-rose-500 hover:bg-rose-600 text-white font-bold py-6 rounded-xl rounded-2xl shadow-lg shadow-rose-500/30">Yes, log out</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}