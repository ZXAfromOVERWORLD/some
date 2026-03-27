import { useState } from 'react';
import { useNavigate, Link } from 'react-router';
import { useAuth } from '../../context/AuthContext';
import { LogIn, Recycle, AlertCircle, Eye, EyeOff, Mail, Lock, ShieldCheck } from 'lucide-react';
import { login as apiLogin } from '../../services/authService';

// Admin credentials (still available for quick fill, but validation happens on backend)
const ADMIN_EMAIL = 'admin@wastesmart.com';
const ADMIN_PASSWORD = 'admin123';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const response = await apiLogin({ email, password });
      
      // Update global context with token & full user profile
      login(response.token, response.user);
      
      // Route based on role
      if (response.user.role === 'admin') {
        navigate('/dashboard');
      } else if (response.user.role === 'driver') {
        navigate('/driver-map');
      } else {
        navigate('/user-collection');
      }
    } catch (err: any) {
      setError(err.response?.data?.message || 'Invalid email or password. Please check your credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  const fillAdmin = () => {
    setEmail(ADMIN_EMAIL);
    setPassword(ADMIN_PASSWORD);
    setError('');
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden" style={{ background: 'linear-gradient(135deg, #1a2744 0%, #243354 40%, #1a3040 100%)' }}>
      {/* Animated gradient orbs */}
      <div style={{ position: 'absolute', top: '8%', left: '5%', width: 380, height: 380, background: 'radial-gradient(circle, rgba(34,197,94,0.18) 0%, transparent 70%)', borderRadius: '50%', filter: 'blur(40px)', animation: 'pulse 4s ease-in-out infinite' }} />
      <div style={{ position: 'absolute', bottom: '8%', right: '5%', width: 320, height: 320, background: 'radial-gradient(circle, rgba(59,130,246,0.18) 0%, transparent 70%)', borderRadius: '50%', filter: 'blur(40px)', animation: 'pulse 5s ease-in-out infinite 1s' }} />
      <div style={{ position: 'absolute', top: '50%', left: '50%', width: 500, height: 500, background: 'radial-gradient(circle, rgba(16,185,129,0.07) 0%, transparent 70%)', borderRadius: '50%', filter: 'blur(60px)', transform: 'translate(-50%, -50%)' }} />

      {/* Grid overlay */}
      <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)', backgroundSize: '50px 50px', pointerEvents: 'none' }} />

      <div className="w-full max-w-md relative z-10">
        {/* Logo & Title */}
        <div className="text-center mb-8" style={{ animation: 'fadeInDown 0.6s ease' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 80, height: 80, background: 'linear-gradient(135deg, #22c55e, #16a34a)', borderRadius: 24, marginBottom: 16, boxShadow: '0 0 40px rgba(34,197,94,0.4), 0 20px 40px rgba(0,0,0,0.3)' }}>
            <Recycle style={{ width: 40, height: 40, color: 'white' }} />
          </div>
          <h1 style={{ fontSize: 32, fontWeight: 800, color: 'white', marginBottom: 6, letterSpacing: '-0.5px' }}>
            Smart<span style={{ color: '#22c55e' }}>Waste</span>
          </h1>
          <p style={{ color: '#94a3b8', fontSize: 15 }}>Sign in to your account to continue</p>
        </div>

        {/* Card */}
        <div style={{ background: 'rgba(255,255,255,0.05)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 24, padding: 36, boxShadow: '0 25px 60px rgba(0,0,0,0.5)', animation: 'fadeInUp 0.7s ease' }}>

          {/* Error */}
          {error && (
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, padding: '12px 16px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 12, marginBottom: 20 }}>
              <AlertCircle style={{ width: 18, height: 18, color: '#f87171', flexShrink: 0, marginTop: 1 }} />
              <p style={{ fontSize: 13, color: '#fca5a5', lineHeight: 1.5 }}>{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* Email */}
            <div>
              <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#cbd5e1', marginBottom: 8 }}>Email Address</label>
              <div style={{ position: 'relative' }}>
                <Mail style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', width: 18, height: 18, color: '#64748b' }} />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  required
                  style={{ width: '100%', paddingLeft: 44, paddingRight: 16, paddingTop: 13, paddingBottom: 13, background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 12, color: 'white', fontSize: 15, outline: 'none', boxSizing: 'border-box', transition: 'border-color 0.2s' }}
                  onFocus={e => (e.target.style.borderColor = '#22c55e')}
                  onBlur={e => (e.target.style.borderColor = 'rgba(255,255,255,0.12)')}
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#cbd5e1', marginBottom: 8 }}>Password</label>
              <div style={{ position: 'relative' }}>
                <Lock style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', width: 18, height: 18, color: '#64748b' }} />
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  required
                  style={{ width: '100%', paddingLeft: 44, paddingRight: 48, paddingTop: 13, paddingBottom: 13, background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 12, color: 'white', fontSize: 15, outline: 'none', boxSizing: 'border-box', transition: 'border-color 0.2s' }}
                  onFocus={e => (e.target.style.borderColor = '#22c55e')}
                  onBlur={e => (e.target.style.borderColor = 'rgba(255,255,255,0.12)')}
                />
                <button type="button" onClick={() => setShowPassword(!showPassword)} style={{ position: 'absolute', right: 14, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: '#64748b', padding: 0 }}>
                  {showPassword ? <EyeOff style={{ width: 18, height: 18 }} /> : <Eye style={{ width: 18, height: 18 }} />}
                </button>
              </div>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={isLoading}
              style={{ width: '100%', padding: '14px 0', background: isLoading ? 'rgba(34,197,94,0.5)' : 'linear-gradient(135deg, #22c55e, #16a34a)', border: 'none', borderRadius: 12, color: 'white', fontSize: 16, fontWeight: 700, cursor: isLoading ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, boxShadow: '0 4px 20px rgba(34,197,94,0.35)', transition: 'all 0.2s', letterSpacing: '0.3px' }}
              onMouseOver={e => { if (!isLoading) (e.currentTarget.style.transform = 'translateY(-1px)'); }}
              onMouseOut={e => { (e.currentTarget.style.transform = 'translateY(0)'); }}
            >
              {isLoading ? (
                <div style={{ width: 20, height: 20, border: '2px solid rgba(255,255,255,0.4)', borderTopColor: 'white', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />
              ) : (
                <><LogIn style={{ width: 18, height: 18 }} /> Sign In</>
              )}
            </button>
          </form>

          {/* Divider */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, margin: '24px 0' }}>
            <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.1)' }} />
            <span style={{ color: '#475569', fontSize: 13 }}>or</span>
            <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.1)' }} />
          </div>

          {/* Admin Quick Fill */}
          <button
            onClick={fillAdmin}
            style={{ width: '100%', padding: '12px 16px', background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.3)', borderRadius: 12, color: '#a5b4fc', fontSize: 13, fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, transition: 'all 0.2s' }}
            onMouseOver={e => { (e.currentTarget.style.background = 'rgba(99,102,241,0.2)'); }}
            onMouseOut={e => { (e.currentTarget.style.background = 'rgba(99,102,241,0.1)'); }}
          >
            <ShieldCheck style={{ width: 16, height: 16 }} />
            Use Demo Admin Credentials
          </button>

          {/* Signup link */}
          <p style={{ textAlign: 'center', marginTop: 24, color: '#64748b', fontSize: 14 }}>
            Don't have an account?{' '}
            <Link to="/signup" style={{ color: '#22c55e', fontWeight: 700, textDecoration: 'none' }}>
              Sign Up Free
            </Link>
          </p>
        </div>

        <p style={{ textAlign: 'center', color: '#94a3b8', fontSize: 12, marginTop: 20 }}>
          🌱 Smart City Initiative 2026 • Powered by AI
        </p>
      </div>

      <style>{`
        @keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity: 0.6; transform: scale(1); } 50% { opacity: 1; transform: scale(1.05); } }
        input::placeholder { color: #9ca3af !important; }
      `}</style>
    </div>
  );
}