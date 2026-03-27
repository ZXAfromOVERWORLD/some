import { useState } from 'react';
import { useNavigate, Link } from 'react-router';
import { UserPlus, Recycle, AlertCircle, Eye, EyeOff, Mail, Lock, CheckCircle2 } from 'lucide-react';

function getPasswordStrength(pwd: string): { label: string; color: string; width: string; score: number } {
  let score = 0;
  if (pwd.length >= 6) score++;
  if (pwd.length >= 10) score++;
  if (/[A-Z]/.test(pwd)) score++;
  if (/[^a-zA-Z0-9]/.test(pwd)) score++;

  if (score === 0) return { label: '', color: '#334155', width: '0%', score: 0 };
  if (score === 1) return { label: 'Weak', color: '#ef4444', width: '25%', score: 1 };
  if (score === 2) return { label: 'Fair', color: '#f59e0b', width: '50%', score: 2 };
  if (score === 3) return { label: 'Good', color: '#3b82f6', width: '75%', score: 3 };
  return { label: 'Strong', color: '#22c55e', width: '100%', score: 4 };
}

export default function Signup() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const strength = getPasswordStrength(password);
  const passwordsMatch = confirmPassword.length > 0 && password === confirmPassword;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setIsLoading(true);
    await new Promise(r => setTimeout(r, 600));

    const users = JSON.parse(localStorage.getItem('waste_mgmt_registered_users') || '[]');
    if (users.find((u: any) => u.email === email)) {
      setError('This email is already registered. Please sign in instead.');
      setIsLoading(false);
      return;
    }

    localStorage.setItem('waste_mgmt_temp_signup', JSON.stringify({ email, password }));
    navigate('/select-role');
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden" style={{ background: 'linear-gradient(135deg, #1a2744 0%, #243354 40%, #1a3040 100%)' }}>
      {/* Animated orbs */}
      <div style={{ position: 'absolute', top: '5%', right: '8%', width: 350, height: 350, background: 'radial-gradient(circle, rgba(16,185,129,0.18) 0%, transparent 70%)', borderRadius: '50%', filter: 'blur(40px)', animation: 'pulse 4s ease-in-out infinite' }} />
      <div style={{ position: 'absolute', bottom: '5%', left: '8%', width: 300, height: 300, background: 'radial-gradient(circle, rgba(139,92,246,0.18) 0%, transparent 70%)', borderRadius: '50%', filter: 'blur(40px)', animation: 'pulse 5s ease-in-out infinite 1.5s' }} />

      {/* Grid overlay */}
      <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)', backgroundSize: '50px 50px', pointerEvents: 'none' }} />

      <div className="w-full max-w-md relative z-10">
        {/* Logo */}
        <div className="text-center mb-8" style={{ animation: 'fadeInDown 0.6s ease' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 80, height: 80, background: 'linear-gradient(135deg, #10b981, #059669)', borderRadius: 24, marginBottom: 16, boxShadow: '0 0 40px rgba(16,185,129,0.4), 0 20px 40px rgba(0,0,0,0.3)' }}>
            <Recycle style={{ width: 40, height: 40, color: 'white' }} />
          </div>
          <h1 style={{ fontSize: 32, fontWeight: 800, color: 'white', marginBottom: 6, letterSpacing: '-0.5px' }}>
            Join <span style={{ color: '#10b981' }}>SmartWaste</span>
          </h1>
          <p style={{ color: '#94a3b8', fontSize: 15 }}>Create your account in seconds</p>
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
                  onFocus={e => (e.target.style.borderColor = '#10b981')}
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
                  placeholder="Min. 6 characters"
                  required
                  minLength={6}
                  style={{ width: '100%', paddingLeft: 44, paddingRight: 48, paddingTop: 13, paddingBottom: 13, background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 12, color: 'white', fontSize: 15, outline: 'none', boxSizing: 'border-box', transition: 'border-color 0.2s' }}
                  onFocus={e => (e.target.style.borderColor = '#10b981')}
                  onBlur={e => (e.target.style.borderColor = 'rgba(255,255,255,0.12)')}
                />
                <button type="button" onClick={() => setShowPassword(!showPassword)} style={{ position: 'absolute', right: 14, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: '#64748b', padding: 0 }}>
                  {showPassword ? <EyeOff style={{ width: 18, height: 18 }} /> : <Eye style={{ width: 18, height: 18 }} />}
                </button>
              </div>
              {/* Strength bar */}
              {password.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ height: 4, background: 'rgba(255,255,255,0.1)', borderRadius: 4, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: strength.width, background: strength.color, borderRadius: 4, transition: 'all 0.3s ease' }} />
                  </div>
                  <p style={{ fontSize: 11, color: strength.color, marginTop: 4, fontWeight: 600 }}>{strength.label}</p>
                </div>
              )}
            </div>

            {/* Confirm Password */}
            <div>
              <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#cbd5e1', marginBottom: 8 }}>Confirm Password</label>
              <div style={{ position: 'relative' }}>
                <Lock style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', width: 18, height: 18, color: '#64748b' }} />
                <input
                  type={showConfirm ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Re-enter password"
                  required
                  minLength={6}
                  style={{ width: '100%', paddingLeft: 44, paddingRight: 48, paddingTop: 13, paddingBottom: 13, background: 'rgba(255,255,255,0.06)', border: `1px solid ${confirmPassword.length > 0 ? (passwordsMatch ? 'rgba(34,197,94,0.5)' : 'rgba(239,68,68,0.5)') : 'rgba(255,255,255,0.12)'}`, borderRadius: 12, color: 'white', fontSize: 15, outline: 'none', boxSizing: 'border-box', transition: 'border-color 0.2s' }}
                  onFocus={e => { if (confirmPassword.length === 0) e.target.style.borderColor = '#10b981'; }}
                  onBlur={e => { if (confirmPassword.length === 0) e.target.style.borderColor = 'rgba(255,255,255,0.12)'; }}
                />
                <div style={{ position: 'absolute', right: 14, top: '50%', transform: 'translateY(-50%)', display: 'flex', alignItems: 'center', gap: 6 }}>
                  {confirmPassword.length > 0 && (
                    <CheckCircle2 style={{ width: 16, height: 16, color: passwordsMatch ? '#22c55e' : '#ef4444' }} />
                  )}
                  <button type="button" onClick={() => setShowConfirm(!showConfirm)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#64748b', padding: 0 }}>
                    {showConfirm ? <EyeOff style={{ width: 18, height: 18 }} /> : <Eye style={{ width: 18, height: 18 }} />}
                  </button>
                </div>
              </div>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={isLoading}
              style={{ width: '100%', padding: '14px 0', background: isLoading ? 'rgba(16,185,129,0.5)' : 'linear-gradient(135deg, #10b981, #059669)', border: 'none', borderRadius: 12, color: 'white', fontSize: 16, fontWeight: 700, cursor: isLoading ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, boxShadow: '0 4px 20px rgba(16,185,129,0.35)', transition: 'all 0.2s', letterSpacing: '0.3px' }}
              onMouseOver={e => { if (!isLoading) (e.currentTarget.style.transform = 'translateY(-1px)'); }}
              onMouseOut={e => { (e.currentTarget.style.transform = 'translateY(0)'); }}
            >
              {isLoading ? (
                <div style={{ width: 20, height: 20, border: '2px solid rgba(255,255,255,0.4)', borderTopColor: 'white', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />
              ) : (
                <><UserPlus style={{ width: 18, height: 18 }} /> Continue to Role Selection</>
              )}
            </button>
          </form>

          {/* Login link */}
          <p style={{ textAlign: 'center', marginTop: 24, color: '#64748b', fontSize: 14 }}>
            Already have an account?{' '}
            <Link to="/login" style={{ color: '#10b981', fontWeight: 700, textDecoration: 'none' }}>
              Sign In
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