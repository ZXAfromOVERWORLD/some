import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router';
import { useAuth, UserRole } from '../../context/AuthContext';
import { UserCircle2, Truck, CheckCircle2, ArrowRight, Recycle, MapPin, Bell, Route, BarChart3, Wrench } from 'lucide-react';
import { signup as apiSignup } from '../../services/authService';

const roles: {
  value: UserRole;
  label: string;
  subtitle: string;
  description: string;
  icon: React.ElementType;
  features: { icon: React.ElementType; text: string }[];
  gradient: string;
  glow: string;
  accentColor: string;
}[] = [
  {
    value: 'user',
    label: 'Resident / User',
    subtitle: 'Community Member',
    description: 'Report waste issues and track garbage collection in your area.',
    icon: UserCircle2,
    features: [
      { icon: Bell, text: 'Submit waste complaints' },
      { icon: MapPin, text: 'Track collection status' },
      { icon: BarChart3, text: 'View area statistics' },
    ],
    gradient: 'linear-gradient(135deg, #3b82f6, #2563eb)',
    glow: 'rgba(59,130,246,0.35)',
    accentColor: '#3b82f6',
  },
  {
    value: 'driver',
    label: 'Truck Driver',
    subtitle: 'Field Operative',
    description: 'Access optimized routes and manage waste collection assignments.',
    icon: Truck,
    features: [
      { icon: Route, text: 'Optimized collection routes' },
      { icon: MapPin, text: 'Real-time map navigation' },
      { icon: Wrench, text: 'Assignment management' },
    ],
    gradient: 'linear-gradient(135deg, #f59e0b, #d97706)',
    glow: 'rgba(245,158,11,0.35)',
    accentColor: '#f59e0b',
  },
];

export default function RoleSelection() {
  const [selectedRole, setSelectedRole] = useState<UserRole>('user');
  const [isLoading, setIsLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const tempData = localStorage.getItem('waste_mgmt_temp_signup');
    if (!tempData) {
      navigate('/signup');
    }
  }, [navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const tempData = localStorage.getItem('waste_mgmt_temp_signup');
    if (!tempData) {
      navigate('/signup');
      return;
    }

    setIsLoading(true);
    await new Promise(r => setTimeout(r, 700));

    const { email, password } = JSON.parse(tempData);
    
    try {
      const roleToSubmit = selectedRole;
      
      const response = await apiSignup({
        name: email.split('@')[0],
        email,
        password,
        role: roleToSubmit as 'user' | 'driver'
      });
      
      localStorage.removeItem('waste_mgmt_temp_signup');
      login(response.token, response.user);

      if (response.user.role === 'driver') {
        navigate('/driver-map');
      } else {
        navigate('/user-collection');
      }
    } catch (err: any) {
      alert(err.response?.data?.message || 'Registration failed');
      setIsLoading(false);
    }
  };

  const selected = roles.find(r => r.value === selectedRole)!;

  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden" style={{ background: 'linear-gradient(135deg, #1a2744 0%, #243354 40%, #1a3040 100%)' }}>
      {/* Animated orbs */}
      <div style={{ position: 'absolute', top: '5%', left: '10%', width: 400, height: 400, background: `radial-gradient(circle, ${selected.glow.replace('0.35', '0.15')} 0%, transparent 70%)`, borderRadius: '50%', filter: 'blur(50px)', transition: 'background 0.5s ease', animation: 'pulse 4s ease-in-out infinite' }} />
      <div style={{ position: 'absolute', bottom: '5%', right: '10%', width: 300, height: 300, background: 'radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%)', borderRadius: '50%', filter: 'blur(40px)', animation: 'pulse 5s ease-in-out infinite 2s' }} />

      {/* Grid overlay */}
      <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)', backgroundSize: '50px 50px', pointerEvents: 'none' }} />

      <div className="w-full relative z-10" style={{ maxWidth: 540 }}>
        {/* Header */}
        <div className="text-center mb-8" style={{ animation: 'fadeInDown 0.6s ease' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 80, height: 80, background: 'linear-gradient(135deg, #22c55e, #16a34a)', borderRadius: 24, marginBottom: 16, boxShadow: '0 0 40px rgba(34,197,94,0.4), 0 20px 40px rgba(0,0,0,0.3)' }}>
            <Recycle style={{ width: 40, height: 40, color: 'white' }} />
          </div>
          <h1 style={{ fontSize: 32, fontWeight: 800, color: 'white', marginBottom: 6, letterSpacing: '-0.5px' }}>
            Choose Your <span style={{ color: '#22c55e' }}>Role</span>
          </h1>
          <p style={{ color: '#94a3b8', fontSize: 15 }}>Select how you'll use SmartWaste. You can't change this later.</p>
        </div>

        {/* Card */}
        <div style={{ background: 'rgba(255,255,255,0.05)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 24, padding: 36, boxShadow: '0 25px 60px rgba(0,0,0,0.5)', animation: 'fadeInUp 0.7s ease' }}>
          <form onSubmit={handleSubmit}>
            {/* Role cards */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 28 }}>
              {roles.map((role) => {
                const Icon = role.icon;
                const isSelected = selectedRole === role.value;
                return (
                  <button
                    key={role.value}
                    type="button"
                    onClick={() => setSelectedRole(role.value)}
                    style={{
                      padding: 20,
                      borderRadius: 16,
                      border: isSelected ? `2px solid ${role.accentColor}` : '2px solid rgba(255,255,255,0.08)',
                      background: isSelected ? `linear-gradient(135deg, ${role.accentColor}18, ${role.accentColor}08)` : 'rgba(255,255,255,0.03)',
                      cursor: 'pointer',
                      textAlign: 'left',
                      transition: 'all 0.25s ease',
                      boxShadow: isSelected ? `0 0 25px ${role.glow}` : 'none',
                      transform: isSelected ? 'translateY(-2px)' : 'translateY(0)',
                      position: 'relative',
                      overflow: 'hidden',
                    }}
                  >
                    {/* Selected check */}
                    {isSelected && (
                      <div style={{ position: 'absolute', top: 10, right: 10 }}>
                        <CheckCircle2 style={{ width: 18, height: 18, color: role.accentColor }} />
                      </div>
                    )}

                    {/* Icon */}
                    <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 52, height: 52, background: role.gradient, borderRadius: 14, marginBottom: 14, boxShadow: `0 8px 20px ${role.glow}` }}>
                      <Icon style={{ width: 26, height: 26, color: 'white' }} />
                    </div>

                    <div style={{ fontSize: 16, fontWeight: 700, color: 'white', marginBottom: 4 }}>{role.label}</div>
                    <div style={{ fontSize: 11, fontWeight: 600, color: role.accentColor, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 10 }}>{role.subtitle}</div>
                    <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5 }}>{role.description}</div>
                  </button>
                );
              })}
            </div>

            {/* Features of selected role */}
            <div style={{ background: 'rgba(255,255,255,0.04)', borderRadius: 14, padding: '16px 20px', marginBottom: 24, border: '1px solid rgba(255,255,255,0.07)', transition: 'all 0.3s ease' }}>
              <p style={{ fontSize: 12, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: 12 }}>
                What you'll get as {selected.label}
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {selected.features.map((f, i) => {
                  const FIcon = f.icon;
                  return (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 30, height: 30, background: `${selected.accentColor}18`, borderRadius: 8 }}>
                        <FIcon style={{ width: 15, height: 15, color: selected.accentColor }} />
                      </div>
                      <span style={{ fontSize: 13, color: '#cbd5e1' }}>{f.text}</span>
                    </div>
                  );
                })}
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
                <>Complete Registration <ArrowRight style={{ width: 18, height: 18 }} /></>
              )}
            </button>
          </form>
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
      `}</style>
    </div>
  );
}