import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Shield, ChevronRight } from 'lucide-react';

export default function Login() {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [contactPhone, setContactPhone] = useState('');
  const [role, setRole] = useState<'patient' | 'lab' | 'central_admin'>('patient');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { login, signup, user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (user) {
      if (user.role === 'patient') navigate('/patient');
      else if (user.role === 'lab') navigate('/lab');
      else if (user.role === 'central_admin') navigate('/admin');
    }
  }, [user, navigate]);

  const validateForm = () => {
    if (isSignUp) {
      if (password.length < 8) {
        setError('Password must be at least 8 characters long.');
        return false;
      }
      if (password !== confirmPassword) {
        setError('Passwords do not match.');
        return false;
      }
      if (!fullName.trim()) {
        setError('Full name is required.');
        return false;
      }
    }
    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    if (!validateForm()) {
      setIsLoading(false);
      return;
    }

    let success = false;

    if (isSignUp) {
      success = await signup(email, password, role, fullName, contactPhone);
    } else {
      success = await login(email, password, role);
    }

    if (!success) {
      setError(isSignUp ? 'Failed to create account. Please try again.' : 'Invalid credentials.');
    }

    setIsLoading(false);
  };

  const roleLabels = {
    patient: { label: 'Patient', desc: 'View reports and health data' },
    lab: { label: 'Lab Technician', desc: 'Process clinical data' },
    central_admin: { label: 'Administrator', desc: 'Manage system and models' },
  };

  return (
    <div className="min-h-screen bg-neutral-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo and Brand */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-primary-500 rounded-xl mb-4">
            <Shield className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-semibold text-neutral-900">MedSafe</h1>
          <p className="text-sm text-neutral-500 mt-1">Clinical Data Platform</p>
        </div>

        {/* Card */}
        <div className="card p-8">
          {/* Toggle */}
          <div className="flex rounded-lg bg-neutral-100 p-1 mb-6">
            <button
              type="button"
              onClick={() => setIsSignUp(false)}
              className={`flex-1 py-2 text-sm font-medium rounded-md transition-all duration-150 ${
                !isSignUp
                  ? 'bg-white text-neutral-900 shadow-subtle'
                  : 'text-neutral-500 hover:text-neutral-700'
              }`}
            >
              Sign in
            </button>
            <button
              type="button"
              onClick={() => setIsSignUp(true)}
              className={`flex-1 py-2 text-sm font-medium rounded-md transition-all duration-150 ${
                isSignUp
                  ? 'bg-white text-neutral-900 shadow-subtle'
                  : 'text-neutral-500 hover:text-neutral-700'
              }`}
            >
              Create account
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Role Selection */}
            <div>
              <label className="block text-sm font-medium text-neutral-700 mb-2">
                I am a...
              </label>
              <div className="space-y-2">
                {(['patient', 'lab', 'central_admin'] as const).map((r) => (
                  <button
                    key={r}
                    type="button"
                    onClick={() => setRole(r)}
                    className={`w-full flex items-center justify-between p-3 rounded-lg border transition-all duration-150 ${
                      role === r
                        ? 'border-primary-500 bg-primary-50 ring-1 ring-primary-500'
                        : 'border-neutral-200 hover:border-neutral-300 hover:bg-neutral-50'
                    }`}
                  >
                    <div className="text-left">
                      <div className={`text-sm font-medium ${role === r ? 'text-primary-700' : 'text-neutral-900'}`}>
                        {roleLabels[r].label}
                      </div>
                      <div className="text-xs text-neutral-500">{roleLabels[r].desc}</div>
                    </div>
                    <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                      role === r ? 'border-primary-500 bg-primary-500' : 'border-neutral-300'
                    }`}>
                      {role === r && <div className="w-1.5 h-1.5 bg-white rounded-full" />}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {isSignUp && (
              <div>
                <label htmlFor="fullName" className="block text-sm font-medium text-neutral-700 mb-1.5">
                  Full Name
                </label>
                <input
                  id="fullName"
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Enter your full name"
                  className="form-input"
                  required
                />
              </div>
            )}

            {isSignUp && (
              <div>
                <label htmlFor="contactPhone" className="block text-sm font-medium text-neutral-700 mb-1.5">
                  Contact Phone <span className="text-neutral-400 font-normal">(optional)</span>
                </label>
                <input
                  id="contactPhone"
                  type="tel"
                  value={contactPhone}
                  onChange={(e) => setContactPhone(e.target.value)}
                  placeholder="Enter your phone number"
                  className="form-input"
                />
              </div>
            )}

            <div>
              <label htmlFor="email" className="block text-sm font-medium text-neutral-700 mb-1.5">
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="form-input"
                required
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-neutral-700 mb-1.5">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="form-input"
                required
              />
              {isSignUp && (
                <p className="text-xs text-neutral-500 mt-1.5">Minimum 8 characters</p>
              )}
            </div>

            {isSignUp && (
              <div>
                <label htmlFor="confirmPassword" className="block text-sm font-medium text-neutral-700 mb-1.5">
                  Confirm Password
                </label>
                <input
                  id="confirmPassword"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="••••••••"
                  className="form-input"
                  required
                />
              </div>
            )}

            {error && (
              <div className="alert-error">
                <svg className="w-4 h-4 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className="btn-primary w-full justify-center"
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  {isSignUp ? 'Creating account...' : 'Signing in...'}
                </>
              ) : (
                <>
                  {isSignUp ? 'Create account' : 'Sign in'}
                  <ChevronRight className="w-4 h-4 ml-1" />
                </>
              )}
            </button>
          </form>

          {!isSignUp && (
            <div className="mt-6 pt-6 border-t border-neutral-200">
              <p className="text-xs font-medium text-neutral-500 uppercase tracking-wide mb-3">Demo Credentials</p>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="p-2 bg-neutral-50 rounded-md">
                  <span className="block text-neutral-400">Patient</span>
                  <span className="text-neutral-700">patient@demo.com</span>
                </div>
                <div className="p-2 bg-neutral-50 rounded-md">
                  <span className="block text-neutral-400">Lab</span>
                  <span className="text-neutral-700">lab@demo.com</span>
                </div>
                <div className="p-2 bg-neutral-50 rounded-md">
                  <span className="block text-neutral-400">Admin</span>
                  <span className="text-neutral-700">admin@demo.com</span>
                </div>
              </div>
              <p className="text-xs text-neutral-400 mt-2 text-center">Any password works for demo</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <p className="text-xs text-neutral-400 text-center mt-6">
          Secure clinical data management platform
        </p>
      </div>
    </div>
  );
}
