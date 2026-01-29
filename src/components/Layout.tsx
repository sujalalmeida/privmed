import { ReactNode } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { LogOut, Shield } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface LayoutProps {
  children: ReactNode;
  title: string;
}

export default function Layout({ children, title }: LayoutProps) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <div className="min-h-screen bg-neutral-50">
      {/* Header */}
      <header className="bg-white border-b border-neutral-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo and Brand */}
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-9 h-9 bg-primary-500 rounded-lg">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div className="flex flex-col">
                <span className="text-lg font-semibold text-neutral-900 leading-tight">
                  MedSafe
                </span>
                <span className="text-xs text-neutral-500 leading-tight">
                  Clinical Data Platform
                </span>
              </div>
            </div>

            {/* Page Title (center) */}
            <div className="hidden md:block">
              <h1 className="text-sm font-medium text-neutral-600">{title}</h1>
            </div>

            {/* User Info and Logout */}
            <div className="flex items-center gap-4">
              <div className="hidden sm:flex flex-col items-end">
                <span className="text-sm font-medium text-neutral-900">
                  {user?.fullName || user?.email?.split('@')[0] || 'User'}
                </span>
                <span className="text-xs text-neutral-500 capitalize">
                  {user?.role || 'Patient'}
                </span>
              </div>
              <button
                onClick={handleLogout}
                className="btn-ghost flex items-center gap-2 !px-3 !py-2"
                title="Sign out"
              >
                <LogOut className="w-4 h-4" />
                <span className="hidden sm:inline text-sm">Sign out</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile Title Bar */}
      <div className="md:hidden bg-white border-b border-neutral-200 px-4 py-3">
        <h1 className="text-base font-medium text-neutral-900">{title}</h1>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {children}
      </main>
    </div>
  );
}
