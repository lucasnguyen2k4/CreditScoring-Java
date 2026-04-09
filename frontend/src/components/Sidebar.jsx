import { NavLink, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { getNavItems } from '../utils/permissions';
import {
  LayoutDashboard, Upload, Settings, Brain, BarChart3, Target, Shield, ShieldCheck, LogOut,
} from 'lucide-react';

const iconMap = {
  LayoutDashboard: <LayoutDashboard />,
  Upload: <Upload />,
  Settings: <Settings />,
  Brain: <Brain />,
  BarChart3: <BarChart3 />,
  Target: <Target />,
  Shield: <Shield />,
  ShieldCheck: <ShieldCheck />,
};

export default function Sidebar() {
  const { user, logout } = useAuth();
  const location = useLocation();

  if (!user) return null;

  const navItems = getNavItems(user.role);
  const initials = (user.displayName || user.username || '?').charAt(0).toUpperCase();

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">CreditScoring</div>
        <div className="sidebar-logo-sub">Machine Learning Platform</div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            {iconMap[item.icon]}
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="user-info">
          <div className="user-avatar">{initials}</div>
          <div>
            <div className="user-name">{user.displayName || user.username}</div>
            <div className="user-role">{user.role?.replace('_', ' ')}</div>
          </div>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={logout} style={{ width: '100%' }}>
          <LogOut size={14} /> Logout
        </button>
      </div>
    </aside>
  );
}
