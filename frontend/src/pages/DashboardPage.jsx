import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { dataApi } from '../api/client';
import { Database, Brain, BarChart3, Target } from 'lucide-react';

export default function DashboardPage() {
  const { user } = useAuth();
  const [session, setSession] = useState(null);

  useEffect(() => {
    dataApi.getSessionInfo().then(r => setSession(r.data)).catch(() => {});
  }, []);

  const stats = [
    {
      icon: <Database />, label: 'Data Loaded',
      value: session?.has_data ? `${session.data_shape?.[0]} rows` : 'No data',
      color: session?.has_data ? 'var(--success)' : 'var(--text-muted)',
    },
    {
      icon: <BarChart3 />, label: 'Features',
      value: session?.n_features || 0,
      color: session?.n_features > 0 ? 'var(--info)' : 'var(--text-muted)',
    },
    {
      icon: <Brain />, label: 'Models Trained',
      value: session?.n_trained_models || 0,
      color: session?.n_trained_models > 0 ? 'var(--accent)' : 'var(--text-muted)',
    },
    {
      icon: <Target />, label: 'Active Model',
      value: session?.has_model ? 'Ready' : 'None',
      color: session?.has_model ? 'var(--success)' : 'var(--text-muted)',
    },
  ];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-description">
          Welcome back, {user?.displayName || user?.username}. Here's your ML pipeline overview.
        </p>
      </div>

      <div className="card-grid" style={{ marginBottom: 28 }}>
        {stats.map((s, i) => (
          <div className="stat-card" key={i}>
            <div className="stat-icon" style={{ background: `${s.color}15`, color: s.color }}>
              {s.icon}
            </div>
            <div>
              <div className="stat-value" style={{ color: s.color }}>{s.value}</div>
              <div className="stat-label">{s.label}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="card">
        <div className="card-title" style={{ marginBottom: 16 }}>Quick Start Guide</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
          {[
            { step: '1', title: 'Upload Data', desc: 'Upload your credit dataset (CSV)' },
            { step: '2', title: 'Feature Engineering', desc: 'Encode, scale, handle outliers' },
            { step: '3', title: 'Train Model', desc: 'Select algorithm and train' },
            { step: '4', title: 'Explain & Predict', desc: 'SHAP analysis and scoring' },
          ].map(item => (
            <div key={item.step} style={{
              background: 'var(--bg-input)', borderRadius: 'var(--radius-sm)',
              padding: 16, border: '1px solid var(--border)'
            }}>
              <div style={{
                width: 28, height: 28, borderRadius: '50%', background: 'var(--accent-gradient)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 13, fontWeight: 700, color: 'white', marginBottom: 8,
              }}>{item.step}</div>
              <div style={{ fontWeight: 600, fontSize: 14 }}>{item.title}</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>{item.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
