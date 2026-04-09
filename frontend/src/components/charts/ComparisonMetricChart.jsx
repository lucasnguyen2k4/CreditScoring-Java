import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export default function ComparisonMetricChart({
  title,
  data,
  leftKey,
  rightKey,
  leftLabel,
  rightLabel,
}) {
  if (!data?.length) return null;

  return (
    <div className="card section">
      <div className="card-title" style={{ marginBottom: 16 }}>{title}</div>
      <div className="chart-surface">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
            <XAxis dataKey="label" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
            <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
            <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 10 }} />
            <Legend />
            <Bar dataKey={leftKey} name={leftLabel} fill="#f59e0b" radius={[6, 6, 0, 0]} />
            <Bar dataKey={rightKey} name={rightLabel} fill="#10b981" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
