import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export default function WoeBinsChart({ title, data }) {
  if (!data?.length) return null;

  return (
    <div className="card section">
      <div className="card-title" style={{ marginBottom: 16 }}>{title}</div>
      <div className="chart-surface">
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
            <XAxis dataKey="label" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} angle={-15} textAnchor="end" height={48} />
            <YAxis yAxisId="left" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
            <YAxis yAxisId="right" orientation="right" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
            <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 10 }} />
            <Legend />
            <Bar yAxisId="left" dataKey="count" name="Count" fill="#3b82f6" radius={[6, 6, 0, 0]} />
            <Line yAxisId="right" type="monotone" dataKey="woe" name="WoE" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
