import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

function formatNumber(value) {
  return typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value;
}

function formatSummaryValue(value) {
  if (value == null) return '—';
  if (typeof value === 'number') return formatNumber(value);
  if (typeof value === 'string' || typeof value === 'boolean') return String(value);
  if (Array.isArray(value)) return value.map((v) => formatSummaryValue(v)).join(', ');
  if (typeof value === 'object') {
    return Object.entries(value)
      .map(([k, v]) => `${k}: ${formatSummaryValue(v)}`)
      .join(' | ');
  }
  return String(value);
}

export default function DataDistributionCard({
  title,
  description,
  columns,
  selectedColumn,
  onColumnChange,
  distribution,
  loading,
  processed = false,
}) {
  return (
    <div className="card section">
      <div className="chart-header">
        <div>
          <div className="card-title">{title}</div>
          {description && <div className="chart-subtitle">{description}</div>}
        </div>
        <div className="chart-controls">
          <div className="form-group" style={{ margin: 0, minWidth: 220 }}>
            <label className="form-label">Column</label>
            <select className="form-select" value={selectedColumn} onChange={(e) => onColumnChange(e.target.value)}>
              {columns.map((column) => (
                <option key={column} value={column}>{column}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {loading && (
        <div className="flex items-center gap-sm">
          <div className="spinner" />
          Loading chart data...
        </div>
      )}

      {!loading && !distribution && (
        <div className="empty-state" style={{ padding: '32px 20px' }}>
          Select a column to visualize the data.
        </div>
      )}

      {!loading && distribution && (
        <>
          <div className="chart-surface">
            {distribution.data_type === 'numeric' ? (
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={distribution.histogram} margin={{ top: 10, right: 16, left: 0, bottom: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                  <XAxis
                    dataKey="label"
                    angle={-25}
                    height={72}
                    interval={Math.max(0, Math.floor((distribution.histogram?.length || 1) / 8))}
                    textAnchor="end"
                    tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                  />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 10 }}
                    formatter={(value) => [formatNumber(value), 'Count']}
                  />
                  <Bar dataKey="count" fill={processed ? '#38bdf8' : '#818cf8'} radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={distribution.categories} layout="vertical" margin={{ top: 10, right: 16, left: 48, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                  <XAxis type="number" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <YAxis
                    type="category"
                    dataKey="label"
                    width={120}
                    tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 10 }}
                    formatter={(value) => [formatNumber(value), 'Count']}
                  />
                  <Bar dataKey="count" fill={processed ? '#38bdf8' : '#818cf8'} radius={[0, 6, 6, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="summary-grid">
            {Object.entries(distribution.summary || {}).map(([key, value]) => (
              <div className="summary-card" key={key}>
                <div className="summary-label">{key.replace(/_/g, ' ')}</div>
                <div className="summary-value" style={{ fontSize: typeof value === 'object' ? 13 : 18 }}>
                  {formatSummaryValue(value)}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
