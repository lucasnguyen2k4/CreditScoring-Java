import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell,
} from 'recharts';

function formatNumber(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return '-';
  if (typeof value !== 'number') return String(value);
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

export default function BinnedCategoricalPanel({
  analysisColumn,
  numericColumns,
  onAnalysisColumnChange,
  analysisBins,
  onAnalysisBinsChange,
  binnedRows,
  categoricalSummary,
  catLoading,
}) {
  return (
    <>
      <div className="card section">
        <div className="chart-header">
          <div>
            <div className="card-title">Binned Value Distribution</div>
            <div className="chart-subtitle">Choose a numeric column and number of bins.</div>
          </div>
          <div className="chart-controls">
            <div className="form-group" style={{ margin: 0, minWidth: 220 }}>
              <label className="form-label">Numeric Column</label>
              <select className="form-select" value={analysisColumn} onChange={(e) => onAnalysisColumnChange(e.target.value)}>
                {numericColumns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div className="form-group" style={{ margin: 0, minWidth: 220 }}>
              <label className="form-label">Number of bins: {analysisBins}</label>
              <input type="range" className="form-input" min="5" max="20" value={analysisBins} onChange={(e) => onAnalysisBinsChange(+e.target.value)} />
            </div>
          </div>
        </div>

        <div className="table-container" style={{ marginBottom: 16 }}>
          <table>
            <thead><tr><th>Value Range</th><th>Count</th><th>Ratio (%)</th></tr></thead>
            <tbody>
              {binnedRows.map((r) => (
                <tr key={r.range}>
                  <td>{r.range}</td>
                  <td>{r.count}</td>
                  <td>{r.ratio_pct}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="chart-surface" style={{ marginBottom: 0 }}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={binnedRows}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
              <XAxis dataKey="range" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} angle={-25} textAnchor="end" height={80} />
              <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 10 }} />
              <Bar dataKey="count">
                {binnedRows.map((row) => (
                  <Cell key={row.range} fill={`hsl(${220 - Math.round(row.ratio_pct * 1.8)}, 72%, 55%)`} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card section">
        <div className="card-title" style={{ marginBottom: 12 }}>Categorical Variables</div>
        {catLoading && <div className="flex items-center gap-sm"><div className="spinner" /> Loading categorical summary...</div>}
        {!catLoading && (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Column Name</th>
                  <th>Unique Values</th>
                  <th>Most Common</th>
                  <th>Top Frequency</th>
                  <th>Missing</th>
                  <th>Missing Ratio (%)</th>
                </tr>
              </thead>
              <tbody>
                {(categoricalSummary?.columns || []).map((row) => (
                  <tr key={row.column_name}>
                    <td style={{ fontWeight: 700 }}>{row.column_name}</td>
                    <td>{row.unique_values}</td>
                    <td>{row.most_common || '-'}</td>
                    <td>{row.top_frequency}</td>
                    <td>{row.missing}</td>
                    <td>{formatNumber(row.missing_ratio_pct, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
