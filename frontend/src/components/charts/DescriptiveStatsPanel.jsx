import { useMemo } from 'react';
import { Download } from 'lucide-react';

const DESCRIPTIVE_KEYS = [
  { key: 'count', label: 'count' },
  { key: 'mean', label: 'mean' },
  { key: 'std', label: 'std' },
  { key: 'min', label: 'min' },
  { key: 'q1', label: '25%' },
  { key: 'median', label: '50%' },
  { key: 'q3', label: '75%' },
  { key: 'max', label: 'max' },
  { key: 'missing', label: 'missing' },
  { key: 'missing_pct', label: 'missing_pct' },
];

function formatNumber(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return '-';
  if (typeof value !== 'number') return String(value);
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

export default function DescriptiveStatsPanel({
  descriptiveRows,
  numericColumns,
  analysisColumn,
  onAnalysisColumnChange,
  analysisSummary,
  onExportCsv,
}) {
  const heatRanges = useMemo(() => {
    const out = {};
    DESCRIPTIVE_KEYS.forEach(({ key }) => {
      const values = descriptiveRows.map((r) => r[key]).filter((v) => typeof v === 'number' && !Number.isNaN(v));
      out[key] = {
        min: values.length ? Math.min(...values) : 0,
        max: values.length ? Math.max(...values) : 1,
      };
    });
    return out;
  }, [descriptiveRows]);

  const getHeatStyle = (key, value) => {
    if (typeof value !== 'number') return undefined;
    const range = heatRanges[key];
    const denom = Math.max(1e-9, range.max - range.min);
    const t = (value - range.min) / denom;
    const alpha = 0.15 + t * 0.45;
    return { background: `rgba(139, 92, 246, ${alpha.toFixed(3)})` };
  };

  return (
    <>
      <div className="card section">
        <div className="flex justify-between items-center" style={{ marginBottom: 12 }}>
          <div className="card-title">Descriptive Statistics</div>
          <button className="btn btn-secondary btn-sm" onClick={onExportCsv} disabled={!descriptiveRows.length}>
            <Download size={14} /> Export CSV
          </button>
        </div>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Variable</th>
                {DESCRIPTIVE_KEYS.map((k) => <th key={k.key}>{k.label}</th>)}
              </tr>
            </thead>
            <tbody>
              {descriptiveRows.map((row) => (
                <tr key={row.column}>
                  <td style={{ fontWeight: 700 }}>{row.column}</td>
                  {DESCRIPTIVE_KEYS.map((k) => (
                    <td key={`${row.column}-${k.key}`} style={getHeatStyle(k.key, row[k.key])}>
                      {formatNumber(row[k.key], 4)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card section">
        <div className="chart-header">
          <div className="card-title">Column Detail Summary</div>
          <div className="form-group" style={{ margin: 0, minWidth: 280 }}>
            <label className="form-label">Numeric Column</label>
            <select className="form-select" value={analysisColumn} onChange={(e) => onAnalysisColumnChange(e.target.value)}>
              {numericColumns.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        </div>
        <div className="summary-grid">
          <div className="summary-card"><div className="summary-label">Count</div><div className="summary-value">{formatNumber(analysisSummary?.count, 0)}</div></div>
          <div className="summary-card"><div className="summary-label">Mean</div><div className="summary-value">{formatNumber(analysisSummary?.mean, 2)}</div></div>
          <div className="summary-card"><div className="summary-label">Median</div><div className="summary-value">{formatNumber(analysisSummary?.median, 2)}</div></div>
          <div className="summary-card"><div className="summary-label">Std</div><div className="summary-value">{formatNumber(analysisSummary?.std, 2)}</div></div>
          <div className="summary-card"><div className="summary-label">Min</div><div className="summary-value">{formatNumber(analysisSummary?.min, 2)}</div></div>
          <div className="summary-card"><div className="summary-label">Max</div><div className="summary-value">{formatNumber(analysisSummary?.max, 2)}</div></div>
        </div>
      </div>
    </>
  );
}
