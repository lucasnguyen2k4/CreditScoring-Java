import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';

function formatNumber(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return '-';
  if (typeof value !== 'number') return String(value);
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

function BoxOutlierPlot({ summary, outlierValues = [] }) {
  if (!summary) return null;

  const box = {
    min: Number(summary.min ?? 0),
    q1: Number(summary.q1 ?? 0),
    median: Number(summary.median ?? 0),
    q3: Number(summary.q3 ?? 0),
    max: Number(summary.max ?? 0),
    lowerBound: Number(summary.lower_bound ?? 0),
    upperBound: Number(summary.upper_bound ?? 0),
  };

  const cleanOutliers = (outlierValues || [])
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v));

  const allValues = [box.min, box.max, box.lowerBound, box.upperBound, ...cleanOutliers].filter(Number.isFinite);
  let domainMin = Math.min(...allValues);
  let domainMax = Math.max(...allValues);
  if (domainMin === domainMax) {
    domainMin -= 1;
    domainMax += 1;
  }

  const width = 1000;
  const height = 220;
  const left = 60;
  const right = 60;
  const axisY = 130;
  const boxH = 54;
  const plotW = width - left - right;
  const toX = (v) => left + ((v - domainMin) / (domainMax - domainMin)) * plotW;

  const outlierDots = cleanOutliers.map((v, idx) => ({
    value: v,
    x: toX(v),
    y: axisY + ((idx % 7) - 3) * 6,
  }));

  return (
    <div style={{ width: '100%', height: 240 }}>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height="100%">
        <line x1={left} y1={axisY} x2={width - right} y2={axisY} stroke="rgba(148,163,184,0.35)" strokeWidth="1.5" />

        <line x1={toX(box.lowerBound)} y1="24" x2={toX(box.lowerBound)} y2={height - 28} stroke="rgba(245,158,11,0.8)" strokeDasharray="5 4" />
        <line x1={toX(box.upperBound)} y1="24" x2={toX(box.upperBound)} y2={height - 28} stroke="rgba(245,158,11,0.8)" strokeDasharray="5 4" />

        <line x1={toX(box.min)} y1={axisY} x2={toX(box.max)} y2={axisY} stroke="rgba(148,163,184,0.7)" strokeWidth="3" />
        <line x1={toX(box.min)} y1={axisY - 14} x2={toX(box.min)} y2={axisY + 14} stroke="rgba(148,163,184,0.95)" strokeWidth="2.5" />
        <line x1={toX(box.max)} y1={axisY - 14} x2={toX(box.max)} y2={axisY + 14} stroke="rgba(148,163,184,0.95)" strokeWidth="2.5" />

        <rect
          x={Math.min(toX(box.q1), toX(box.q3))}
          y={axisY - boxH / 2}
          width={Math.max(2, Math.abs(toX(box.q3) - toX(box.q1)))}
          height={boxH}
          fill="rgba(168,85,247,0.5)"
          stroke="rgba(168,85,247,0.9)"
          strokeWidth="1.5"
        />
        <line x1={toX(box.median)} y1={axisY - boxH / 2} x2={toX(box.median)} y2={axisY + boxH / 2} stroke="#f59e0b" strokeWidth="3" />

        {outlierDots.map((p, idx) => (
          <circle key={`${p.value}-${idx}`} cx={p.x} cy={p.y} r="4.2" fill="rgba(239,68,68,0.95)" />
        ))}

        <text x={toX(box.lowerBound)} y="18" textAnchor="middle" fill="var(--text-muted)" fontSize="12">LB</text>
        <text x={toX(box.upperBound)} y="18" textAnchor="middle" fill="var(--text-muted)" fontSize="12">UB</text>
        <text x={toX(box.min)} y={height - 10} textAnchor="middle" fill="var(--text-muted)" fontSize="11">min</text>
        <text x={toX(box.max)} y={height - 10} textAnchor="middle" fill="var(--text-muted)" fontSize="11">max</text>
      </svg>
    </div>
  );
}

export default function DistributionDetailPanel({
  analysisColumn,
  numericColumns,
  onAnalysisColumnChange,
  analysisLoading,
  analysisData,
}) {
  const quantiles = analysisData?.summary?.quantiles || {};
  const quantileRows = [
    ['1%', quantiles.p01],
    ['5%', quantiles.p05],
    ['25%', quantiles.p25],
    ['50% (Median)', quantiles.p50],
    ['75%', quantiles.p75],
    ['95%', quantiles.p95],
    ['99%', quantiles.p99],
  ];

  return (
    <>
      <div className="chart-header">
        <div className="form-group" style={{ margin: 0, minWidth: 280 }}>
          <label className="form-label">Numeric Column</label>
          <select className="form-select" value={analysisColumn} onChange={(e) => onAnalysisColumnChange(e.target.value)}>
            {numericColumns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      </div>

      {analysisLoading && <div className="flex items-center gap-sm mt-md"><div className="spinner" /> Loading analysis...</div>}

      {!analysisLoading && analysisData?.data_type === 'numeric' && (
        <>
          <div className="analysis-grid section">
            <div className="card">
              <div className="card-title" style={{ marginBottom: 8 }}>Histogram</div>
              <div className="chart-surface" style={{ marginBottom: 0 }}>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={analysisData.histogram}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                    <XAxis dataKey="label" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} interval={Math.max(0, Math.floor((analysisData.histogram.length || 1) / 8))} />
                    <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 10 }} />
                    <Bar dataKey="count" fill="#7389ff" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="card">
              <div className="card-title" style={{ marginBottom: 8 }}>Box / IQR + Outliers</div>
              <div className="chart-surface" style={{ marginBottom: 0 }}>
                <BoxOutlierPlot
                  summary={analysisData.box_stats || analysisData.summary}
                  outlierValues={analysisData.outlier_values}
                />
                <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text-muted)' }}>
                  Red dots are outliers (outside IQR bounds). Dashed lines are lower/upper IQR bounds.
                </div>
              </div>
            </div>
          </div>

          <div className="analysis-grid section">
            <div className="card">
              <div className="card-title" style={{ marginBottom: 8 }}>Quantiles</div>
              <div className="table-container">
                <table>
                  <thead><tr><th>Quantile</th><th>Value</th></tr></thead>
                  <tbody>
                    {quantileRows.map(([q, v]) => <tr key={q}><td>{q}</td><td>{formatNumber(v, 4)}</td></tr>)}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="card">
              <div className="card-title" style={{ marginBottom: 8 }}>Outlier Analysis (IQR)</div>
              <div className="table-container">
                <table>
                  <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                  <tbody>
                    <tr><td>Lower bound</td><td>{formatNumber(analysisData.summary?.lower_bound, 4)}</td></tr>
                    <tr><td>Upper bound</td><td>{formatNumber(analysisData.summary?.upper_bound, 4)}</td></tr>
                    <tr><td>Outlier count</td><td>{formatNumber(analysisData.summary?.outliers_count, 0)}</td></tr>
                    <tr><td>Outlier ratio (%)</td><td>{formatNumber(analysisData.summary?.outliers_ratio_pct, 2)}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="card section">
            <div className="card-title" style={{ marginBottom: 12 }}>Distribution Features</div>
            <div className="summary-grid">
              <div className="summary-card"><div className="summary-label">Skewness</div><div className="summary-value">{formatNumber(analysisData.summary?.skewness, 3)}</div></div>
              <div className="summary-card"><div className="summary-label">Kurtosis</div><div className="summary-value">{formatNumber(analysisData.summary?.kurtosis, 3)}</div></div>
              <div className="summary-card"><div className="summary-label">Range</div><div className="summary-value">{formatNumber(analysisData.summary?.range, 3)}</div></div>
              <div className="summary-card"><div className="summary-label">CV (%)</div><div className="summary-value">{formatNumber(analysisData.summary?.cv_pct, 2)}</div></div>
            </div>
          </div>
        </>
      )}
    </>
  );
}
