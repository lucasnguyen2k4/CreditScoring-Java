/**
 * BoxPlotChart — Horizontal box-and-whisker chart matching the Streamlit reference.
 *
 * Props
 *   boxStats   – { min, q1, median, q3, max } (numbers)
 *   column     – column name shown on the x-axis label
 *   color      – accent color (default #667eea)
 *   height     – SVG height (default 120)
 */
export default function BoxPlotChart({
  boxStats,
  column = '',
  color = '#667eea',
  height = 120,
}) {
  if (!boxStats) return null;

  const { min, q1, median, q3, max } = boxStats;
  if ([min, q1, median, q3, max].some((v) => v == null || !Number.isFinite(+v))) return null;

  const nMin = +min;
  const nQ1 = +q1;
  const nMed = +median;
  const nQ3 = +q3;
  const nMax = +max;

  // Layout
  const paddingLeft = 40;
  const paddingRight = 24;
  const paddingTop = 18;
  const paddingBottom = 36;
  const boxH = 28;  // height of the box rectangle
  const midY = paddingTop + (height - paddingTop - paddingBottom) / 2;

  // Scale
  const range = nMax - nMin || 1;
  const margin = range * 0.08;          // 8% margin on each side
  const domainMin = nMin - margin;
  const domainMax = nMax + margin;
  const plotW = 1000 - paddingLeft - paddingRight;  // viewBox 1000 wide
  const toX = (v) => paddingLeft + ((v - domainMin) / (domainMax - domainMin)) * plotW;

  const xMin = toX(nMin);
  const xQ1 = toX(nQ1);
  const xMed = toX(nMed);
  const xQ3 = toX(nQ3);
  const xMax = toX(nMax);

  // Tick labels (5 ticks)
  const ticks = [nMin, nQ1, nMed, nQ3, nMax];

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
      <svg viewBox={`0 0 1000 ${height}`} width="100%" height="100%" style={{ display: 'block' }}>
        {/* horizontal grid line */}
        <line x1={paddingLeft} y1={midY} x2={1000 - paddingRight} y2={midY} stroke="rgba(148,163,184,0.12)" />

        {/* whiskers (min to Q1, Q3 to max) */}
        <line x1={xMin} y1={midY} x2={xQ1} y2={midY} stroke={color} strokeWidth="2" />
        <line x1={xQ3} y1={midY} x2={xMax} y2={midY} stroke={color} strokeWidth="2" />

        {/* min cap */}
        <line x1={xMin} y1={midY - boxH / 3} x2={xMin} y2={midY + boxH / 3} stroke={color} strokeWidth="2" />
        {/* max cap */}
        <line x1={xMax} y1={midY - boxH / 3} x2={xMax} y2={midY + boxH / 3} stroke={color} strokeWidth="2" />

        {/* box (Q1 to Q3) */}
        <rect
          x={xQ1}
          y={midY - boxH / 2}
          width={Math.max(2, xQ3 - xQ1)}
          height={boxH}
          fill={`${color}55`}
          stroke={color}
          strokeWidth="2"
          rx="3"
        />

        {/* median line */}
        <line x1={xMed} y1={midY - boxH / 2} x2={xMed} y2={midY + boxH / 2} stroke="#f59e0b" strokeWidth="2.5" />

        {/* tick labels */}
        {ticks.map((v, i) => (
          <text
            key={`tick-${i}`}
            x={toX(v)}
            y={height - paddingBottom + 18}
            textAnchor="middle"
            fill="var(--text-muted)"
            fontSize="11"
          >
            {typeof v === 'number' ? v.toLocaleString(undefined, { maximumFractionDigits: 1 }) : v}
          </text>
        ))}

        {/* x-axis label */}
        {column && (
          <text
            x={(paddingLeft + 1000 - paddingRight) / 2}
            y={height - 4}
            textAnchor="middle"
            fill="var(--text-muted)"
            fontSize="12"
            fontWeight="600"
          >
            {column}
          </text>
        )}

        {/* y-axis labels – just show tick marks for min/max visually */}
        {[nMin, nQ1, nMed, nQ3, nMax].map((v, i) => (
          <line
            key={`xt-${i}`}
            x1={toX(v)}
            y1={height - paddingBottom}
            x2={toX(v)}
            y2={height - paddingBottom + 5}
            stroke="var(--text-muted)"
            strokeWidth="1"
          />
        ))}

        {/* bottom axis line */}
        <line x1={paddingLeft} y1={height - paddingBottom} x2={1000 - paddingRight} y2={height - paddingBottom} stroke="rgba(148,163,184,0.25)" />
      </svg>
    </div>
  );
}
