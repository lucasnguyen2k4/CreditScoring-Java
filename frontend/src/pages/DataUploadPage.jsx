import { useEffect, useMemo, useRef, useState } from 'react';
import { dataApi, llmApi } from '../api/client';
import { Upload, FileSpreadsheet, AlertCircle, Sparkles } from 'lucide-react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import DataDistributionCard from '../components/charts/DataDistributionCard';
import DescriptiveStatsPanel from '../components/charts/DescriptiveStatsPanel';
import DistributionDetailPanel from '../components/charts/DistributionDetailPanel';
import BinnedCategoricalPanel from '../components/charts/BinnedCategoricalPanel';
import ColumnMultiSelector from '../components/ColumnMultiSelector';

function downloadCsv(filename, rows) {
  const csvText = rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n');
  const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function formatNum(value, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function jitter(seed) {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return (x - Math.floor(x)) - 0.5;
}

function GroupedDistributionPlot({ type, groupedData }) {
  const groups = groupedData?.shown_groups || [];
  const stats = groupedData?.stats || [];
  const points = groupedData?.points || [];
  if (!groups.length) return null;

  const width = Math.max(920, groups.length * 120);
  const height = 360;
  const left = 60;
  const right = 24;
  const top = 20;
  const bottom = 80;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const step = plotW / groups.length;
  const halfStep = step / 2;

  const numericValues = points.map((p) => Number(p.value)).filter((v) => Number.isFinite(v));
  const fallbackMin = Math.min(...stats.map((s) => Number(s.min)).filter((v) => Number.isFinite(v)));
  const fallbackMax = Math.max(...stats.map((s) => Number(s.max)).filter((v) => Number.isFinite(v)));
  let minVal = numericValues.length ? Math.min(...numericValues) : fallbackMin;
  let maxVal = numericValues.length ? Math.max(...numericValues) : fallbackMax;
  if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null;
  if (minVal === maxVal) {
    minVal -= 1;
    maxVal += 1;
  }
  const toY = (v) => top + ((maxVal - v) / (maxVal - minVal)) * plotH;
  const toX = (idx) => left + (idx * step) + halfStep;

  const yTicks = Array.from({ length: 6 }).map((_, i) => minVal + ((maxVal - minVal) * i / 5));
  const statsByGroup = Object.fromEntries(stats.map((s) => [s.group, s]));

  const renderBox = () => groups.map((g, idx) => {
    const s = statsByGroup[g];
    if (!s) return null;
    const x = toX(idx);
    const yMin = toY(Number(s.min));
    const yQ1 = toY(Number(s.q1));
    const yMed = toY(Number(s.median));
    const yQ3 = toY(Number(s.q3));
    const yMax = toY(Number(s.max));
    const boxW = Math.min(34, step * 0.45);
    return (
      <g key={`box-${g}`}>
        <line x1={x} y1={yMin} x2={x} y2={yMax} stroke="rgba(148,163,184,0.75)" strokeWidth="1.5" />
        <line x1={x - boxW / 2} y1={yMin} x2={x + boxW / 2} y2={yMin} stroke="rgba(148,163,184,0.9)" />
        <line x1={x - boxW / 2} y1={yMax} x2={x + boxW / 2} y2={yMax} stroke="rgba(148,163,184,0.9)" />
        <rect
          x={x - boxW / 2}
          y={Math.min(yQ1, yQ3)}
          width={boxW}
          height={Math.max(2, Math.abs(yQ3 - yQ1))}
          fill="rgba(168,85,247,0.36)"
          stroke="rgba(168,85,247,0.85)"
        />
        <line x1={x - boxW / 2} y1={yMed} x2={x + boxW / 2} y2={yMed} stroke="#ef4444" strokeWidth="2" />
      </g>
    );
  });

  const renderViolin = () => {
    const bins = 20;
    const maxHalfWidth = Math.min(30, step * 0.42);
    return groups.map((g, idx) => {
      const vals = points.filter((p) => p.group === g).map((p) => Number(p.value)).filter((v) => Number.isFinite(v));
      if (vals.length < 2) return null;
      const hist = Array.from({ length: bins }, () => 0);
      vals.forEach((v) => {
        const t = (v - minVal) / (maxVal - minVal);
        const bi = Math.max(0, Math.min(bins - 1, Math.floor(t * bins)));
        hist[bi] += 1;
      });
      const peak = Math.max(...hist, 1);
      const xCenter = toX(idx);
      const rightPts = hist.map((count, bi) => {
        const frac = count / peak;
        const yVal = minVal + ((bi + 0.5) / bins) * (maxVal - minVal);
        return { x: xCenter + frac * maxHalfWidth, y: toY(yVal) };
      });
      const leftPts = hist.map((count, bi) => {
        const frac = count / peak;
        const yVal = minVal + ((bi + 0.5) / bins) * (maxVal - minVal);
        return { x: xCenter - frac * maxHalfWidth, y: toY(yVal) };
      }).reverse();
      const all = [...rightPts, ...leftPts];
      const d = all.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' ') + ' Z';
      const s = statsByGroup[g];
      const yMed = s ? toY(Number(s.median)) : null;
      return (
        <g key={`violin-${g}`}>
          <path d={d} fill="rgba(14,165,233,0.24)" stroke="rgba(14,165,233,0.9)" strokeWidth="1.2" />
          {yMed != null && <line x1={xCenter - maxHalfWidth * 0.55} y1={yMed} x2={xCenter + maxHalfWidth * 0.55} y2={yMed} stroke="#f59e0b" strokeWidth="2" />}
        </g>
      );
    });
  };

  const renderStrip = () => points.map((p, idx) => {
    const gi = groups.indexOf(p.group);
    if (gi < 0) return null;
    const x = toX(gi) + jitter(idx + Number(p.value || 0)) * Math.min(16, step * 0.32);
    const y = toY(Number(p.value));
    return <circle key={`pt-${p.group}-${idx}`} cx={x} cy={y} r="2.6" fill="rgba(96,165,250,0.86)" />;
  });

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height="100%">
        {yTicks.map((tick, i) => {
          const y = toY(tick);
          return (
            <g key={`grid-${i}`}>
              <line x1={left} y1={y} x2={width - right} y2={y} stroke="rgba(148,163,184,0.16)" />
              <text x={left - 8} y={y + 4} textAnchor="end" fill="var(--text-muted)" fontSize="11">{formatNum(tick, 1)}</text>
            </g>
          );
        })}

        {type === 'box' && renderBox()}
        {type === 'violin' && renderViolin()}
        {type === 'strip' && renderStrip()}

        {groups.map((g, idx) => {
          const x = toX(idx);
          return (
            <g key={`gx-${g}`}>
              <line x1={x} y1={top} x2={x} y2={height - bottom} stroke="rgba(148,163,184,0.08)" />
              <text x={x} y={height - bottom + 18} textAnchor="end" fill="var(--text-muted)" fontSize="10" transform={`rotate(-35 ${x} ${height - bottom + 18})`}>
                {g}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export default function DataUploadPage() {
  const [data, setData] = useState(null);
  const [info, setInfo] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [tab, setTab] = useState('preview');
  const [aiAnalysis, setAiAnalysis] = useState('');
  const [aiLoading, setAiLoading] = useState(false);
  const [visualColumn, setVisualColumn] = useState('');
  const [visualData, setVisualData] = useState(null);
  const [visualLoading, setVisualLoading] = useState(false);
  const [analysisColumn, setAnalysisColumn] = useState('');
  const [analysisBins, setAnalysisBins] = useState(10);
  const [analysisData, setAnalysisData] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [categoricalSummary, setCategoricalSummary] = useState(null);
  const [catLoading, setCatLoading] = useState(false);
  const [correlationData, setCorrelationData] = useState(null);
  const [correlationLoading, setCorrelationLoading] = useState(false);
  const [corrMethod, setCorrMethod] = useState('pearson');
  const [corrThreshold, setCorrThreshold] = useState(0.8);
  const [jointColumnX, setJointColumnX] = useState('');
  const [jointColumnY, setJointColumnY] = useState('');
  const [jointBins, setJointBins] = useState(8);
  const [jointData, setJointData] = useState(null);
  const [jointLoading, setJointLoading] = useState(false);
  const [scatterColumns, setScatterColumns] = useState([]);
  const [scatterData, setScatterData] = useState(null);
  const [scatterLoading, setScatterLoading] = useState(false);
  const [scatterMaxPoints, setScatterMaxPoints] = useState(800);
  const [groupValueColumn, setGroupValueColumn] = useState('');
  const [groupByColumn, setGroupByColumn] = useState('');
  const [groupTopN, setGroupTopN] = useState(10);
  const [groupedData, setGroupedData] = useState(null);
  const [groupedLoading, setGroupedLoading] = useState(false);
  const [groupChartType, setGroupChartType] = useState('box');
  const fileRef = useRef();
  const [initialLoading, setInitialLoading] = useState(true);

  // Auto-restore data if already uploaded on the backend (e.g. after navigating away and back)
  useEffect(() => {
    let cancelled = false;
    const restoreSession = async () => {
      try {
        const sessionRes = await dataApi.getSessionInfo();
        if (cancelled) return;
        if (!sessionRes.data?.has_data) return;

        // Data exists on the server — fetch preview, info, stats
        const [previewRes, infoRes, statsRes] = await Promise.all([
          dataApi.getPreview(100),
          dataApi.getInfo(),
          dataApi.getStats(),
        ]);
        if (cancelled) return;

        setData(previewRes.data);
        setInfo(infoRes.data);
        setStats(statsRes.data.numeric_stats);

        // Set default columns
        const numericCols = infoRes.data.numeric_columns || [];
        const catCols = infoRes.data.categorical_columns || [];
        const defaultColumn = numericCols[0] || catCols[0] || '';
        const defaultNumeric = numericCols[0] || '';
        setVisualColumn(defaultColumn);
        setAnalysisColumn(defaultNumeric);
        setScatterColumns(numericCols.slice(0, 5));
        setGroupValueColumn(defaultNumeric);
        setGroupByColumn(catCols[0] || infoRes.data.columns?.find((c) => c !== defaultNumeric) || '');

        // Load categorical summary
        try {
          const catRes = await dataApi.getCategoricalSummary(false);
          if (!cancelled) setCategoricalSummary(catRes.data);
        } catch { /* ignore */ }
      } catch { /* no session or no data — show upload zone */ }
      finally {
        if (!cancelled) setInitialLoading(false);
      }
    };
    restoreSession();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const columns = [
      ...(info?.numeric_columns || []),
      ...(info?.categorical_columns || []),
    ];

    if (!columns.length) {
      setVisualColumn('');
      setVisualData(null);
      setAnalysisColumn('');
      setAnalysisData(null);
      return;
    }

    if (!visualColumn || !columns.includes(visualColumn)) {
      setVisualColumn(columns[0]);
    }
    const numericCols = info?.numeric_columns || [];
    const categoricalCols = info?.categorical_columns || [];
    if (!analysisColumn || !numericCols.includes(analysisColumn)) {
      setAnalysisColumn(numericCols[0] || columns[0]);
    }
    if (!jointColumnX || !columns.includes(jointColumnX)) {
      setJointColumnX(columns[0]);
    }
    if (!jointColumnY || !columns.includes(jointColumnY) || jointColumnY === (jointColumnX || columns[0])) {
      setJointColumnY(columns[1] || columns[0]);
    }
    if (!scatterColumns.length && numericCols.length >= 2) {
      setScatterColumns(numericCols.slice(0, 5));
    } else if (scatterColumns.length) {
      const filtered = scatterColumns.filter((c) => numericCols.includes(c)).slice(0, 5);
      if (filtered.length !== scatterColumns.length) {
        setScatterColumns(filtered);
      }
    }
    if (!groupValueColumn || !numericCols.includes(groupValueColumn)) {
      setGroupValueColumn(numericCols[0] || '');
    }
    if (!groupByColumn || !columns.includes(groupByColumn) || groupByColumn === (numericCols[0] || '')) {
      const fallback = categoricalCols[0] || columns.find((c) => c !== (numericCols[0] || '')) || '';
      setGroupByColumn(fallback);
    }
  }, [info, visualColumn, analysisColumn, jointColumnX, jointColumnY, scatterColumns, groupValueColumn, groupByColumn]);

  useEffect(() => {
    if (!visualColumn) return;
    loadDistribution(visualColumn);
  }, [visualColumn]);

  useEffect(() => {
    if (!analysisColumn) return;
    loadAnalysisDistribution(analysisColumn, analysisBins);
  }, [analysisColumn, analysisBins]);

  useEffect(() => {
    if (tab !== 'correlation') return;
    if (!info?.numeric_columns?.length) return;
    loadCorrelation();
  }, [tab, corrMethod, info]);

  useEffect(() => {
    if (tab !== 'joint-distribution') return;
    if (!jointColumnX || !jointColumnY) return;
    loadJointDistribution(jointColumnX, jointColumnY, jointBins);
  }, [tab, jointColumnX, jointColumnY, jointBins]);

  useEffect(() => {
    if (tab !== 'scatter-matrix') return;
    if ((scatterColumns || []).length < 2) return;
    loadScatterMatrix(scatterColumns);
  }, [tab, scatterColumns, scatterMaxPoints]);

  useEffect(() => {
    if (tab !== 'grouped-analysis') return;
    if (!groupValueColumn || !groupByColumn) return;
    loadGroupedAnalysis(groupValueColumn, groupByColumn);
  }, [tab, groupValueColumn, groupByColumn, groupTopN]);

  useEffect(() => {
    if (!info?.columns?.length) return;
    if (!groupValueColumn || !groupByColumn || groupValueColumn !== groupByColumn) return;
    const fallback = (info.categorical_columns || []).find((c) => c !== groupValueColumn)
      || info.columns.find((c) => c !== groupValueColumn)
      || '';
    setGroupByColumn(fallback);
  }, [info, groupValueColumn, groupByColumn]);

  const loadDistribution = async (column) => {
    setVisualLoading(true);
    try {
      const res = await dataApi.getDistribution(column, false, 24);
      setVisualData(res.data);
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
      setVisualData(null);
    } finally {
      setVisualLoading(false);
    }
  };

  const loadAnalysisDistribution = async (column, bins) => {
    setAnalysisLoading(true);
    try {
      const res = await dataApi.getDistribution(column, false, bins);
      setAnalysisData(res.data);
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
      setAnalysisData(null);
    } finally {
      setAnalysisLoading(false);
    }
  };

  const loadCategoricalSummary = async () => {
    setCatLoading(true);
    try {
      const res = await dataApi.getCategoricalSummary(false);
      setCategoricalSummary(res.data);
    } catch {
      setCategoricalSummary(null);
    } finally {
      setCatLoading(false);
    }
  };

  const loadCorrelation = async () => {
    setCorrelationLoading(true);
    try {
      const res = await dataApi.getCorrelation(false, corrMethod, corrThreshold);
      setCorrelationData(res.data);
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
      setCorrelationData(null);
    } finally {
      setCorrelationLoading(false);
    }
  };

  const loadJointDistribution = async (colX, colY, bins) => {
    setJointLoading(true);
    try {
      const res = await dataApi.getJointDistribution(colX, colY, false, bins);
      setJointData(res.data);
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
      setJointData(null);
    } finally {
      setJointLoading(false);
    }
  };

  const loadScatterMatrix = async (columns = scatterColumns) => {
    setScatterLoading(true);
    try {
      const cols = (columns || []).slice(0, 5);
      const res = await dataApi.getScatterMatrix({
        columns: cols,
        processed: false,
        max_points: scatterMaxPoints,
      });
      setScatterData(res.data);
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
      setScatterData(null);
    } finally {
      setScatterLoading(false);
    }
  };

  const loadGroupedAnalysis = async (valueCol = groupValueColumn, groupCol = groupByColumn) => {
    setGroupedLoading(true);
    try {
      const res = await dataApi.getGroupedAnalysis(valueCol, groupCol, false, groupTopN, 2500);
      setGroupedData(res.data);
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
      setGroupedData(null);
    } finally {
      setGroupedLoading(false);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setLoading(true);
    setMessage('');
    try {
      const res = await dataApi.upload(file);
      setData(res.data.preview);
      setMessage(res.data.message);
      const [infoRes, statsRes] = await Promise.all([
        dataApi.getInfo(),
        dataApi.getStats(),
      ]);
      setInfo(infoRes.data);
      setStats(statsRes.data.numeric_stats);
      const defaultColumn = infoRes.data.numeric_columns?.[0] || infoRes.data.categorical_columns?.[0] || '';
      const defaultNumeric = infoRes.data.numeric_columns?.[0] || '';
      const numericCols = infoRes.data.numeric_columns || [];
      const catCols = infoRes.data.categorical_columns || [];
      setVisualColumn(defaultColumn);
      setAnalysisColumn(defaultNumeric);
      setAnalysisBins(10);
      setScatterColumns(numericCols.slice(0, 5));
      setScatterData(null);
      setGroupValueColumn(defaultNumeric);
      setGroupByColumn(catCols[0] || infoRes.data.columns?.find((c) => c !== defaultNumeric) || '');
      setGroupedData(null);
      loadCategoricalSummary();
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateSample = async () => {
    setLoading(true);
    setMessage('');
    try {
      const res = await dataApi.generateSample(100);
      setData(res.data.preview);
      setMessage(res.data.message);
      const [infoRes, statsRes] = await Promise.all([
        dataApi.getInfo(),
        dataApi.getStats(),
      ]);
      setInfo(infoRes.data);
      setStats(statsRes.data.numeric_stats);
      const defaultColumn = infoRes.data.numeric_columns?.[0] || infoRes.data.categorical_columns?.[0] || '';
      const defaultNumeric = infoRes.data.numeric_columns?.[0] || '';
      const numericCols = infoRes.data.numeric_columns || [];
      const catCols = infoRes.data.categorical_columns || [];
      setVisualColumn(defaultColumn);
      setAnalysisColumn(defaultNumeric);
      setAnalysisBins(10);
      setScatterColumns(numericCols.slice(0, 5));
      setScatterData(null);
      setGroupValueColumn(defaultNumeric);
      setGroupByColumn(catCols[0] || infoRes.data.columns?.find((c) => c !== defaultNumeric) || '');
      setGroupedData(null);
      loadCategoricalSummary();
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const runAiAnalysis = async () => {
    setAiLoading(true);
    try {
      const res = await llmApi.analyzeEda();
      setAiAnalysis(res.data.analysis);
    } catch (err) {
      setAiAnalysis('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setAiLoading(false);
    }
  };

  const visualColumns = [
    ...(info?.numeric_columns || []),
    ...(info?.categorical_columns || []),
  ];
  const numericColumns = info?.numeric_columns || [];

  const descriptiveRows = useMemo(() => {
    if (!stats || !numericColumns.length) return [];
    const totalRows = info?.shape?.[0] || 0;
    return numericColumns
      .filter((col) => stats[col])
      .map((col) => {
        const s = stats[col];
        const missingCount = info?.missing?.[col]?.count ?? Math.max(0, totalRows - (s.count || 0));
        return {
          column: col,
          count: s.count ?? 0,
          mean: s.mean,
          std: s.std,
          min: s.min,
          q1: s.q1,
          median: s.median,
          q3: s.q3,
          max: s.max,
          missing: missingCount,
          missing_pct: totalRows > 0 ? (missingCount / totalRows) * 100 : 0,
        };
      });
  }, [stats, numericColumns, info]);

  const analysisSummary = useMemo(() => {
    if (analysisData?.summary) return analysisData.summary;
    if (analysisColumn && stats?.[analysisColumn]) return stats[analysisColumn];
    return null;
  }, [analysisData, analysisColumn, stats]);

  const binnedRows = useMemo(() => {
    const bins = analysisData?.histogram || [];
    if (!bins.length) return [];
    const total = bins.reduce((sum, b) => sum + (b.count || 0), 0);
    return bins.map((b) => ({
      range: b.label,
      count: b.count || 0,
      ratio_pct: total > 0 ? Number((((b.count || 0) / total) * 100).toFixed(2)) : 0,
    }));
  }, [analysisData]);

  const handleExportDescriptiveCsv = () => {
    const header = ['column', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing', 'missing_pct'];
    const rows = descriptiveRows.map((r) => [
      r.column,
      r.count,
      r.mean,
      r.std,
      r.min,
      r.q1,
      r.median,
      r.q3,
      r.max,
      r.missing,
      r.missing_pct,
    ]);
    downloadCsv('descriptive_statistics.csv', [header, ...rows]);
  };

  const allColumns = info?.columns || visualColumns;

  const correlationCellStyle = (value) => {
    if (typeof value !== 'number') return {};
    const intensity = Math.min(1, Math.abs(value));
    const alpha = 0.12 + intensity * 0.68;
    const bg = value >= 0
      ? `rgba(239, 68, 68, ${alpha.toFixed(3)})`
      : `rgba(59, 130, 246, ${alpha.toFixed(3)})`;
    return {
      background: bg,
      color: intensity > 0.45 ? '#f8fafc' : 'var(--text-primary)',
      fontWeight: 600,
    };
  };

  const jointCellStyle = (count, maxCount) => {
    if (!maxCount) return {};
    const ratio = Math.max(0, Math.min(1, Number(count || 0) / maxCount));
    const alpha = 0.1 + ratio * 0.7;
    const bg = `rgba(99, 102, 241, ${alpha.toFixed(3)})`;
    return {
      background: bg,
      color: ratio > 0.45 ? '#f8fafc' : 'var(--text-primary)',
      fontWeight: 600,
    };
  };

  const jointTableRows = useMemo(() => {
    if (!jointData?.rows?.length || !jointData?.y_labels?.length) return [];
    const rows = [];
    jointData.rows.forEach((r) => {
      (jointData.y_labels || []).forEach((yLabel, idx) => {
        rows.push({
          x: r.x_value,
          y: yLabel,
          count: r.counts?.[idx] ?? 0,
          ratio: r.row_ratio_pct?.[idx] ?? 0,
        });
      });
    });
    return rows;
  }, [jointData]);

  const jointMaxCount = useMemo(() => {
    if (!jointData?.rows?.length) return 0;
    let max = 0;
    jointData.rows.forEach((r) => {
      (r.counts || []).forEach((v) => { if (v > max) max = v; });
    });
    return max;
  }, [jointData]);

  const scatterPairRows = useMemo(() => {
    const cols = scatterData?.columns || [];
    if (cols.length < 2) return [];
    const rows = [];
    for (let yIdx = 1; yIdx < cols.length; yIdx += 1) {
      const pairRow = [];
      for (let xIdx = 0; xIdx < yIdx; xIdx += 1) {
        pairRow.push({ xKey: cols[xIdx], yKey: cols[yIdx] });
      }
      rows.push(pairRow);
    }
    return rows;
  }, [scatterData]);

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Data Upload & EDA</h1>
        <p className="page-description">Upload your credit dataset and explore the data</p>
      </div>

      {message && <div className={`alert ${message.startsWith('Error') ? 'alert-error' : 'alert-success'}`}>{message}</div>}

      {initialLoading && !data && (
        <div className="flex items-center gap-sm mt-md" style={{ justifyContent: 'center', padding: 40 }}>
          <div className="spinner" /> Loading session data...
        </div>
      )}

      {!data && !initialLoading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', alignItems: 'center' }}>
          <div className="upload-zone" onClick={() => fileRef.current?.click()}>
            <Upload size={40} style={{ color: 'var(--text-muted)', marginBottom: 12 }} />
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Click to upload CSV file</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>or drag and drop</div>
            <input ref={fileRef} type="file" accept=".csv" onChange={handleUpload} hidden />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
            <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>No data available?</div>
            <button 
              className="btn btn-secondary" 
              onClick={handleGenerateSample}
              disabled={loading}
              style={{ display: 'flex', gap: '8px', alignItems: 'center' }}
            >
              <Sparkles size={16} /> Generate Sample Data
            </button>
          </div>
        </div>
      )}

      {loading && <div className="flex items-center gap-sm mt-md"><div className="spinner" /> Processing...</div>}

      {data && (
        <>
          {info && (
            <div className="card section">
              <div className="flex items-center gap-md" style={{ justifyContent: 'flex-end', marginBottom: 12 }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => {
                    setData(null);
                    setInfo(null);
                    setStats(null);
                    setVisualData(null);
                    setVisualColumn('');
                    setAnalysisColumn('');
                    setAnalysisData(null);
                    setAnalysisBins(10);
                    setCategoricalSummary(null);
                    setCorrelationData(null);
                    setJointData(null);
                    setJointColumnX('');
                    setJointColumnY('');
                    setScatterColumns([]);
                    setScatterData(null);
                    setGroupValueColumn('');
                    setGroupByColumn('');
                    setGroupedData(null);
                    setGroupChartType('box');
                  }}
                  style={{ marginTop: 18 }}
                >
                  📤 Upload New Data
                </button>
              </div>
            </div>
          )}

          {info && (
            <div className="card-grid section">
              <div className="stat-card">
                <div className="stat-icon" style={{ background: 'var(--info-bg)', color: 'var(--info)' }}><FileSpreadsheet size={20} /></div>
                <div><div className="stat-value">{info.shape?.[0]?.toLocaleString()}</div><div className="stat-label">Rows</div></div>
              </div>
              <div className="stat-card">
                <div className="stat-icon" style={{ background: 'var(--accent-glow)', color: 'var(--accent)' }}><FileSpreadsheet size={20} /></div>
                <div><div className="stat-value">{info.shape?.[1]}</div><div className="stat-label">Columns</div></div>
              </div>
              <div className="stat-card">
                <div className="stat-icon" style={{ background: 'var(--warning-bg)', color: 'var(--warning)' }}><AlertCircle size={20} /></div>
                <div><div className="stat-value">{Object.keys(info.missing || {}).length}</div><div className="stat-label">Missing Columns</div></div>
              </div>
              <div className="stat-card">
                <div className="stat-icon" style={{ background: 'var(--success-bg)', color: 'var(--success)' }}><FileSpreadsheet size={20} /></div>
                <div><div className="stat-value">{info.memory_mb} MB</div><div className="stat-label">Memory</div></div>
              </div>
            </div>
          )}

          <div className="tabs">
            {['preview', 'descriptive', 'distribution-detail', 'binned-categorical', 'correlation', 'scatter-matrix', 'joint-distribution', 'grouped-analysis', 'missing', 'visualization', 'ai-analysis'].map((t) => (
              <button key={t} className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
                {t === 'ai-analysis'
                  ? '✨ AI Analysis'
                  : t === 'distribution-detail'
                    ? 'Distribution Detail'
                    : t === 'binned-categorical'
                      ? 'Binned & Categorical'
                      : t === 'scatter-matrix'
                        ? 'Scatter Matrix'
                        : t === 'grouped-analysis'
                          ? 'Grouped Analysis'
                      : t === 'joint-distribution'
                        ? '2-Variable Distribution'
                      : t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>

          {tab === 'preview' && data && (
            <div className="table-container">
              <table>
                <thead><tr>{data.columns?.map((c) => <th key={c}>{c}</th>)}</tr></thead>
                <tbody>
                  {data.data?.slice(0, 50).map((row, i) => (
                    <tr key={i}>{data.columns?.map((c) => <td key={c}>{row[c] != null ? String(row[c]).substring(0, 30) : '—'}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {tab === 'descriptive' && (
            <DescriptiveStatsPanel
              descriptiveRows={descriptiveRows}
              numericColumns={numericColumns}
              analysisColumn={analysisColumn}
              onAnalysisColumnChange={setAnalysisColumn}
              analysisSummary={analysisSummary}
              onExportCsv={handleExportDescriptiveCsv}
            />
          )}

          {tab === 'distribution-detail' && (
            <DistributionDetailPanel
              analysisColumn={analysisColumn}
              numericColumns={numericColumns}
              onAnalysisColumnChange={setAnalysisColumn}
              analysisLoading={analysisLoading}
              analysisData={analysisData}
            />
          )}

          {tab === 'binned-categorical' && (
            <BinnedCategoricalPanel
              analysisColumn={analysisColumn}
              numericColumns={numericColumns}
              onAnalysisColumnChange={setAnalysisColumn}
              analysisBins={analysisBins}
              onAnalysisBinsChange={setAnalysisBins}
              binnedRows={binnedRows}
              categoricalSummary={categoricalSummary}
              catLoading={catLoading}
            />
          )}

          {tab === 'correlation' && (
            <div className="card">
              <div className="card-title" style={{ marginBottom: 6 }}>📈 Distribution & Correlation Analysis</div>
              <div className="chart-subtitle" style={{ marginBottom: 10 }}>Choose analysis type:</div>
              <div className="analysis-mode-group">
                <label className="analysis-mode-option active">
                  <input type="radio" name="corr-dist-mode" checked readOnly />
                  Correlation Heatmap
                </label>
                <label className="analysis-mode-option">
                  <input type="radio" name="corr-dist-mode" checked={false} onChange={() => setTab('scatter-matrix')} />
                  Scatter Plot Matrix
                </label>
                <label className="analysis-mode-option">
                  <input
                    type="radio"
                    name="corr-dist-mode"
                    checked={false}
                    onChange={() => setTab('joint-distribution')}
                  />
                  Scatter Plot (2 Variables)
                </label>
                <label className="analysis-mode-option">
                  <input type="radio" name="corr-dist-mode" checked={false} onChange={() => setTab('grouped-analysis')} />
                  Grouped Analysis
                </label>
              </div>

              <div className="chart-header" style={{ marginTop: 16 }}>
                <div>
                  <div className="card-title">🔥 Correlation Matrix</div>
                  <div className="chart-subtitle">Correlation matrix between variables</div>
                </div>
                <div className="chart-controls">
                  <div className="form-group" style={{ margin: 0, minWidth: 180 }}>
                    <label className="form-label">Method</label>
                    <select className="form-select" value={corrMethod} onChange={(e) => setCorrMethod(e.target.value)}>
                      <option value="pearson">Pearson</option>
                      <option value="spearman">Spearman</option>
                      <option value="kendall">Kendall</option>
                    </select>
                  </div>
                  <button className="btn btn-secondary" onClick={loadCorrelation} disabled={correlationLoading}>
                    Refresh
                  </button>
                </div>
              </div>

              {correlationLoading && <div className="flex items-center gap-sm"><div className="spinner" /> Loading correlation matrix...</div>}

              {!correlationLoading && correlationData && (
                <>
                  <div className="table-container corr-heatmap-table" style={{ marginBottom: 10 }}>
                    <table>
                      <thead>
                        <tr>
                          <th>Feature</th>
                          {(correlationData.columns || []).map((col) => <th key={col}>{col}</th>)}
                        </tr>
                      </thead>
                      <tbody>
                        {(correlationData.matrix || []).map((row) => (
                          <tr key={row.column}>
                            <td style={{ fontWeight: 700 }}>{row.column}</td>
                            {(row.values || []).map((v, idx) => (
                              <td key={`${row.column}-${idx}`} style={correlationCellStyle(v)}>
                                {Number(v).toFixed(2)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="corr-legend">
                    <span>-1</span>
                    <div className="corr-legend-bar" />
                    <span>1</span>
                  </div>

                  <div className="card" style={{ marginTop: 16 }}>
                    <div className="card-title" style={{ marginBottom: 10 }}>🔍 Highly Correlated Pairs</div>
                  <div className="threshold-row">
                      <label className="form-label" style={{ margin: 0 }}>Correlation Threshold:</label>
                      <input
                        className="threshold-slider"
                        type="range"
                        min="0.5"
                        max="0.99"
                        step="0.01"
                        value={corrThreshold}
                        onChange={(e) => setCorrThreshold(Number(e.target.value))}
                      />
                      <span className="threshold-value">{Number(corrThreshold).toFixed(2)}</span>
                      <button className="btn btn-secondary btn-sm" onClick={loadCorrelation} disabled={correlationLoading}>Apply</button>
                    </div>
                    {!(correlationData.high_pairs || []).length ? (
                      <div className="empty-state">No high-correlation pairs at this threshold.</div>
                    ) : (
                      <div className="table-container">
                        <table>
                          <thead><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th><th>Type</th></tr></thead>
                          <tbody>
                            {(correlationData.high_pairs || []).map((p, idx) => (
                              <tr key={`${p.feature_1}-${p.feature_2}-${idx}`}>
                                <td>{p.feature_1}</td>
                                <td>{p.feature_2}</td>
                                <td>{Number(p.correlation).toFixed(3)}</td>
                                <td>{p.correlation >= 0 ? 'Positive' : 'Negative'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}

          {tab === 'joint-distribution' && (
            <div className="card">
              <div className="card-title" style={{ marginBottom: 6 }}>📈 Distribution & Correlation Analysis</div>
              <div className="chart-subtitle" style={{ marginBottom: 10 }}>Choose analysis type:</div>
              <div className="analysis-mode-group">
                <label className="analysis-mode-option">
                  <input
                    type="radio"
                    name="joint-dist-mode"
                    checked={false}
                    onChange={() => setTab('correlation')}
                  />
                  Correlation Heatmap
                </label>
                <label className="analysis-mode-option">
                  <input type="radio" name="joint-dist-mode" checked={false} onChange={() => setTab('scatter-matrix')} />
                  Scatter Plot Matrix
                </label>
                <label className="analysis-mode-option active">
                  <input type="radio" name="joint-dist-mode" checked readOnly />
                  Scatter Plot (2 Variables)
                </label>
                <label className="analysis-mode-option">
                  <input type="radio" name="joint-dist-mode" checked={false} onChange={() => setTab('grouped-analysis')} />
                  Grouped Analysis
                </label>
              </div>

              <div className="chart-header">
                <div>
                  <div className="card-title">🧭 Two-Variable Distribution</div>
                  <div className="chart-subtitle">Cross-tab heatmap + distribution table for any two columns.</div>
                </div>
                <div className="chart-controls">
                  <div className="form-group" style={{ margin: 0, minWidth: 220 }}>
                    <label className="form-label">Column X</label>
                    <select className="form-select" value={jointColumnX} onChange={(e) => setJointColumnX(e.target.value)}>
                      {allColumns.map((c) => <option key={`x-${c}`} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div className="form-group" style={{ margin: 0, minWidth: 220 }}>
                    <label className="form-label">Column Y</label>
                    <select className="form-select" value={jointColumnY} onChange={(e) => setJointColumnY(e.target.value)}>
                      {allColumns.map((c) => <option key={`y-${c}`} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div className="form-group" style={{ margin: 0, width: 140 }}>
                    <label className="form-label">Bins</label>
                    <input className="form-input" type="number" min="3" max="20" value={jointBins} onChange={(e) => setJointBins(+e.target.value)} />
                  </div>
                  <button
                    className="btn btn-secondary"
                    onClick={() => loadJointDistribution(jointColumnX, jointColumnY, jointBins)}
                    disabled={jointLoading || !jointColumnX || !jointColumnY}
                  >
                    Refresh
                  </button>
                </div>
              </div>

              {jointLoading && <div className="flex items-center gap-sm"><div className="spinner" /> Loading 2-variable distribution...</div>}

              {!jointLoading && jointData && (
                <>
                  <div className="summary-grid" style={{ marginBottom: 12 }}>
                    <div className="summary-card"><div className="summary-label">Column X Type</div><div className="summary-value" style={{ fontSize: 14 }}>{jointData.x_type}</div></div>
                    <div className="summary-card"><div className="summary-label">Column Y Type</div><div className="summary-value" style={{ fontSize: 14 }}>{jointData.y_type}</div></div>
                    <div className="summary-card"><div className="summary-label">Total Pairs</div><div className="summary-value">{jointData.total_count}</div></div>
                    <div className="summary-card"><div className="summary-label">Bins Used</div><div className="summary-value">{jointData.bins_used}</div></div>
                  </div>

                  <div className="table-container corr-heatmap-table" style={{ marginBottom: 14 }}>
                    <table>
                      <thead>
                        <tr>
                          <th>{jointData.column_x} \\ {jointData.column_y}</th>
                          {(jointData.y_labels || []).map((y) => <th key={`yh-${y}`}>{y}</th>)}
                          <th>Total</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(jointData.rows || []).map((r) => (
                          <tr key={`xr-${r.x_value}`}>
                            <td style={{ fontWeight: 700 }}>{r.x_value}</td>
                            {(r.counts || []).map((count, idx) => (
                              <td key={`${r.x_value}-${idx}`} style={jointCellStyle(count, jointMaxCount)}>{count}</td>
                            ))}
                            <td>{r.total}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="card">
                    <div className="card-title" style={{ marginBottom: 8 }}>Flattened Distribution</div>
                    <div className="table-container">
                      <table>
                        <thead><tr><th>{jointData.column_x}</th><th>{jointData.column_y}</th><th>Count</th><th>Row Ratio (%)</th></tr></thead>
                        <tbody>
                          {jointTableRows.map((row, idx) => (
                            <tr key={`${row.x}-${row.y}-${idx}`}>
                              <td>{row.x}</td>
                              <td>{row.y}</td>
                              <td>{row.count}</td>
                              <td>{Number(row.ratio).toFixed(2)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}

          {tab === 'scatter-matrix' && (
            <div className="card">
              <div className="card-title" style={{ marginBottom: 6 }}>📈 Distribution & Correlation Analysis</div>
              <div className="chart-subtitle" style={{ marginBottom: 10 }}>Choose analysis type:</div>
              <div className="analysis-mode-group">
                <label className="analysis-mode-option">
                  <input type="radio" name="scatter-mode" checked={false} onChange={() => setTab('correlation')} />
                  Correlation Heatmap
                </label>
                <label className="analysis-mode-option active">
                  <input type="radio" name="scatter-mode" checked readOnly />
                  Scatter Plot Matrix
                </label>
                <label className="analysis-mode-option">
                  <input type="radio" name="scatter-mode" checked={false} onChange={() => setTab('joint-distribution')} />
                  Scatter Plot (2 Variables)
                </label>
                <label className="analysis-mode-option">
                  <input type="radio" name="scatter-mode" checked={false} onChange={() => setTab('grouped-analysis')} />
                  Grouped Analysis
                </label>
              </div>

              <div className="chart-header" style={{ marginTop: 14 }}>
                <div>
                  <div className="card-title">🔷 Scatter Plot Matrix (Pair Plot)</div>
                  <div className="chart-subtitle">Show pairwise relationships between selected numeric variables.</div>
                </div>
                <div className="chart-controls">
                  <div className="form-group" style={{ margin: 0, width: 180 }}>
                    <label className="form-label">Max Points</label>
                    <input
                      className="form-input"
                      type="number"
                      min="200"
                      max="5000"
                      value={scatterMaxPoints}
                      onChange={(e) => setScatterMaxPoints(Math.max(200, Math.min(5000, Number(e.target.value) || 200)))}
                    />
                  </div>
                  <button
                    className="btn btn-secondary"
                    disabled={scatterLoading || (scatterColumns || []).length < 2}
                    onClick={() => loadScatterMatrix(scatterColumns)}
                  >
                    Refresh
                  </button>
                </div>
              </div>

              <div style={{ marginTop: 8 }}>
                <ColumnMultiSelector
                  label="Columns To Show (max 5)"
                  columns={numericColumns}
                  values={scatterColumns}
                  onChange={(cols) => setScatterColumns(cols.slice(0, 5))}
                  placeholder="Type or choose numeric columns"
                  showSelectAll={numericColumns.length > 0}
                />
              </div>

              {(scatterColumns || []).length < 2 && (
                <div className="alert alert-info">Select at least 2 numeric columns to render the scatter matrix.</div>
              )}

              {scatterLoading && <div className="flex items-center gap-sm"><div className="spinner" /> Loading scatter matrix...</div>}

              {!scatterLoading && scatterData && (
                <>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10 }}>
                    Displaying {scatterData.sampled_rows?.toLocaleString()} sampled rows from {scatterData.total_rows?.toLocaleString()} valid rows.
                  </div>

                  {(scatterPairRows || []).map((pairRow, rowIdx) => {
                    const matrixCols = Math.max(1, (scatterData.columns?.length || 2) - 1);
                    return (
                      <div
                        key={`pair-row-${rowIdx}`}
                        className="pair-matrix-row"
                        style={{ gridTemplateColumns: `repeat(${matrixCols}, minmax(220px, 1fr))` }}
                      >
                        {pairRow.map((pair) => (
                          <div className="pair-cell" key={`${pair.xKey}-${pair.yKey}`}>
                            <div className="pair-cell-title">{pair.yKey} vs {pair.xKey}</div>
                            <ResponsiveContainer width="100%" height={190}>
                              <ScatterChart margin={{ top: 8, right: 10, left: 0, bottom: 16 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                                <XAxis type="number" dataKey={pair.xKey} tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                                <YAxis type="number" dataKey={pair.yKey} width={50} tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                                <Tooltip
                                  contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 10 }}
                                  formatter={(value, name) => [formatNum(value, 3), name]}
                                />
                                <Scatter data={scatterData.rows || []} fill="rgba(96,165,250,0.78)" />
                              </ScatterChart>
                            </ResponsiveContainer>
                          </div>
                        ))}
                        {Array.from({ length: Math.max(0, Math.max(1, (scatterData.columns?.length || 2) - 1) - pairRow.length) }).map((_, idx) => (
                          <div className="pair-cell-empty" key={`pair-empty-${rowIdx}-${idx}`} />
                        ))}
                      </div>
                    );
                  })}

                  <div className="alert alert-info" style={{ marginTop: 14 }}>
                    Tip: Look for linear or non-linear trends between feature pairs.
                  </div>
                </>
              )}
            </div>
          )}

          {tab === 'grouped-analysis' && (
            <div className="card">
              <div className="card-title" style={{ marginBottom: 6 }}>📈 Distribution & Correlation Analysis</div>
              <div className="chart-subtitle" style={{ marginBottom: 10 }}>Choose analysis type:</div>
              <div className="analysis-mode-group">
                <label className="analysis-mode-option">
                  <input type="radio" name="group-mode" checked={false} onChange={() => setTab('correlation')} />
                  Correlation Heatmap
                </label>
                <label className="analysis-mode-option">
                  <input type="radio" name="group-mode" checked={false} onChange={() => setTab('scatter-matrix')} />
                  Scatter Plot Matrix
                </label>
                <label className="analysis-mode-option">
                  <input type="radio" name="group-mode" checked={false} onChange={() => setTab('joint-distribution')} />
                  Scatter Plot (2 Variables)
                </label>
                <label className="analysis-mode-option active">
                  <input type="radio" name="group-mode" checked readOnly />
                  Grouped Analysis
                </label>
              </div>

              <div className="chart-header">
                <div>
                  <div className="card-title">📦 Grouped Analysis</div>
                  <div className="chart-subtitle">Compare numeric distributions across groups.</div>
                </div>
                <div className="chart-controls">
                  <div className="form-group" style={{ margin: 0, width: 150 }}>
                    <label className="form-label">Top Groups</label>
                    <input
                      className="form-input"
                      type="number"
                      min="2"
                      max="30"
                      value={groupTopN}
                      onChange={(e) => setGroupTopN(Math.max(2, Math.min(30, Number(e.target.value) || 2)))}
                    />
                  </div>
                  <button
                    className="btn btn-secondary"
                    onClick={() => loadGroupedAnalysis(groupValueColumn, groupByColumn)}
                    disabled={groupedLoading || !groupValueColumn || !groupByColumn}
                  >
                    Refresh
                  </button>
                </div>
              </div>

              <div className="analysis-grid" style={{ marginBottom: 12 }}>
                <div className="form-group" style={{ margin: 0 }}>
                  <label className="form-label">Numeric Column</label>
                  <select className="form-select" value={groupValueColumn} onChange={(e) => setGroupValueColumn(e.target.value)}>
                    {numericColumns.map((c) => <option key={`gv-${c}`} value={c}>{c}</option>)}
                  </select>
                </div>
                <div className="form-group" style={{ margin: 0 }}>
                  <label className="form-label">Group By Column</label>
                  <select className="form-select" value={groupByColumn} onChange={(e) => setGroupByColumn(e.target.value)}>
                    {allColumns
                      .filter((c) => c !== groupValueColumn)
                      .map((c) => <option key={`gb-${c}`} value={c}>{c}</option>)}
                  </select>
                </div>
              </div>

              <div className="analysis-mode-group" style={{ marginBottom: 14 }}>
                {['box', 'violin', 'strip'].map((m) => (
                  <label key={m} className={`analysis-mode-option ${groupChartType === m ? 'active' : ''}`}>
                    <input type="radio" name="group-plot-type" checked={groupChartType === m} onChange={() => setGroupChartType(m)} />
                    {m === 'box' ? 'Box Plot' : m === 'violin' ? 'Violin Plot' : 'Strip Plot'}
                  </label>
                ))}
              </div>

              {groupedLoading && <div className="flex items-center gap-sm"><div className="spinner" /> Loading grouped analysis...</div>}

              {!groupedLoading && groupedData && (
                <>
                  {groupedData.warning && (
                    <div className="alert alert-warning">{groupedData.warning}</div>
                  )}

                  <div className="card" style={{ marginBottom: 16 }}>
                    <div className="card-title" style={{ marginBottom: 8 }}>
                      {groupChartType === 'box' ? 'Box Plot' : groupChartType === 'violin' ? 'Violin Plot' : 'Strip Plot'} of {groupedData.value_column} by {groupedData.group_column}
                    </div>
                    <div className="chart-surface" style={{ marginBottom: 0 }}>
                      <GroupedDistributionPlot type={groupChartType} groupedData={groupedData} />
                    </div>
                  </div>

                  <div className="card">
                    <div className="card-title" style={{ marginBottom: 8 }}>📊 Group Statistics</div>
                    <div className="table-container">
                      <table>
                        <thead><tr><th>{groupedData.group_column}</th><th>Count</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Q1</th><th>Q3</th><th>Max</th></tr></thead>
                        <tbody>
                          {(groupedData.stats || []).map((row) => (
                            <tr key={`gr-${row.group}`}>
                              <td>{row.group}</td>
                              <td>{row.count}</td>
                              <td>{formatNum(row.mean, 3)}</td>
                              <td>{formatNum(row.median, 3)}</td>
                              <td>{row.std == null ? '-' : formatNum(row.std, 3)}</td>
                              <td>{formatNum(row.min, 3)}</td>
                              <td>{formatNum(row.q1, 3)}</td>
                              <td>{formatNum(row.q3, 3)}</td>
                              <td>{formatNum(row.max, 3)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}

          {tab === 'missing' && info?.missing && (
            <div className="card">
              {Object.keys(info.missing).length === 0 ? (
                <div className="empty-state">No missing values</div>
              ) : (
                <div className="table-container">
                  <table>
                    <thead><tr><th>Column</th><th>Count</th><th>Percentage</th></tr></thead>
                    <tbody>
                      {Object.entries(info.missing).map(([col, m]) => (
                        <tr key={col}>
                          <td>{col}</td><td>{m.count}</td>
                          <td><span className={m.percentage > 30 ? 'badge badge-error' : 'badge badge-warning'}>{m.percentage}%</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {tab === 'visualization' && (
            <DataDistributionCard
              title="Column Visualization"
              description="Inspect the full uploaded dataset by column, not just the preview rows."
              columns={visualColumns}
              selectedColumn={visualColumn}
              onColumnChange={setVisualColumn}
              distribution={visualData}
              loading={visualLoading}
            />
          )}

          {tab === 'ai-analysis' && (
            <div className="card">
              {!aiAnalysis && (
                <div className="empty-state">
                  <Sparkles size={48} />
                  <p>Run AI-powered analysis of your dataset using Gemini</p>
                  <button className="btn btn-primary mt-md" onClick={runAiAnalysis} disabled={aiLoading}>
                    {aiLoading ? <><div className="spinner" /> Analyzing...</> : <><Sparkles size={16} /> Analyze with AI</>}
                  </button>
                </div>
              )}
              {aiAnalysis && (
                <div style={{ whiteSpace: 'pre-wrap', fontSize: 14, lineHeight: 1.8 }}>{aiAnalysis}</div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
