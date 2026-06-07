import { useState, useEffect } from 'react';
import { dataApi } from '../api/client';
import { isViewOnly } from '../utils/permissions';
import { useAuth } from '../context/AuthContext';
import ColumnMultiSelector from '../components/ColumnMultiSelector';
import DataDistributionCard from '../components/charts/DataDistributionCard';
import ComparisonMetricChart from '../components/charts/ComparisonMetricChart';
import MetricRankingChart from '../components/charts/MetricRankingChart';
import WoeBinsChart from '../components/charts/WoeBinsChart';
import { BoxOutlierPlot } from '../components/charts/DistributionDetailPanel';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';

const STEP_CONFIG = [
  { key: 'split', num: 1, title: 'Train / Valid / Test Split', desc: 'Split data before any preprocessing to prevent data leakage' },
  { key: 'cleanup', num: 2, title: 'Data Cleanup (Pre-Missing)', desc: 'After split: remove categorical features and clean invalid numeric values' },
  { key: 'missing', num: 3, title: 'Handle Missing Values', desc: 'Impute or drop missing values (fit on train, transform all)' },
  { key: 'outliers', num: 4, title: 'Outlier Handling', desc: 'Detect and handle outliers in numeric columns' },
  { key: 'skewness', num: 5, title: 'Distribution Transform', desc: 'Reduce skewness with Log, Sqrt, Box-Cox, or Yeo-Johnson transforms' },
  { key: 'encoding', num: 6, title: 'Categorical Encoding', desc: 'Convert categorical variables to numeric representations' },
  { key: 'binning', num: 7, title: 'WoE / IV Binning', desc: 'Optimal binning — transforms columns to WoE values' },
  { key: 'woe_analysis', num: 8, title: 'WoE Analysis', desc: 'Evaluate feature predictive power using Information Value (read-only analysis)' },
  { key: 'multicollinearity', num: 9, title: 'Multicollinearity Detection', desc: 'Detect and remove highly correlated features using VIF' },
  { key: 'scaling', num: 10, title: 'Feature Scaling', desc: 'Normalize numeric features to comparable ranges' },
  { key: 'balance', num: 11, title: 'Class Balancing', desc: 'Balance class distribution (applied on training set only)' },
  { key: 'importance', num: 12, title: 'Feature Importance', desc: 'Evaluate and select the most impactful features' },
];

export default function FeatureEngineeringPage() {
  const { user } = useAuth();
  const viewOnly = isViewOnly(user?.role, '/feature-engineering');
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionInfo, setSessionInfo] = useState(null);
  const [dataInfo, setDataInfo] = useState(null);
  const [completedSteps, setCompletedSteps] = useState(new Set());
  const [importanceResults, setImportanceResults] = useState(null);
  const [binningResults, setBinningResults] = useState(null);
  const [skewnessResults, setSkewnessResults] = useState(null);
  const [woeResults, setWoeResults] = useState(null);
  const [vifResults, setVifResults] = useState(null);
  const [corrPairs, setCorrPairs] = useState(null);
  const [autoRemoveVif, setAutoRemoveVif] = useState(false);
  const [restoreColsInput, setRestoreColsInput] = useState('');
  const [distributionColumn, setDistributionColumn] = useState('');
  const [distributionData, setDistributionData] = useState(null);
  const [distributionLoading, setDistributionLoading] = useState(false);
  const [selectedBinningFeature, setSelectedBinningFeature] = useState('');
  const [selectedWoeFeature, setSelectedWoeFeature] = useState('');
  const [outlierResults, setOutlierResults] = useState(null);

  const [testSize, setTestSize] = useState(0.15);
  const [validSize, setValidSize] = useState(0.15);
  const [target, setTarget] = useState('');
  const [randomSeed, setRandomSeed] = useState(42);
  const [splitStratify, setSplitStratify] = useState(true);

  const [cleanupStrategy, setCleanupStrategy] = useState('drop_rows');
  const [cleanupCols, setCleanupCols] = useState([]);
  const [invalidCols, setInvalidCols] = useState([]);

  const [missingMethod, setMissingMethod] = useState('Mean Imputation');
  const [missingCols, setMissingCols] = useState([]);
  const [missingConstantValue, setMissingConstantValue] = useState('');

  const [outlierMethod, setOutlierMethod] = useState('IQR Method');
  const [outlierCols, setOutlierCols] = useState([]);
  const [iqrMultiplier, setIqrMultiplier] = useState(1.5);
  const [iqrAction, setIqrAction] = useState('clip');
  const [zThreshold, setZThreshold] = useState(3.0);
  const [zAction, setZAction] = useState('clip');
  const [winsorLower, setWinsorLower] = useState(0.05);
  const [winsorUpper, setWinsorUpper] = useState(0.95);
  const [previewOutlierCol, setPreviewOutlierCol] = useState('');

  const [skewMethod, setSkewMethod] = useState('Log');
  const [skewCols, setSkewCols] = useState([]);
  const [skewTransformCol, setSkewTransformCol] = useState('');

  const [encMethod, setEncMethod] = useState('Label Encoding');
  const [encCols, setEncCols] = useState([]);
  const [catSummary, setCatSummary] = useState([]);
  const [encCol, setEncCol] = useState('');
  const [encDropFirst, setEncDropFirst] = useState(false);

  const [binningCols, setBinningCols] = useState([]);
  const [binningMaxBins, setBinningMaxBins] = useState(10);
  const [binningMethod, setBinningMethod] = useState('Optimal Binning (WoE/IV)');
  const [binningPreviewCol, setBinningPreviewCol] = useState('');
  const [binningMonotonic, setBinningMonotonic] = useState(true);
  const [binningNewColName, setBinningNewColName] = useState('');

  const [scaleMethod, setScaleMethod] = useState('StandardScaler');
  const [scaleCols, setScaleCols] = useState([]);
  const [scaleCreateNew, setScaleCreateNew] = useState(false);

  const [balanceMethod, setBalanceMethod] = useState('SMOTE');
  const [balStrategy, setBalStrategy] = useState('auto');
  const [balDist, setBalDist] = useState(null);
  const [balanceInfo, setBalanceInfo] = useState(null);

  const [impMethod, setImpMethod] = useState('Random Forest');
  const [impTopN, setImpTopN] = useState(15);
  const [impThreshold, setImpThreshold] = useState(0.01);
  const [selectedTrainCols, setSelectedTrainCols] = useState([]);

  const normalizeImportanceResults = (payload) => {
    if (!payload) return null;
    if (payload.importance && typeof payload.importance === 'object' && !Array.isArray(payload.importance)) {
      return payload.importance;
    }
    if (Array.isArray(payload.feature_names) && Array.isArray(payload.importance_scores)) {
      return payload.feature_names.reduce((acc, feature, idx) => {
        acc[feature] = Number(payload.importance_scores[idx] ?? 0);
        return acc;
      }, {});
    }
    return null;
  };

  // Load session state on mount
  useEffect(() => {
    loadSessionInfo();
  }, []);

  useEffect(() => {
    if (!numericCols.length) {
      setDistributionColumn('');
      setDistributionData(null);
      return;
    }
    if (!distributionColumn || !numericCols.includes(distributionColumn)) {
      setDistributionColumn(numericCols[0]);
    }
  }, [dataInfo, sessionInfo, distributionColumn]);

  useEffect(() => {
    if (!distributionColumn) return;
    loadDistribution(distributionColumn);
  }, [distributionColumn]);

  useEffect(() => {
    if (binningCols.length === 1) {
      setBinningNewColName(`${binningCols[0]}_woe`);
    } else {
      setBinningNewColName('');
    }
  }, [binningCols]);

  const loadSessionInfo = async () => {
    try {
      const res = await dataApi.getSessionInfo();
      setSessionInfo(res.data);
      setCompletedSteps(new Set(res.data.completed_steps || []));
      setBalanceInfo(res.data.balance_info || null);
      const initialSelected = (res.data.selected_features && res.data.selected_features.length > 0)
        ? res.data.selected_features
        : (res.data.split_feature_columns || []);
      setSelectedTrainCols(initialSelected);
      if (res.data.target_column) {
        setTarget(res.data.target_column);
      }
      if (res.data.has_data) {
        const infoRes = await dataApi.getInfo();
        setDataInfo(infoRes.data);
        const catRes = await dataApi.getCategoricalSummary(true).catch(() => ({ data: { columns: [] } }));
        setCatSummary(catRes.data?.columns || []);
      }
    } catch { /* ignore */ }
  };

  const loadDistribution = async (column) => {
    setDistributionLoading(true);
    try {
      const res = await dataApi.getDistribution(column, true, 24);
      setDistributionData(res.data);
    } catch {
      setDistributionData(null);
    } finally {
      setDistributionLoading(false);
    }
  };

  const run = async (stepKey, fn) => {
    setLoading(true);
    setMessage('');
    try {
      const res = await fn();
      setMessage(res.data.message || 'Done');
      setCompletedSteps(prev => new Set([...prev, stepKey]));
      await loadSessionInfo();
      return res.data;
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const selectedTarget = sessionInfo?.target_column || target || '';
    if (selectedTarget && sessionInfo?.has_splits) {
      dataApi.getDistribution(selectedTarget, true)
        .then(res => setBalDist(res.data))
        .catch(() => setBalDist(null));
      return;
    }
    setBalDist(null);
  }, [sessionInfo?.target_column, target, sessionInfo?.has_splits]);

  const noData = !sessionInfo?.has_data;
  const noSplits = !sessionInfo?.has_splits && !completedSteps.has('split');

  const splitNumericCols = sessionInfo?.split_numeric_columns || [];
  const splitCategoricalCols = sessionInfo?.split_categorical_columns || [];
  const splitMissingColsList = sessionInfo?.split_missing ? Object.keys(sessionInfo.split_missing) : [];
  const useSplitColumns = Boolean(sessionInfo?.has_splits);

  const categoricalCols = useSplitColumns ? splitCategoricalCols : (dataInfo?.categorical_columns || []);
  const numericCols = useSplitColumns ? splitNumericCols : (dataInfo?.numeric_columns || []);
  const missingColsList = useSplitColumns
    ? splitMissingColsList
    : (dataInfo?.missing ? Object.keys(dataInfo.missing) : []);

  const targetCol = sessionInfo?.target_column || target || '';
  const cleanupInfo = dataInfo;
  const cleanupCategoricalCols = cleanupInfo?.categorical_columns || [];
  const cleanupNumericCols = cleanupInfo?.numeric_columns || [];
  const cleanupTableCols = [...cleanupCategoricalCols, ...cleanupNumericCols];
  const cleanupColumnCount = cleanupTableCols.length;
  const cleanupTotalRows = cleanupInfo?.shape?.[0] || sessionInfo?.data_shape?.[0] || 1;
  const cleanupColumnsForRemove = useSplitColumns
    ? (sessionInfo?.split_feature_columns || []).filter((c) => c !== targetCol)
    : cleanupTableCols.filter((c) => c !== targetCol);
  const cleanupNumericForInvalid = useSplitColumns
    ? splitNumericCols.filter((c) => c !== targetCol)
    : cleanupNumericCols.filter((c) => c !== targetCol);
  const skewChartData = Object.entries(skewnessResults || {})
    .filter(([, value]) => !value.error)
    .map(([label, value]) => ({
      label,
      before: value.skew_before,
      after: value.skew_after,
    }));
  const outlierChartRows = Object.entries(outlierResults || {}).map(([label, value]) => ({
    label,
    count: value.n_outliers ?? value.outliers_detected ?? 0,
  }));
  const binningChartData = Object.entries(binningResults || {})
    .filter(([, value]) => !value.error)
    .sort(([, a], [, b]) => (b.iv || 0) - (a.iv || 0))
    .map(([label, value]) => ({
      label,
      value: value.iv,
    }));
  const selectedBinningBins = selectedBinningFeature && binningResults?.[selectedBinningFeature]?.table
    ? binningResults[selectedBinningFeature].table
        .filter((row) => row.Bin && row.Bin !== 'Totals')
        .map((row) => ({
          label: row.Bin,
          count: Number(row.Count || 0),
          woe: row.WoE == null ? null : Number(row.WoE),
        }))
    : [];
  const woeChartData = Object.entries(woeResults?.results || {})
    .sort(([, a], [, b]) => (b.iv || 0) - (a.iv || 0))
    .map(([label, value]) => ({
      label,
      value: value.iv,
    }));
  const selectedWoeBins = selectedWoeFeature && woeResults?.results?.[selectedWoeFeature]?.bins
    ? woeResults.results[selectedWoeFeature].bins.map((row) => ({
        label: row.bin,
        count: Number(row.count || 0),
        woe: row.woe == null ? null : Number(row.woe),
      }))
    : [];
  const importanceRankedRows = Object.entries(importanceResults || {})
    .map(([feature, score]) => ({
      feature,
      score: typeof score === 'number' ? score : Number(score) || 0,
    }))
    .sort((a, b) => b.score - a.score);
  const importanceMaxScore = Math.max(1e-9, ...importanceRankedRows.map((row) => row.score));
  const selectedByThreshold = importanceRankedRows.filter((row) => row.score >= impThreshold).map((row) => row.feature);

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Feature Engineering</h1>
        <p className="page-description">Preprocess your data step by step — split first, then transform to prevent data leakage</p>
      </div>

      {viewOnly && <div className="alert alert-info">You are in view-only mode (Validator role)</div>}
      {message && <div className={`alert ${message.startsWith('Error') ? 'alert-error' : 'alert-success'}`}>{message}</div>}
      {loading && <div className="flex items-center gap-sm mb-md"><div className="spinner" /> Processing...</div>}

      {noData && (
        <div className="card section">
          <div className="empty-state">
            <p style={{ fontSize: 16, marginBottom: 8 }}>⚠️ No data loaded</p>
            <p style={{ color: 'var(--text-muted)' }}>Go to <strong>Data Upload & EDA</strong> first to upload a CSV file and set a target column.</p>
          </div>
        </div>
      )}

      {!noData && STEP_CONFIG.map(({ key, num, title, desc }) => (
        <div className="card section" key={key} style={{ opacity: getStepOpacity(key, completedSteps, noSplits), transition: 'opacity 0.3s' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <div style={{
              width: 32, height: 32, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 14, fontWeight: 700,
              background: completedSteps.has(key) ? 'var(--success)' : 'var(--surface-hover)',
              color: completedSteps.has(key) ? '#fff' : 'var(--text-muted)',
            }}>
              {completedSteps.has(key) ? '✓' : num}
            </div>
            <div>
              <div className="card-title" style={{ margin: 0 }}>{title}</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{desc}</div>
            </div>
          </div>

          {/* Step 1: Split */}
          {key === 'split' && (
            <>
              <div className="cleanup-callout cleanup-callout-info" style={{ marginBottom: 16 }}>
                <div className="cleanup-callout-icon">💡</div>
                <div style={{ fontSize: 13, fontWeight: 600 }}>Important: All preprocessing steps (missing values, outliers, encoding...) will be fitted on the Train set, then transformed to the Valid and Test sets to prevent data leakage.</div>
              </div>

              <div className="cleanup-grid">
                {/* LEFT PANEL: Split Configuration */}
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">📊</span>
                    <span className="cleanup-panel-title">Split Configuration</span>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Choose target column:</label>
                    <select className="form-select" value={target} onChange={(e) => setTarget(e.target.value)} disabled={viewOnly}>
                      <option value="">Select target...</option>
                      {dataInfo?.columns?.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>

                  <button 
                    className="btn btn-secondary btn-sm" 
                    onClick={() => run('setTarget', () => dataApi.setTarget(target))} 
                    disabled={!target || viewOnly}
                    style={{ width: '100%', marginBottom: 16, justifyContent: 'center' }}
                  >
                    💾 Save Target
                  </button>

                  <div className="flex gap-md" style={{ marginBottom: 12 }}>
                    <div className="form-group" style={{ flex: 1 }}>
                      <label className="form-label">Train Ratio (%): {((1 - testSize - validSize) * 100).toFixed(0)}</label>
                      <input className="form-input" type="range" min="50" max="90" step="1" 
                        value={((1 - testSize - validSize) * 100).toFixed(0)} 
                        onChange={e => {
                          const tr = +e.target.value;
                          const remaining = 100 - tr;
                          const newV = Math.floor(remaining / 2) / 100;
                          const newT = (remaining - Math.floor(remaining / 2)) / 100;
                          setValidSize(newV);
                          setTestSize(newT);
                        }} disabled={viewOnly} style={{ accentColor: '#3b82f6' }} />
                    </div>
                    <div className="form-group" style={{ flex: 1 }}>
                      <label className="form-label">Valid Ratio (%): {(validSize * 100).toFixed(0)}</label>
                      <input className="form-input" type="range" min="5" max="30" step="1" 
                        value={(validSize * 100).toFixed(0)} 
                        onChange={e => {
                          const val = +e.target.value;
                          const currentTrPercentage = ((1 - testSize - validSize) * 100).toFixed(0);
                          const remaining = 100 - currentTrPercentage - val;
                          if (remaining >= 5) { // test size at least 5%
                            setValidSize(val / 100);
                            setTestSize(remaining / 100);
                          }
                        }} disabled={viewOnly} style={{ accentColor: '#ec4899' }} />
                    </div>
                  </div>

                  <div className="cleanup-badge" style={{ marginBottom: 16, justifyContent: 'center' }}>
                    <span className="cleanup-badge-dot" />
                    Split Ratio: Train {((1 - testSize - validSize) * 100).toFixed(0)}% | Valid {(validSize * 100).toFixed(0)}% | Test {(testSize * 100).toFixed(0)}%
                  </div>

                  <div className="form-group">
                    <label className="form-label">Random Seed:</label>
                    <input className="form-input" type="number" 
                      value={randomSeed} onChange={e => setRandomSeed(+e.target.value)} disabled={viewOnly} />
                  </div>

                  <div className="form-group" style={{ marginTop: 8 }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, fontWeight: 600 }}>
                      <input
                        type="checkbox"
                        checked={splitStratify}
                        onChange={(e) => setSplitStratify(e.target.checked)}
                        disabled={viewOnly}
                        style={{ accentColor: '#ef4444' }}
                      />
                      🎯 Stratify (keep target ratio)
                    </label>
                  </div>

                  <button className="btn btn-primary btn-sm" disabled={viewOnly || loading || !target}
                    onClick={() => run('split', () => dataApi.split({ target_column: target, test_size: testSize, valid_size: validSize, random_state: randomSeed, stratify: splitStratify }))}
                    style={{ width: '100%', justifyContent: 'center', marginTop: 8 }}>
                    ✂️ Split Data
                  </button>
                </div>

                {/* RIGHT PANEL: Split Status */}
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">📈</span>
                    <span className="cleanup-panel-title">Split Status</span>
                  </div>

                  {completedSteps.has('split') ? (
                    <div className="cleanup-callout cleanup-callout-success" style={{ marginBottom: 16 }}>
                      <div className="cleanup-callout-icon">✅</div>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 700 }}>Data has been split</div>
                        <ul style={{ margin: '6px 0 0 16px', fontSize: 13, lineHeight: 1.7 }}>
                          <li>Train: {sessionInfo?.split_sizes?.train ?? 0} rows</li>
                          <li>Valid: {sessionInfo?.split_sizes?.valid ?? 0} rows</li>
                          <li>Test: {sessionInfo?.split_sizes?.test ?? 0} rows</li>
                        </ul>
                      </div>
                    </div>
                  ) : (
                    <div className="cleanup-callout cleanup-callout-warning" style={{ marginBottom: 16 }}>
                      <div className="cleanup-callout-icon">⏳</div>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 700 }}>Not Split Yet</div>
                        <div style={{ fontSize: 12, marginTop: 4 }}>Current data: {dataInfo?.shape?.[0] || 'Unknown'} rows</div>
                        <div style={{ fontSize: 12, marginTop: 4 }}>Configure the split on the left side and press Split Data.</div>
                      </div>
                    </div>
                  )}

                  <div className="cleanup-callout cleanup-callout-info" style={{ marginTop: 8, background: 'var(--surface)' }}>
                    <div className="cleanup-callout-icon">📗</div>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>Benefits of splitting:</div>
                      <ul style={{ margin: '0 0 0 16px', fontSize: 12, lineHeight: 1.7, color: 'var(--text-secondary)' }}>
                        <li><strong>Train:</strong> Used to fit models and calculate statistics</li>
                        <li><strong>Valid:</strong> Evaluate model during training phase</li>
                        <li><strong>Test:</strong> Final evaluation, never seen by the model</li>
                        <li><strong>Avoid overfitting:</strong> Model has never seen Valid/Test data during training</li>
                        <li><strong>Avoid data leakage:</strong> Statistics strictly drawn from Train only</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Step 2: Cleanup */}
          {key === 'cleanup' && (
            <div className="cleanup-grid">
              {/* LEFT PANEL: Remove Identifier Variables */}
              <div className="cleanup-panel">
                <div className="cleanup-panel-header">
                  <span className="cleanup-panel-icon">🔍</span>
                  <span className="cleanup-panel-title">Remove Identifier Variables</span>
                </div>

                <div className="cleanup-callout cleanup-callout-info">
                  <div className="cleanup-callout-icon">💡</div>
                  <div>
                    <p style={{ margin: 0, fontSize: 13, fontWeight: 600 }}>Identifier variables can't be used to predict, should be removed from model:</p>
                    <ul style={{ margin: '6px 0 0 16px', fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                      <li>Customer ID (customer_id, user_id)</li>
                      <li>Contract ID (contract_id, loan_id)</li>
                      <li>ID card, account number</li>
                      <li>Other identifiers</li>
                    </ul>
                  </div>
                </div>

                <div className="cleanup-badge">
                  <span className="cleanup-badge-dot" />
                  Dataset currently has {cleanupColumnCount || '—'} columns
                </div>

                {/* Columns info table */}
                <div className="table-container" style={{ marginBottom: 14, maxHeight: 260, overflowY: 'auto' }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Column</th>
                        <th style={{ textAlign: 'right' }}>Number Of Unique Values</th>
                        <th style={{ textAlign: 'right' }}>Unique Rate (%)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cleanupTableCols.map((col) => {
                        const uniqueCount = cleanupInfo?.unique_counts?.[col] ?? '—';
                        const uniquePct = typeof uniqueCount === 'number'
                          ? ((uniqueCount / cleanupTotalRows) * 100).toFixed(1)
                          : '—';
                        return (
                          <tr key={col}>
                            <td style={{ fontWeight: 500 }}>{col}</td>
                            <td style={{ textAlign: 'right' }}>{uniqueCount}</td>
                            <td style={{ textAlign: 'right' }}>{uniquePct}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <ColumnMultiSelector
                  label="Choose columns to remove:"
                  columns={cleanupColumnsForRemove}
                  values={cleanupCols}
                  onChange={setCleanupCols}
                  disabled={viewOnly || noSplits}
                  placeholder="Choose options"
                  showSelectAll={false}
                />

                <button
                  className="btn btn-danger btn-sm"
                  style={{ width: '100%', marginTop: 8, justifyContent: 'center', gap: 6 }}
                  disabled={viewOnly || loading || noSplits || cleanupCols.length === 0}
                  onClick={() => run('cleanup', () => dataApi.removeCategorical({ columns: cleanupCols, processed: false, apply_on_splits: true }))}
                >
                  🗑️ Remove Selected Columns
                </button>
              </div>

              {/* RIGHT PANEL: Handle Invalid Values */}
              <div className="cleanup-panel">
                <div className="cleanup-panel-header">
                  <span className="cleanup-panel-icon" style={{ color: 'var(--warning)' }}>⚠️</span>
                  <span className="cleanup-panel-title">Handle Invalid Values</span>
                </div>

                <div className="cleanup-callout cleanup-callout-warning">
                  <div className="cleanup-callout-icon">💡</div>
                  <div>
                    <p style={{ margin: 0, fontSize: 13, fontWeight: 600 }}>Invalid Values Examples:</p>
                    <ul style={{ margin: '6px 0 0 16px', fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                      <li>Negative income → 0 or NA</li>
                      <li>Age &lt; 18 or &gt; 90 → threshold</li>
                      <li>Negative debt → 0</li>
                      <li>Term ≤ 0 → NA or min</li>
                    </ul>
                  </div>
                </div>

                <ColumnMultiSelector
                  label="Columns (optional):"
                  columns={cleanupNumericForInvalid}
                  values={invalidCols}
                  onChange={setInvalidCols}
                  disabled={viewOnly || noSplits}
                  placeholder="Choose options"
                  showSelectAll={cleanupNumericForInvalid.length > 0}
                />

                <div className="form-group" style={{ marginTop: 4 }}>
                  <label className="form-label">Strategy</label>
                  <select className="form-select" value={cleanupStrategy} onChange={e => setCleanupStrategy(e.target.value)} disabled={viewOnly || noSplits}>
                    <option value="drop_rows">Drop rows with invalid values</option>
                    <option value="fill_median">Convert invalid to NaN → fill train median</option>
                  </select>
                </div>

                <button
                  className="btn btn-primary btn-sm"
                  style={{ width: '100%', marginTop: 4, justifyContent: 'center' }}
                  disabled={viewOnly || loading || noSplits || invalidCols.length === 0}
                  onClick={() => run('cleanup', () => dataApi.cleanInvalidNumbers({
                    columns: invalidCols.length ? invalidCols : null,
                    strategy: cleanupStrategy,
                    processed: false,
                    apply_on_splits: true,
                  }))}
                >
                  ✅ Clean Invalid Numbers
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Missing Values */}
          {key === 'missing' && (
            <>
              <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
                <div className="form-group" style={{ flex: 1, minWidth: 200 }}>
                  <label className="form-label">Method</label>
                  <select className="form-select" value={missingMethod} onChange={e => setMissingMethod(e.target.value)} disabled={viewOnly || noSplits}>
                    {['Mean Imputation', 'Median Imputation', 'Mode Imputation', 'Constant Value', 'Drop Rows', 'Forward Fill', 'Backward Fill'].map(m =>
                      <option key={m} value={m}>{m}</option>
                    )}
                  </select>
                </div>
                {missingMethod === 'Constant Value' && (
                  <div className="form-group" style={{ flex: 1, minWidth: 200 }}>
                    <label className="form-label">Constant Value</label>
                    <input
                      className="form-input"
                      type="number"
                      value={missingConstantValue}
                      onChange={(e) => setMissingConstantValue(e.target.value)}
                      placeholder="Enter value (e.g. 0)"
                      disabled={viewOnly || noSplits}
                    />
                  </div>
                )}
                <ColumnMultiSelector
                  label="Columns"
                  columns={missingColsList}
                  values={missingCols}
                  onChange={setMissingCols}
                  disabled={viewOnly || noSplits}
                  placeholder="Type or choose columns with missing values"
                  showSelectAll={missingColsList.length > 0}
                />
              </div>
              <button className="btn btn-primary btn-sm" disabled={viewOnly || loading || noSplits || missingCols.length === 0 || (missingMethod === 'Constant Value' && missingConstantValue === '')}
                onClick={() => run('missing', () => dataApi.handleMissing({
                  method: missingMethod,
                  columns: missingCols,
                  constant_value: missingMethod === 'Constant Value' ? Number(missingConstantValue) : undefined,
                }))}>
                Handle Missing Values
              </button>
            </>
          )}

          {/* Step 4: Outliers */}
          {key === 'outliers' && (
            <>
              <div className="cleanup-grid">
                {/* LEFT PANEL: Outlier Configuration */}
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">⚙️</span>
                    <span className="cleanup-panel-title">Outlier Configuration</span>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Method</label>
                    <select className="form-select" value={outlierMethod} onChange={e => setOutlierMethod(e.target.value)} disabled={viewOnly}>
                      {['IQR Method', 'Z-Score', 'Winsorization', 'Keep All'].map(m =>
                        <option key={m} value={m}>{m}</option>
                      )}
                    </select>
                  </div>

                  {/* Method-specific parameters */}
                  {outlierMethod === 'IQR Method' && (
                    <>
                      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>IQR Configuration:</div>
                      <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
                        <div className="form-group" style={{ flex: 1, minWidth: 140 }}>
                          <label className="form-label">IQR Multiplier</label>
                          <input className="form-input" type="range" min="1.0" max="3.0" step="0.1"
                            value={iqrMultiplier} onChange={e => setIqrMultiplier(+e.target.value)} disabled={viewOnly}
                            style={{ accentColor: 'var(--error)' }} />
                          <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', textAlign: 'center', marginTop: 2 }}>{iqrMultiplier.toFixed(2)}</div>
                        </div>
                        <div className="form-group" style={{ flex: 1, minWidth: 140 }}>
                          <label className="form-label">Action</label>
                          <select className="form-select" value={iqrAction} onChange={e => setIqrAction(e.target.value)} disabled={viewOnly}>
                            <option value="clip">clip</option>
                            <option value="remove">remove</option>
                            <option value="nan">nan</option>
                          </select>
                        </div>
                      </div>
                    </>
                  )}

                  {outlierMethod === 'Z-Score' && (
                    <>
                      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Z-Score Configuration:</div>
                      <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
                        <div className="form-group" style={{ flex: 1, minWidth: 140 }}>
                          <label className="form-label">Z-Score Threshold</label>
                          <input className="form-input" type="range" min="2.0" max="4.0" step="0.1"
                            value={zThreshold} onChange={e => setZThreshold(+e.target.value)} disabled={viewOnly}
                            style={{ accentColor: 'var(--error)' }} />
                          <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', textAlign: 'center', marginTop: 2 }}>{zThreshold.toFixed(2)}</div>
                        </div>
                        <div className="form-group" style={{ flex: 1, minWidth: 140 }}>
                          <label className="form-label">Action</label>
                          <select className="form-select" value={zAction} onChange={e => setZAction(e.target.value)} disabled={viewOnly}>
                            <option value="clip">clip</option>
                            <option value="remove">remove</option>
                            <option value="nan">nan</option>
                          </select>
                        </div>
                      </div>
                    </>
                  )}

                  {outlierMethod === 'Winsorization' && (
                    <>
                      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Percentile Configuration:</div>
                      <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
                        <div className="form-group" style={{ flex: 1, minWidth: 140 }}>
                          <label className="form-label">Lower Percentile</label>
                          <input className="form-input" type="number" min="0" max="0.5" step="0.01"
                            value={winsorLower} onChange={e => setWinsorLower(+e.target.value)} disabled={viewOnly} />
                        </div>
                        <div className="form-group" style={{ flex: 1, minWidth: 140 }}>
                          <label className="form-label">Upper Percentile</label>
                          <input className="form-input" type="number" min="0.5" max="1.0" step="0.01"
                            value={winsorUpper} onChange={e => setWinsorUpper(+e.target.value)} disabled={viewOnly} />
                        </div>
                      </div>
                    </>
                  )}

                  <ColumnMultiSelector
                    label="Columns to handle"
                    columns={numericCols}
                    values={outlierCols}
                    onChange={setOutlierCols}
                    disabled={viewOnly}
                    placeholder="Choose options"
                    showSelectAll={numericCols.length > 0}
                  />

                  <button
                    className="btn btn-primary btn-sm"
                    style={{ width: '100%', marginTop: 4, justifyContent: 'center' }}
                    disabled={viewOnly || loading || outlierCols.length === 0}
                    onClick={async () => {
                      const params = { method: outlierMethod, columns: outlierCols };
                      if (outlierMethod === 'IQR Method') {
                        params.multiplier = iqrMultiplier;
                        params.action = iqrAction;
                      } else if (outlierMethod === 'Z-Score') {
                        params.threshold = zThreshold;
                        params.action = zAction;
                      } else if (outlierMethod === 'Winsorization') {
                        params.lower_percentile = winsorLower;
                        params.upper_percentile = winsorUpper;
                      }
                      const data = await run('outliers', () => dataApi.handleOutliers(params));
                      if (data?.info) setOutlierResults(data.info);
                      if (distributionColumn) await loadDistribution(distributionColumn);
                    }}
                  >
                    ✅ Apply Outlier Handling
                  </button>
                </div>

                {/* RIGHT PANEL: Preview Outliers */}
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">📊</span>
                    <span className="cleanup-panel-title">Preview Outliers</span>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Column to preview</label>
                    <select className="form-select" value={previewOutlierCol || (numericCols[0] || '')} onChange={e => { setPreviewOutlierCol(e.target.value); loadDistribution(e.target.value); }}>
                      {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>

                  {(() => {
                    const col = previewOutlierCol || numericCols[0] || '';
                    const outlierCount = distributionData?.summary?.outliers_count ?? 0;
                    const totalCount = distributionData?.summary?.count ?? sessionInfo?.split_shape?.[0] ?? 0;
                    const pct = totalCount > 0 ? ((outlierCount / totalCount) * 100).toFixed(2) : '0.00';
                    return (
                      <>
                        <div className="flex gap-md" style={{ marginBottom: 16 }}>
                          <div className="outlier-metric-card">
                            <div className="outlier-metric-label">Outlier Count</div>
                            <div className="outlier-metric-value">{outlierCount}</div>
                          </div>
                          <div className="outlier-metric-card">
                            <div className="outlier-metric-label">Percentage</div>
                            <div className="outlier-metric-value">{pct}%</div>
                          </div>
                        </div>

                        {distributionLoading && (
                          <div className="flex items-center gap-sm" style={{ padding: 16 }}>
                            <div className="spinner" /> Loading chart...
                          </div>
                        )}

                        {!distributionLoading && distributionData && (
                          <>
                            <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6 }}>Box / IQR + Outliers</div>
                            <BoxOutlierPlot
                              summary={distributionData.box_stats || distributionData.summary}
                              outlierValues={distributionData.outlier_values}
                            />
                            <div style={{ marginTop: 4, fontSize: 11, color: 'var(--text-muted)' }}>
                              Red dots are outliers (outside IQR bounds). Dashed lines are lower/upper IQR bounds.
                            </div>
                          </>
                        )}
                      </>
                    );
                  })()}
                </div>
              </div>

              {/* Results below */}
              {outlierChartRows.length > 0 && (
                <div style={{ marginTop: 20 }}>
                  <MetricRankingChart
                    title="Detected Outliers by Column"
                    data={outlierChartRows}
                    valueKey="count"
                    color="#f59e0b"
                  />
                </div>
              )}
            </>
          )}

          {/* Step 5: Skewness / Distribution Transform */}
          {key === 'skewness' && (() => {
            const METHOD_DESC = {
              'Log': 'Reduces positive skew, requires values > 0',
              'Sqrt': 'Milder than Log for reducing positive skew',
              'Box-Cox': 'Finds optimal λ automatically, requires values > 0',
              'Yeo-Johnson': 'Like Box-Cox but handles negative values',
              'Reciprocal': 'For strongly right-skewed distributions',
            };
            const selCol = skewTransformCol || numericCols[0] || '';
            const skewVal = distributionData?.column === selCol ? distributionData?.summary?.skewness : null;
            const skewAbs = skewVal != null ? Math.abs(skewVal) : null;
            let skewBadge = { label: '—', cls: '' };
            if (skewAbs != null) {
              if (skewAbs < 0.5) skewBadge = { label: '✅ Near normal', cls: 'skew-badge-good' };
              else if (skewAbs < 1.0) skewBadge = { label: '⚠️ Moderate skew', cls: 'skew-badge-warn' };
              else skewBadge = { label: '❌ Highly skewed', cls: 'skew-badge-bad' };
            }
            const previewCol = distributionColumn || numericCols[0] || '';
            return (
            <>
              <div className="cleanup-grid">
                {/* LEFT PANEL: Transform Configuration */}
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">⚙️</span>
                    <span className="cleanup-panel-title">Transform Configuration</span>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Column to transform</label>
                    <select className="form-select" value={selCol} onChange={e => { setSkewTransformCol(e.target.value); loadDistribution(e.target.value); }} disabled={viewOnly}>
                      {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>

                  <div className="flex gap-md" style={{ marginBottom: 12, alignItems: 'stretch' }}>
                    <div className="outlier-metric-card" style={{ flex: '0 0 auto', minWidth: 100 }}>
                      <div className="outlier-metric-label">Skewness</div>
                      <div className="outlier-metric-value">{skewVal != null ? skewVal.toFixed(3) : '—'}</div>
                    </div>
                    <div className={`skew-badge ${skewBadge.cls}`}>
                      {skewBadge.label}
                    </div>
                  </div>

                  <div style={{ borderTop: '1px solid var(--border)', marginBottom: 12 }} />

                  <div className="form-group">
                    <label className="form-label">Transform method</label>
                    <select className="form-select" value={skewMethod} onChange={e => setSkewMethod(e.target.value)} disabled={viewOnly}>
                      {['Log', 'Sqrt', 'Box-Cox', 'Yeo-Johnson', 'Reciprocal'].map(m =>
                        <option key={m} value={m}>{m}</option>
                      )}
                    </select>
                  </div>

                  <div className="cleanup-callout cleanup-callout-info" style={{ marginBottom: 12 }}>
                    <div className="cleanup-callout-icon">📝</div>
                    <div style={{ fontSize: 12 }}>{METHOD_DESC[skewMethod] || ''}</div>
                  </div>

                  <button
                    className="btn btn-primary btn-sm"
                    style={{ width: '100%', justifyContent: 'center' }}
                    disabled={viewOnly || loading || !selCol}
                    onClick={async () => {
                      const data = await run('skewness', () => dataApi.transformSkewness({
                        method: skewMethod,
                        columns: [selCol],
                      }));
                      if (data?.results) setSkewnessResults(data.results);
                      await loadDistribution(selCol);
                    }}
                  >
                    ✅ Apply Transform
                  </button>
                </div>

                {/* RIGHT PANEL: Distribution Visualization */}
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">📊</span>
                    <span className="cleanup-panel-title">Distribution Visualization</span>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Column to visualize</label>
                    <select className="form-select" value={previewCol} onChange={e => { setDistributionColumn(e.target.value); loadDistribution(e.target.value); }}>
                      {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>

                  {distributionLoading && (
                    <div className="flex items-center gap-sm" style={{ padding: 16 }}>
                      <div className="spinner" /> Loading chart...
                    </div>
                  )}

                  {!distributionLoading && distributionData?.histogram && (
                    <>
                      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6 }}>Distribution - {distributionData.column || previewCol}</div>
                      <div className="chart-surface" style={{ marginBottom: 12 }}>
                        <ResponsiveContainer width="100%" height={200}>
                          <BarChart data={distributionData.histogram} margin={{ top: 6, right: 8, left: 0, bottom: 4 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                            <XAxis dataKey="label" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} interval={Math.max(0, Math.floor((distributionData.histogram.length || 1) / 6))} />
                            <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                            <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 10 }} formatter={v => [v, 'Count']} />
                            <Bar dataKey="count" fill="#7389ff" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </>
                  )}

                  {!distributionLoading && distributionData?.summary && (
                    <>
                      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6 }}>📊 Statistics</div>
                      <div className="table-container">
                        <table>
                          <thead><tr><th>Metric</th><th style={{ textAlign: 'right' }}>Value</th></tr></thead>
                          <tbody>
                            {[
                              ['Mean', distributionData.summary.mean],
                              ['Median', distributionData.summary.median],
                              ['Std', distributionData.summary.std],
                              ['Min', distributionData.summary.min],
                              ['Max', distributionData.summary.max],
                              ['Skewness', distributionData.summary.skewness],
                            ].map(([k, v]) => (
                              <tr key={k}>
                                <td>{k}</td>
                                <td style={{ textAlign: 'right' }}>{v != null ? Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Results table below */}
              {skewnessResults && (
                <div style={{ marginTop: 20 }}>
                  <ComparisonMetricChart
                    title="Skewness Before vs After"
                    data={skewChartData}
                    leftKey="before"
                    rightKey="after"
                    leftLabel="Before"
                    rightLabel="After"
                  />
                  <div className="table-container" style={{ marginTop: 16 }}>
                    <table>
                      <thead><tr><th>Feature</th><th>Skew Before</th><th>Skew After</th><th>Improved?</th></tr></thead>
                      <tbody>
                        {Object.entries(skewnessResults)
                          .filter(([, v]) => !v.error)
                          .map(([col, info]) => (
                            <tr key={col}>
                              <td style={{ fontWeight: 600 }}>{col}</td>
                              <td>{info.skew_before}</td>
                              <td>{info.skew_after}</td>
                              <td>{info.improved ? '✅ Yes' : '❌ No'}</td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
            );
          })()}

          {/* Step 6: Encoding */}
          {key === 'encoding' && (() => {
            const selStats = catSummary.find(c => c.column_name === encCol) || null;
            const uniqueVals = selStats?.unique_values || 0;
            
            let recMethod = 'One-Hot Encoding';
            let recReason = 'Default';
            if (uniqueVals === 2) {
              recMethod = 'Label Encoding';
              recReason = `Binary variable (${uniqueVals} categories) - Label Encoding is enough.`;
            } else if (uniqueVals > 2 && uniqueVals <= 10) {
              recMethod = 'One-Hot Encoding';
              recReason = `Low cardinality (${uniqueVals} categories) - One-Hot reduces the assumption of order.`;
            } else if (uniqueVals > 10) {
              recMethod = 'Frequency Encoding';
              recReason = `High cardinality (${uniqueVals} categories) - Frequency Encoding helps reduce data dimensionality.`;
            }

            return (
            <>
              {categoricalCols.length > 0 && (
                <div className="cleanup-callout cleanup-callout-warning" style={{ marginBottom: 16 }}>
                  <div className="cleanup-callout-icon">⚠️</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                    There are {categoricalCols.length} categorical variables that need to be encoded
                  </div>
                </div>
              )}

              <div className="cleanup-grid">
                {/* LEFT PANEL: Danh sách biến phân loại */}
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">📝</span>
                    <span className="cleanup-panel-title">Categorical Variables List</span>
                  </div>
                  <div className="table-container">
                    <table>
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th style={{ textAlign: 'right' }}>Num. of unique values</th>
                          <th>Most common values</th>
                        </tr>
                      </thead>
                      <tbody>
                        {categoricalCols.map(colName => {
                          const stats = catSummary.find(c => c.column_name === colName) || {};
                          return (
                            <tr key={colName}>
                              <td style={{ fontWeight: 600 }}>{colName}</td>
                              <td style={{ textAlign: 'right' }}>{stats.unique_values ?? '—'}</td>
                              <td>{stats.most_common ?? '—'}</td>
                            </tr>
                          );
                        })}
                        {categoricalCols.length === 0 && (
                          <tr><td colSpan="3" style={{ textAlign: 'center', color: 'var(--text-muted)' }}>No categorical variables left to process.</td></tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* RIGHT PANEL: Cấu hình mã hóa */}
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">⚙️</span>
                    <span className="cleanup-panel-title">Configuration for Encoding Each Column</span>
                  </div>

                  <div className="form-group" style={{ marginBottom: 12 }}>
                    <label className="form-label">Select column to encode:</label>
                    <select className="form-select" value={encCol} disabled={viewOnly || categoricalCols.length === 0} onChange={e => {
                      const newCol = e.target.value;
                      setEncCol(newCol);
                      // Auto apply recommendation
                      const stats = catSummary.find(c => c.column_name === newCol);
                      const uVals = stats?.unique_values || 0;
                      if (uVals === 2) setEncMethod('Label Encoding');
                      else if (uVals > 2 && uVals <= 10) setEncMethod('One-Hot Encoding');
                      else if (uVals > 10) setEncMethod('Frequency Encoding');
                    }}>
                      <option value="">-- Choosing column --</option>
                      {categoricalCols.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>

                  {encCol && (
                    <>
                      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 2 }}>Number of unique values</div>
                      <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 16 }}>{uniqueVals}</div>

                      <div className="cleanup-callout cleanup-callout-info" style={{ marginBottom: 16, backgroundColor: 'rgba(59, 130, 246, 0.1)' }}>
                        <div className="cleanup-callout-icon">💡</div>
                        <div style={{ fontSize: 12 }}>
                          <strong>Suggestion: {recMethod}</strong><br/>
                          <span style={{ color: 'var(--text-secondary)' }}>{recReason}</span>
                        </div>
                      </div>

                      <div className="form-group" style={{ marginBottom: 16 }}>
                        <label className="form-label">Encoding method:</label>
                        <select className="form-select" value={encMethod} onChange={e => setEncMethod(e.target.value)} disabled={viewOnly}>
                          {['One-Hot Encoding', 'Label Encoding', 'Target Encoding', 'Ordinal Encoding', 'Frequency Encoding'].map(m =>
                            <option key={m} value={m}>{m}</option>
                          )}
                        </select>
                      </div>

                      {encMethod === 'One-Hot Encoding' && (
                        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, paddingBottom: 16, cursor: 'pointer' }}>
                          <input type="checkbox" checked={encDropFirst} onChange={e => setEncDropFirst(e.target.checked)} disabled={viewOnly} />
                          Drop first dummy (avoid multicollinearity)
                        </label>
                      )}

                      <button
                        className="btn btn-primary btn-sm"
                        style={{ width: '100%', justifyContent: 'center' }}
                        disabled={viewOnly || loading || !encCol}
                        onClick={async () => {
                          const payload = { method: encMethod, columns: [encCol], drop_first: encDropFirst };
                          await run('encoding', () => dataApi.encode(payload));
                          setEncCol(''); // Reset sau khi xử lý thành công
                        }}
                      >
                        ➕ Add Setup
                      </button>
                    </>
                  )}
                  
                  {!encCol && categoricalCols.length > 0 && (
                    <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: 13, marginTop: 32 }}>
                      Please select a column to configure.
                    </div>
                  )}
                </div>
              </div>
            </>
            );
          })()}

          {/* Step 7: WoE/IV Binning */}
          {key === 'binning' && (() => {
            const bCol = binningPreviewCol || numericCols[0] || '';
            const bSummary = distributionData?.column === bCol ? distributionData?.summary : null;
            return (
            <>
              <div className="cleanup-grid">
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">⚙️</span>
                    <span className="cleanup-panel-title">Binning Configuration</span>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Column to preview</label>
                    <select className="form-select" value={bCol} onChange={e => { setBinningPreviewCol(e.target.value); loadDistribution(e.target.value); }} disabled={viewOnly}>
                      {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Binning method</label>
                    <select className="form-select" value={binningMethod} onChange={e => setBinningMethod(e.target.value)} disabled={viewOnly}>
                      <option value="Optimal Binning (WoE/IV)">Optimal Binning (WoE/IV)</option>
                    </select>
                  </div>

                  <div className="cleanup-callout cleanup-callout-info" style={{ marginBottom: 8 }}>
                    <div className="cleanup-callout-icon">ℹ️</div>
                    <div style={{ fontSize: 12 }}>Optimal Binning uses Information Value (IV) to distinguish Good/Bad. Requires a target column.</div>
                  </div>

                  <div className="cleanup-callout cleanup-callout-warning" style={{ marginBottom: 12 }}>
                    <div className="cleanup-callout-icon">⚠️</div>
                    <div style={{ fontSize: 12 }}>Make sure target column is selected in Feature Engineering.</div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Max bins</label>
                    <input className="form-input" type="range" min="3" max="20" step="1"
                      value={binningMaxBins} onChange={e => setBinningMaxBins(+e.target.value)} disabled={viewOnly}
                      style={{ accentColor: 'var(--success)' }} />
                    <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', textAlign: 'center', marginTop: 2 }}>{binningMaxBins}</div>
                  </div>

                  <ColumnMultiSelector
                    label="Columns to bin"
                    columns={numericCols}
                    values={binningCols}
                    onChange={setBinningCols}
                    disabled={viewOnly}
                    placeholder="Choose options"
                    showSelectAll={numericCols.length > 0}
                  />

                  <div className="form-group" style={{ marginTop: 12 }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, cursor: 'pointer', fontWeight: 600 }}>
                      <input type="checkbox" checked={binningMonotonic} onChange={e => setBinningMonotonic(e.target.checked)} disabled={viewOnly} style={{ accentColor: 'var(--error)' }} />
                      📈 Monotonic (Increasing/decreasing bad rate)
                    </label>
                  </div>

                  <div className="form-group" style={{ marginTop: 16 }}>
                    <label className="form-label" style={{ fontSize: 13 }}>New column name:</label>
                    <input className="form-input" style={{ width: '100%', marginBottom: 12 }} type="text" value={binningNewColName} onChange={e => setBinningNewColName(e.target.value)} disabled={viewOnly || binningCols.length !== 1} placeholder={binningCols.length > 1 ? 'Only support new column name when selecting 1 column' : 'Example: age_woe'} />
                  </div>

                  <button
                    className="btn btn-primary btn-sm"
                    style={{ width: '100%', marginTop: 4, justifyContent: 'center' }}
                    disabled={viewOnly || loading || binningCols.length === 0}
                    onClick={async () => {
                      const data = await run('binning', () => dataApi.binning({
                        columns: binningCols,
                        max_n_bins: binningMaxBins,
                        monotonic_trend: binningMonotonic ? "auto_asc_desc" : "auto",
                        new_column_name: (binningNewColName && binningCols.length === 1) ? binningNewColName : null
                      }));
                      if (data?.results) {
                        setBinningResults(data.results);
                        setSelectedBinningFeature(Object.keys(data.results).find((key) => !data.results[key].error) || '');
                      }
                      if (bCol) await loadDistribution(bCol);
                    }}
                  >
                    🔄 Apply Binning
                  </button>
                </div>

                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">📊</span>
                    <span className="cleanup-panel-title">Analysis & Visualization</span>
                  </div>

                  {bSummary && (
                    <div className="flex gap-md" style={{ marginBottom: 12 }}>
                      {[
                        ['Min', bSummary.min],
                        ['Mean', bSummary.mean],
                        ['Max', bSummary.max],
                      ].map(([k, v]) => (
                        <div key={k} className="outlier-metric-card" style={{ flex: 1, textAlign: 'center' }}>
                          <div className="outlier-metric-label">{k}</div>
                          <div className="outlier-metric-value" style={{ fontSize: 18 }}>{v != null ? Number(v).toFixed(2) : '—'}</div>
                        </div>
                      ))}
                    </div>
                  )}

                  {distributionLoading && (
                    <div className="flex items-center gap-sm" style={{ padding: 16 }}>
                      <div className="spinner" /> Loading chart...
                    </div>
                  )}

                  {!distributionLoading && distributionData?.histogram && (
                    <>
                      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6 }}>Distribution - {distributionData.column || bCol}</div>
                      <div className="chart-surface" style={{ marginBottom: 12 }}>
                        <ResponsiveContainer width="100%" height={200}>
                          <BarChart data={distributionData.histogram} margin={{ top: 6, right: 8, left: 0, bottom: 4 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                            <XAxis dataKey="label" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} interval={Math.max(0, Math.floor((distributionData.histogram.length || 1) / 6))} />
                            <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                            <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 10 }} formatter={v => [v, 'Count']} />
                            <Bar dataKey="count" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Results below */}
              {binningResults && (
                <div style={{ marginTop: 20 }}>
                  <MetricRankingChart title="IV Ranking After Binning" data={binningChartData} valueKey="value" color="#8b5cf6" />
                  {selectedBinningFeature && (
                    <div className="form-group" style={{ marginTop: 16 }}>
                      <label className="form-label">Feature Bin Details</label>
                      <select className="form-select" value={selectedBinningFeature} onChange={(e) => setSelectedBinningFeature(e.target.value)}>
                        {Object.keys(binningResults).filter((key) => !binningResults[key].error).map((key) => (
                          <option key={key} value={key}>{key}</option>
                        ))}
                      </select>
                    </div>
                  )}
                  <WoeBinsChart title="WoE and Count by Bin" data={selectedBinningBins} />
                  <div className="table-container" style={{ marginTop: 16 }}>
                    <table>
                      <thead><tr><th>Feature</th><th>IV</th><th>Bins</th><th>Predictive Power</th></tr></thead>
                      <tbody>
                        {Object.entries(binningResults)
                          .filter(([, v]) => !v.error)
                          .sort(([, a], [, b]) => (b.iv || 0) - (a.iv || 0))
                          .map(([col, info]) => (
                            <tr key={col}>
                              <td style={{ fontWeight: 600 }}>{col}</td>
                              <td>{info.iv}</td>
                              <td>{info.n_bins}</td>
                              <td><span className={`badge ${info.predictive_power === 'Suspicious' ? 'badge-error' : info.predictive_power === 'Strong' ? 'badge-success' : info.predictive_power === 'Medium' ? 'badge-warning' : 'badge-error'}`}>{info.predictive_power}</span></td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
            );
          })()}

          {/* Step 8: WoE Analysis */}
          {key === 'woe_analysis' && (
            <>
              <div className="cleanup-grid">
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">⚙️</span>
                    <span className="cleanup-panel-title">WoE Analysis Configuration</span>
                  </div>

                  <div className="cleanup-callout cleanup-callout-info" style={{ marginBottom: 12 }}>
                    <div className="cleanup-callout-icon">ℹ️</div>
                    <div style={{ fontSize: 12 }}>Analyzes all numeric features against the target column. Does NOT transform data — use Binning step to apply WoE transformation.</div>
                  </div>

                  <button
                    className="btn btn-primary btn-sm"
                    style={{ width: '100%', justifyContent: 'center' }}
                    disabled={viewOnly || loading}
                    onClick={async () => {
                      const data = await run('woe_analysis', () => dataApi.woeAnalysis({}));
                      if (data?.results) {
                        setWoeResults(data);
                        setSelectedWoeFeature(Object.keys(data.results)[0] || '');
                      }
                    }}
                  >
                    ✅ Run WoE Analysis
                  </button>

                  {woeResults && (
                    <>
                      <div className="flex gap-md" style={{ marginTop: 16, flexWrap: 'wrap' }}>
                        {['Suspicious', 'Strong', 'Medium', 'Weak', 'Useless'].map(p => (
                          <div key={p} className="outlier-metric-card" style={{ flex: 1, textAlign: 'center', minWidth: 80 }}>
                            <div className="outlier-metric-label">{p}</div>
                            <div className="outlier-metric-value" style={{ fontSize: 20 }}>{woeResults.summary?.[p.toLowerCase()] || 0}</div>
                          </div>
                        ))}
                      </div>

                      {selectedWoeFeature && (
                        <div className="form-group" style={{ marginTop: 16 }}>
                          <label className="form-label">Feature Bin Details</label>
                          <select className="form-select" value={selectedWoeFeature} onChange={(e) => setSelectedWoeFeature(e.target.value)}>
                            {Object.keys(woeResults.results).map((key) => (
                              <option key={key} value={key}>{key}</option>
                            ))}
                          </select>
                        </div>
                      )}
                    </>
                  )}
                </div>

                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">📊</span>
                    <span className="cleanup-panel-title">WoE Results</span>
                  </div>

                  {!woeResults && (
                    <div className="empty-state" style={{ padding: '32px 20px' }}>
                      Run the analysis to see IV ranking and WoE charts.
                    </div>
                  )}

                  {woeResults && (
                    <>
                      <MetricRankingChart title="WoE Analysis IV Ranking" data={woeChartData} valueKey="value" color="#10b981" />
                      <WoeBinsChart title="WoE Analysis by Bin" data={selectedWoeBins} />
                    </>
                  )}
                </div>
              </div>

              {woeResults && (
                <div className="table-container" style={{ marginTop: 20 }}>
                  <table>
                    <thead><tr><th>Feature</th><th>IV</th><th>Power</th><th>Recommendation</th></tr></thead>
                    <tbody>
                      {Object.entries(woeResults.results).map(([col, info]) => (
                        <tr key={col}>
                          <td style={{ fontWeight: 600 }}>{col}</td>
                          <td>{info.iv}</td>
                          <td><span className={`badge ${info.predictive_power === 'Suspicious' ? 'badge-error' : info.predictive_power === 'Strong' ? 'badge-success' : info.predictive_power === 'Medium' ? 'badge-warning' : 'badge-error'}`}>{info.predictive_power}</span></td>
                          <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>{info.recommendation}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}

          {/* Step 9: Multicollinearity */}
          {key === 'multicollinearity' && (
            <>
              <div className="cleanup-grid">
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">⚙️</span>
                    <span className="cleanup-panel-title">Multicollinearity Check</span>
                  </div>

                  <div className="cleanup-callout cleanup-callout-info" style={{ marginBottom: 12 }}>
                    <div className="cleanup-callout-icon">ℹ️</div>
                    <div style={{ fontSize: 12 }}>Detects features with high Variance Inflation Factor (VIF &gt; 10) and highly correlated feature pairs (&ge; 0.8).</div>
                  </div>

                  <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12, cursor: 'pointer' }}>
                    <input type="checkbox" checked={autoRemoveVif} onChange={e => setAutoRemoveVif(e.target.checked)} disabled={viewOnly} />
                    Auto-remove features with high VIF
                  </label>

                  <button
                    className="btn btn-primary btn-sm"
                    style={{ width: '100%', justifyContent: 'center' }}
                    disabled={viewOnly || loading}
                    onClick={async () => {
                      const data = await run('multicollinearity', () => dataApi.multicollinearity({
                        vif_threshold: 10.0,
                        corr_threshold: 0.8,
                        auto_remove: autoRemoveVif,
                      }));
                      if (data?.vif) setVifResults(data.vif);
                      if (data?.high_correlation_pairs) setCorrPairs(data.high_correlation_pairs);
                    }}
                  >
                    ✅ Check Multicollinearity
                  </button>

                  {/* ── Restore accidentally removed columns ── */}
                  <div style={{ marginTop: 20, borderTop: '1px solid var(--border)', paddingTop: 16 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6, color: 'var(--warning)' }}>🔄 Restore Removed Columns</div>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>
                      Accidentally removed columns by VIF auto-remove? Type column names (comma-separated) to restore them from the original pre-split data.
                    </div>
                    <textarea
                      className="form-input"
                      style={{ width: '100%', minHeight: 64, resize: 'vertical', fontFamily: 'monospace', fontSize: 12 }}
                      placeholder="e.g. annual_income, debt_to_income_ratio, age"
                      value={restoreColsInput}
                      onChange={e => setRestoreColsInput(e.target.value)}
                      disabled={viewOnly}
                    />
                    <button
                      className="btn btn-secondary btn-sm"
                      style={{ width: '100%', marginTop: 8, justifyContent: 'center' }}
                      disabled={viewOnly || loading || !restoreColsInput.trim()}
                      onClick={async () => {
                        const cols = restoreColsInput.split(',').map(c => c.trim()).filter(Boolean);
                        if (!cols.length) return;
                        const data = await run('multicollinearity', () => dataApi.restoreColumns(cols));
                        if (data?.restored?.length) {
                          setRestoreColsInput('');
                        }
                      }}
                    >
                      ⬆️ Restore Columns
                    </button>
                  </div>
                </div>

                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">📊</span>
                    <span className="cleanup-panel-title">VIF Results</span>
                  </div>

                  {!vifResults && (
                    <div className="empty-state" style={{ padding: '32px 20px' }}>
                      Run the check to see VIF scores and correlation pairs.
                    </div>
                  )}

                  {vifResults && (
                    <>
                      <div className="flex gap-md" style={{ marginBottom: 12 }}>
                        <div className="outlier-metric-card" style={{ flex: 1, textAlign: 'center' }}>
                          <div className="outlier-metric-label">Total Features</div>
                          <div className="outlier-metric-value" style={{ fontSize: 20 }}>{vifResults.length}</div>
                        </div>
                        <div className="outlier-metric-card" style={{ flex: 1, textAlign: 'center' }}>
                          <div className="outlier-metric-label">High VIF</div>
                          <div className="outlier-metric-value" style={{ fontSize: 20, color: 'var(--error)' }}>{vifResults.filter(v => v.high_vif).length}</div>
                        </div>
                        <div className="outlier-metric-card" style={{ flex: 1, textAlign: 'center' }}>
                          <div className="outlier-metric-label">Corr Pairs</div>
                          <div className="outlier-metric-value" style={{ fontSize: 20, color: 'var(--warning)' }}>{corrPairs?.length || 0}</div>
                        </div>
                      </div>

                      <div className="table-container">
                        <table>
                          <thead><tr><th>Feature</th><th>VIF</th><th>Status</th></tr></thead>
                          <tbody>
                            {vifResults.map(v => (
                              <tr key={v.feature} style={{ background: v.high_vif ? 'rgba(255,100,100,0.08)' : 'transparent' }}>
                                <td style={{ fontWeight: 600 }}>{v.feature}</td>
                                <td>{v.vif}</td>
                                <td>{v.high_vif ? <span className="badge badge-error">High VIF</span> : <span className="badge badge-success">OK</span>}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {corrPairs && corrPairs.length > 0 && (
                <div className="table-container" style={{ marginTop: 20 }}>
                  <div className="card-title" style={{ fontSize: 13, marginBottom: 8 }}>Highly Correlated Pairs (≥ 0.8)</div>
                  <table>
                    <thead><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th></tr></thead>
                    <tbody>
                      {corrPairs.map((p, i) => (
                        <tr key={i}>
                          <td>{p.feature_1}</td>
                          <td>{p.feature_2}</td>
                          <td style={{ fontWeight: 600 }}>{p.correlation}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}

          {/* Step 10: Scaling */}
          {key === 'scaling' && (() => {
            return (
            <>
              <div className="cleanup-callout cleanup-callout-info" style={{ marginBottom: 16 }}>
                <div className="cleanup-callout-icon">💡</div>
                <div style={{ fontSize: 13, fontWeight: 600 }}>Scaling standardizes variables onto a common scale, critical for:</div>
                <ul style={{ margin: '8px 0 0 24px', fontSize: 12, padding: 0 }}>
                  <li>Linear Regression, Logistic Regression</li>
                  <li>Neural Networks, Deep Learning</li>
                  <li>K-Nearest Neighbors (KNN)</li>
                  <li>Support Vector Machines (SVM)</li>
                  <li>Gradient Descent optimization</li>
                </ul>
              </div>

              <div className="cleanup-grid">
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">⚙️</span>
                    <span className="cleanup-panel-title">Scaling Configuration</span>
                  </div>

                  <div className="form-group" style={{ marginBottom: 12 }}>
                    <label className="form-label">Scaling method:</label>
                    <select className="form-select" value={scaleMethod} onChange={e => setScaleMethod(e.target.value)} disabled={viewOnly}>
                      {['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler'].map(m =>
                        <option key={m} value={m}>{m}</option>
                      )}
                    </select>
                  </div>

                  <ColumnMultiSelector
                    label="Select columns to scale:"
                    columns={numericCols}
                    values={scaleCols}
                    onChange={setScaleCols}
                    disabled={viewOnly}
                    placeholder="Choose options"
                    showSelectAll={numericCols.length > 0}
                  />

                  <div className="form-group" style={{ marginTop: 16, marginBottom: 12 }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, cursor: 'pointer' }}>
                      <input type="checkbox" checked={scaleCreateNew} onChange={e => setScaleCreateNew(e.target.checked)} disabled={viewOnly} />
                      Create new columns (keep original columns)
                    </label>
                  </div>

                  <button
                    className="btn btn-primary btn-sm"
                    style={{ width: '100%', justifyContent: 'center' }}
                    disabled={viewOnly || loading || noSplits || scaleCols.length === 0}
                    onClick={() => {
                      // Note: create_new_columns is not natively supported in the basic pipeline yet, 
                      // but we pass it anyway to be handled by future backend updates.
                      run('scaling', () => dataApi.scale({ method: scaleMethod, columns: scaleCols, create_new_columns: scaleCreateNew }));
                    }}
                  >
                    🔄 Execute Scaling
                  </button>
                </div>

                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">📊</span>
                    <span className="cleanup-panel-title">Scaling Information</span>
                  </div>

                  <div style={{ fontSize: 13, marginBottom: 8, fontWeight: 600, color: 'var(--success)' }}>Scaling Methods Summary:</div>
                  <ul style={{ margin: '0 0 16px 24px', fontSize: 13, padding: 0, lineHeight: '1.6' }}>
                    <li><strong>StandardScaler:</strong> Mean=0, Std=1</li>
                    <li><strong>MinMaxScaler:</strong> Scale to [0, 1]</li>
                    <li><strong>RobustScaler:</strong> Uses median & IQR</li>
                    <li><strong>MaxAbsScaler:</strong> Scale to [-1, 1]</li>
                    <li><strong>Normalizer:</strong> Normalize per sample</li>
                  </ul>
                  
                  <div className="cleanup-callout cleanup-callout-warning">
                    <div className="cleanup-callout-icon">⚠️</div>
                    <div style={{ fontSize: 12 }}>Scaler is fitted on train set only and applied to validation/test to prevent data leakage.</div>
                  </div>
                </div>
              </div>
            </>
            );
          })()}

          {/* Step 11: Balance */}
          {key === 'balance' && (() => {
            let classesCount = 0;
            let ratio = 0;
            let classDetails = [];
            const beforeDist = balanceInfo?.original_distribution || null;
            const afterDist = balanceInfo?.balanced_distribution || null;
            const distToRows = (dist) => {
              if (!dist || typeof dist !== 'object') return [];
              const total = Object.values(dist).reduce((sum, value) => sum + Number(value || 0), 0);
              return Object.entries(dist)
                .map(([label, count]) => ({
                  label,
                  count: Number(count || 0),
                  pct: total > 0 ? ((Number(count || 0) / total) * 100).toFixed(1) : '0.0',
                }))
                .sort((a, b) => String(a.label).localeCompare(String(b.label)));
            };
            const beforeRows = distToRows(beforeDist);
            const afterRows = distToRows(afterDist);
            
            if (balDist?.distribution?.counts && balDist?.distribution?.labels) {
              const counts = balDist.distribution.counts;
              const labels = balDist.distribution.labels;
              classesCount = labels.length;
              if (classesCount >= 2) {
                const sortedIdx = counts.map((v, i) => i).sort((a, b) => counts[b] - counts[a]);
                if (counts[sortedIdx[1]] > 0) {
                  ratio = counts[sortedIdx[0]] / counts[sortedIdx[1]];
                }
                const total = counts.reduce((sum, v) => sum + v, 0);
                classDetails = labels.map((l, i) => ({
                  label: l,
                  count: counts[i],
                  pct: total > 0 ? ((counts[i] / total) * 100).toFixed(1) : 0
                })).sort((a, b) => a.label - b.label); // simplified sort if labels are 0,1
              }
            } else if (balDist?.distribution?.histogram_bins) {
              // fallback if it was identified as continuous
              const bins = balDist.summary.unique_values;
              classesCount = bins || 2;
            }

            return (
            <>
              <div className="cleanup-grid">
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">⚙️</span>
                    <span className="cleanup-panel-title">Balancing Configuration</span>
                  </div>

                  {!targetCol && (
                    <div className="cleanup-callout cleanup-callout-warning" style={{ marginBottom: 16 }}>
                      <div className="cleanup-callout-icon">⚠️</div>
                      <div style={{ fontSize: 12, fontWeight: 600 }}>Target column not selected. Please select a target in the Split Train/Valid/Test section.</div>
                    </div>
                  )}

                  <div className="form-group" style={{ marginBottom: 12 }}>
                    <label className="form-label">Target column (from setup):</label>
                    <input
                      className="form-input"
                      type="text"
                      value={targetCol || 'Not selected'}
                      readOnly
                    />
                  </div>

                  <div className="form-group" style={{ marginBottom: 12 }}>
                    <label className="form-label">Method:</label>
                    <select className="form-select" value={balanceMethod} onChange={e => setBalanceMethod(e.target.value)} disabled={viewOnly || !targetCol}>
                      {['SMOTE', 'ADASYN', 'SMOTE-ENN', 'SMOTE-Tomek', 'Random Over-sampling', 'Random Under-sampling'].map(m =>
                        <option key={m} value={m}>{m}</option>
                      )}
                    </select>
                  </div>

                  <div className="form-group" style={{ marginBottom: 16 }}>
                    <label className="form-label">Sampling strategy:</label>
                    <select className="form-select" value={balStrategy} onChange={e => setBalStrategy(e.target.value)} disabled={viewOnly || !targetCol}>
                      <option value="auto">auto</option>
                      <option value="minority">minority</option>
                      <option value="not minority">not minority</option>
                      <option value="all">all</option>
                    </select>
                  </div>

                  <button
                    className="btn btn-primary btn-sm"
                    style={{ width: '100%', justifyContent: 'center' }}
                    disabled={viewOnly || loading || noSplits || !targetCol}
                    onClick={async () => {
                      const data = await run('balance', () => dataApi.balance({ method: balanceMethod, sampling_strategy: balStrategy, target_column: targetCol }));
                      if (data?.info) setBalanceInfo(data.info);
                    }}
                  >
                    ✅ Balance Classes
                  </button>
                </div>

                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">📊</span>
                    <span className="cleanup-panel-title">Class Distribution</span>
                  </div>

                  {targetCol && balDist ? (
                    <>
                      <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Target column</div>
                      <div style={{ fontSize: 28, fontWeight: 800, marginBottom: 24, wordBreak: 'break-all' }}>{targetCol}</div>

                      <div className="flex gap-lg" style={{ marginBottom: 24 }}>
                        <div>
                          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Num classes</div>
                          <div style={{ fontSize: 20, fontWeight: 700 }}>{classesCount || '—'}</div>
                        </div>
                        <div>
                          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Imbalance ratio</div>
                          <div style={{ fontSize: 20, fontWeight: 700 }}>{ratio ? ratio.toFixed(2) : '—'}</div>
                        </div>
                      </div>

                      {beforeRows.length > 0 && (
                        <>
                          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Before balancing:</div>
                          <ul style={{ margin: '0 0 16px 24px', fontSize: 13, padding: 0, lineHeight: '1.6' }}>
                            {beforeRows.map((c) => (
                              <li key={`before-${c.label}`}>Class {c.label}: {c.count} ({c.pct}%)</li>
                            ))}
                          </ul>
                        </>
                      )}

                      {afterRows.length > 0 && (
                        <>
                          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>After balancing:</div>
                          <ul style={{ margin: '0 0 24px 24px', fontSize: 13, padding: 0, lineHeight: '1.6' }}>
                            {afterRows.map((c) => (
                              <li key={`after-${c.label}`}>Class {c.label}: {c.count} ({c.pct}%)</li>
                            ))}
                          </ul>
                        </>
                      )}

                      {beforeRows.length === 0 && afterRows.length === 0 && (
                        <>
                          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Distribution per class:</div>
                          <ul style={{ margin: '0 0 24px 24px', fontSize: 13, padding: 0, lineHeight: '1.6' }}>
                            {classDetails.map(c => (
                              <li key={c.label}>Class {c.label}: {c.count} ({c.pct}%)</li>
                            ))}
                            {classDetails.length === 0 && <li>No distribution data yet. Click Balance Classes to compute before/after.</li>}
                          </ul>
                        </>
                      )}

                      {ratio >= 1.5 && (
                        <div className="cleanup-callout cleanup-callout-warning" style={{ backgroundColor: 'rgba(234, 179, 8, 0.15)', borderColor: 'rgba(234, 179, 8, 0.5)', marginBottom: 12 }}>
                          <div className="cleanup-callout-icon">⚠️</div>
                          <div style={{ fontSize: 12, fontWeight: 600 }}>Dataset is imbalanced! Ratio: {ratio.toFixed(2)}</div>
                        </div>
                      )}
                      
                      {ratio >= 1.5 && (
                        <div className="cleanup-callout cleanup-callout-info" style={{ backgroundColor: 'rgba(59, 130, 246, 0.1)' }}>
                          <div className="cleanup-callout-icon">💡</div>
                          <div style={{ fontSize: 12 }}>
                            Gợi ý: SMOTE recommended - {ratio > 10 ? 'Severe' : 'Moderate'} imbalance
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="empty-state" style={{ padding: '40px 20px' }}>
                      {targetCol ? 'No distribution data available for target.' : 'Please select a target first to view class distribution.'}
                    </div>
                  )}
                </div>
              </div>
            </>
            );
          })()}

          {/* Step 12: Feature Importance */}
          {key === 'importance' && (
            <>
              <div className="cleanup-grid">
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">⚙️</span>
                    <span className="cleanup-panel-title">Importance Configuration</span>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Method</label>
                    <select className="form-select" value={impMethod} onChange={e => setImpMethod(e.target.value)} disabled={viewOnly}>
                      {['Random Forest', 'LightGBM', 'XGBoost', 'Logistic Regression (Coef)'].map(m =>
                        <option key={m} value={m}>{m}</option>
                      )}
                    </select>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Top N Features</label>
                    <input className="form-input" type="range" min="5" max="50" step="1"
                      value={impTopN} onChange={e => setImpTopN(+e.target.value)} disabled={viewOnly}
                      style={{ accentColor: 'var(--accent)' }} />
                    <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', textAlign: 'center', marginTop: 2 }}>{impTopN}</div>
                  </div>

                  <button
                    className="btn btn-primary btn-sm"
                    style={{ width: '100%', justifyContent: 'center' }}
                    disabled={viewOnly || loading || noSplits}
                    onClick={async () => {
                      const data = await run('importance', () => dataApi.featureImportance({ method: impMethod, top_n: impTopN, columns: selectedTrainCols.length ? selectedTrainCols : undefined }));
                      const normalized = normalizeImportanceResults(data);
                      setImportanceResults(normalized);
                    }}
                  >
                    ✅ Calculate Importance
                  </button>

                  <div className="form-group" style={{ marginTop: 16 }}>
                    <label className="form-label">Select features for model training</label>
                    <ColumnMultiSelector
                      label=""
                      columns={sessionInfo?.split_feature_columns || []}
                      values={selectedTrainCols}
                      onChange={setSelectedTrainCols}
                      disabled={viewOnly || noSplits}
                      placeholder="Choose training features"
                      showSelectAll={(sessionInfo?.split_feature_columns || []).length > 0}
                    />
                  </div>

                  <div className="flex gap-sm" style={{ marginTop: 8 }}>
                    <button
                      className="btn btn-sm"
                      style={{ flex: 1, justifyContent: 'center', fontSize: 11, background: 'rgba(239, 68, 68, 0.15)', color: '#ef4444', border: '1px solid rgba(239, 68, 68, 0.3)' }}
                      disabled={viewOnly || noSplits}
                      onClick={() => setSelectedTrainCols(prev => prev.filter(c => !c.endsWith('_woe')))}
                    >
                      🚫 Remove WoE
                    </button>
                    <button
                      className="btn btn-sm"
                      style={{ flex: 1, justifyContent: 'center', fontSize: 11, background: 'rgba(59, 130, 246, 0.15)', color: '#3b82f6', border: '1px solid rgba(59, 130, 246, 0.3)' }}
                      disabled={viewOnly || noSplits}
                      onClick={() => setSelectedTrainCols(prev => prev.filter(c => c.endsWith('_woe')))}
                    >
                      🔄 Only WoE
                    </button>
                  </div>

                  <button
                    className="btn btn-secondary btn-sm"
                    style={{ width: '100%', justifyContent: 'center', marginTop: 8 }}
                    disabled={viewOnly || loading || noSplits || selectedTrainCols.length === 0}
                    onClick={async () => {
                      setLoading(true);
                      setMessage('');
                      try {
                        const res = await dataApi.setSelectedFeatures(selectedTrainCols);
                        setMessage(res.data?.message || 'Selected features updated');
                        await loadSessionInfo();
                      } catch (err) {
                        setMessage('Error: ' + (err.response?.data?.detail || err.message));
                      } finally {
                        setLoading(false);
                      }
                    }}
                  >
                    💾 Use Selected Features For Training
                  </button>
                </div>

                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">📊</span>
                    <span className="cleanup-panel-title">Importance Results</span>
                  </div>

                  {!importanceResults && (
                    <div className="empty-state" style={{ padding: '32px 20px' }}>
                      Run the analysis to see feature importance ranking.
                    </div>
                  )}

                  {importanceResults && (
                    <>
                      <div className="flex gap-md" style={{ marginBottom: 12, flexWrap: 'wrap' }}>
                        <div className="outlier-metric-card" style={{ flex: 1, minWidth: 120, textAlign: 'center' }}>
                          <div className="outlier-metric-label">Method</div>
                          <div className="outlier-metric-value" style={{ fontSize: 18 }}>{impMethod}</div>
                        </div>
                        <div className="outlier-metric-card" style={{ flex: 1, minWidth: 120, textAlign: 'center' }}>
                          <div className="outlier-metric-label">Top Features</div>
                          <div className="outlier-metric-value" style={{ fontSize: 18 }}>{importanceRankedRows.length}</div>
                        </div>
                        <div className="outlier-metric-card" style={{ flex: 1, minWidth: 120, textAlign: 'center' }}>
                          <div className="outlier-metric-label">Selected Train Features</div>
                          <div className="outlier-metric-value" style={{ fontSize: 18 }}>{sessionInfo?.n_features || selectedTrainCols.length}</div>
                        </div>
                      </div>

                      <div className="form-group" style={{ marginBottom: 12 }}>
                        <label className="form-label">Auto-select threshold: {impThreshold.toFixed(3)}</label>
                        <input
                          className="form-input"
                          type="range"
                          min="0"
                          max={Math.max(0.2, importanceMaxScore)}
                          step="0.001"
                          value={impThreshold}
                          onChange={(e) => setImpThreshold(+e.target.value)}
                          style={{ accentColor: 'var(--warning)' }}
                        />
                        <button
                          className="btn btn-secondary btn-sm"
                          style={{ marginTop: 8, width: '100%', justifyContent: 'center' }}
                          disabled={viewOnly || selectedByThreshold.length === 0}
                          onClick={() => setSelectedTrainCols(selectedByThreshold)}
                        >
                          ⚡ Auto-select by Threshold ({selectedByThreshold.length})
                        </button>
                      </div>

                      <div className="chart-surface" style={{ marginBottom: 12 }}>
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart
                            data={importanceRankedRows}
                            layout="vertical"
                            margin={{ top: 6, right: 8, left: 20, bottom: 6 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                            <XAxis type="number" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                            <YAxis
                              type="category"
                              dataKey="feature"
                              width={140}
                              tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                            />
                            <Tooltip
                              contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 10 }}
                              formatter={(value) => [Number(value).toFixed(4), 'Importance']}
                            />
                            <Bar dataKey="score" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="table-container">
                        <table>
                          <thead><tr><th>#</th><th>Feature</th><th>Importance</th><th style={{ width: '40%' }}>Bar</th></tr></thead>
                          <tbody>
                            {importanceRankedRows.map(({ feature, score }, i) => (
                              <tr key={feature}>
                                <td>{i + 1}</td>
                                <td style={{ fontWeight: 600 }}>{feature}</td>
                                <td>{score.toFixed(4)}</td>
                                <td>
                                  <div style={{
                                    height: 8, borderRadius: 4,
                                    background: `linear-gradient(90deg, var(--accent), var(--accent-hover))`,
                                    width: `${Math.min(100, score * 100 / importanceMaxScore)}%`,
                                  }} />
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </>
                  )}
                  {importanceResults && importanceRankedRows.length === 0 && (
                    <div className="empty-state" style={{ padding: '24px 20px' }}>
                      Importance analysis completed but returned no feature scores.
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      ))}

      {/* Session summary */}
      {sessionInfo && !noData && (
        <div className="card section" style={{ background: 'var(--surface-hover)', borderStyle: 'dashed' }}>
          <div className="card-title" style={{ marginBottom: 12 }}>📊 Session Summary</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, fontSize: 13 }}>
            <div><span style={{ color: 'var(--text-muted)' }}>Data Shape:</span> {sessionInfo.data_shape ? `${sessionInfo.data_shape[0]} × ${sessionInfo.data_shape[1]}` : '—'}</div>
            <div><span style={{ color: 'var(--text-muted)' }}>Target:</span> {sessionInfo.target_column || '—'}</div>
            <div><span style={{ color: 'var(--text-muted)' }}>Has Splits:</span> {sessionInfo.has_splits ? '✅ Yes' : '❌ No'}</div>
            <div><span style={{ color: 'var(--text-muted)' }}>Features:</span> {sessionInfo.n_features || 0}</div>
            <div><span style={{ color: 'var(--text-muted)' }}>Models Trained:</span> {sessionInfo.n_trained_models || 0}</div>
            <div><span style={{ color: 'var(--text-muted)' }}>Steps Done:</span> {completedSteps.size} / {STEP_CONFIG.length}</div>
          </div>
        </div>
      )}
    </div>
  );
}

function getStepOpacity(key, completedSteps, noSplits) {
  if (key === 'split') return 1;
  if (noSplits && key !== 'split') return 0.5;
  return 1;
}
