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
  const [distributionColumn, setDistributionColumn] = useState('');
  const [distributionData, setDistributionData] = useState(null);
  const [distributionLoading, setDistributionLoading] = useState(false);
  const [selectedBinningFeature, setSelectedBinningFeature] = useState('');
  const [selectedWoeFeature, setSelectedWoeFeature] = useState('');
  const [outlierResults, setOutlierResults] = useState(null);

  // Step states
  const [testSize, setTestSize] = useState(0.2);
  const [validSize, setValidSize] = useState(0.1);

  const [cleanupStrategy, setCleanupStrategy] = useState('drop_rows');
  const [cleanupCols, setCleanupCols] = useState([]);
  const [invalidCols, setInvalidCols] = useState([]);

  const [missingMethod, setMissingMethod] = useState('Mean Imputation');
  const [missingCols, setMissingCols] = useState([]);

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

  const [binningCols, setBinningCols] = useState([]);
  const [binningMaxBins, setBinningMaxBins] = useState(10);
  const [binningMethod, setBinningMethod] = useState('Optimal Binning (WoE/IV)');
  const [binningPreviewCol, setBinningPreviewCol] = useState('');

  const [scaleMethod, setScaleMethod] = useState('StandardScaler');

  const [balanceMethod, setBalanceMethod] = useState('SMOTE');

  const [impMethod, setImpMethod] = useState('Random Forest');
  const [impTopN, setImpTopN] = useState(15);

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

  const loadSessionInfo = async () => {
    try {
      const res = await dataApi.getSessionInfo();
      setSessionInfo(res.data);
      if (res.data.has_splits) setCompletedSteps(prev => new Set([...prev, 'split']));
      if (res.data.has_data) {
        const infoRes = await dataApi.getInfo();
        setDataInfo(infoRes.data);
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
              <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
                <div className="form-group" style={{ flex: 1, minWidth: 150 }}>
                  <label className="form-label">Test Size</label>
                  <input className="form-input" type="number" min="0.05" max="0.5" step="0.05"
                    value={testSize} onChange={e => setTestSize(+e.target.value)} disabled={viewOnly} />
                </div>
                <div className="form-group" style={{ flex: 1, minWidth: 150 }}>
                  <label className="form-label">Validation Size</label>
                  <input className="form-input" type="number" min="0.05" max="0.3" step="0.05"
                    value={validSize} onChange={e => setValidSize(+e.target.value)} disabled={viewOnly} />
                </div>
                <div className="form-group" style={{ flex: 1, minWidth: 150 }}>
                  <label className="form-label">Train Size (auto)</label>
                  <input className="form-input" type="text" disabled value={`${((1 - testSize - validSize) * 100).toFixed(0)}%`} />
                </div>
              </div>
              <button className="btn btn-primary btn-sm" disabled={viewOnly || loading}
                onClick={() => run('split', () => dataApi.split({ test_size: testSize, valid_size: validSize }))}>
                Split Data
              </button>
            </>
          )}

          {/* Step 2: Cleanup */}
          {key === 'cleanup' && (
            <div className="cleanup-grid">
              {/* LEFT PANEL: Remove Identifier Variables */}
              <div className="cleanup-panel">
                <div className="cleanup-panel-header">
                  <span className="cleanup-panel-icon">🔍</span>
                  <span className="cleanup-panel-title">Remove Categorical Variables</span>
                </div>

                <div className="cleanup-callout cleanup-callout-info">
                  <div className="cleanup-callout-icon">💡</div>
                  <div>
                    <p style={{ margin: 0, fontSize: 13, fontWeight: 600 }}>Categorical variables can't be used to predict, should be removed from model:</p>
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
                  Dataset currently has {(numericCols.length + categoricalCols.length) || '—'} columns
                </div>

                {/* Columns info table */}
                <div className="table-container" style={{ marginBottom: 14, maxHeight: 260, overflowY: 'auto' }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Cột</th>
                        <th style={{ textAlign: 'right' }}>Number Of Unique Values</th>
                        <th style={{ textAlign: 'right' }}>Unique Rate (%)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...categoricalCols, ...numericCols].map((col) => {
                        const totalRows = sessionInfo?.split_shape?.[0] || sessionInfo?.data_shape?.[0] || 1;
                        const uniqueCount = dataInfo?.unique_counts?.[col] ?? '—';
                        const uniquePct = typeof uniqueCount === 'number'
                          ? ((uniqueCount / totalRows) * 100).toFixed(1)
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
                  columns={[...categoricalCols, ...numericCols]}
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
                  onClick={() => run('cleanup', () => dataApi.removeCategorical(false, true))}
                >
                  🗑️ Remove Categorical Columns
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
                  columns={numericCols}
                  values={invalidCols}
                  onChange={setInvalidCols}
                  disabled={viewOnly || noSplits}
                  placeholder="Choose options"
                  showSelectAll={numericCols.length > 0}
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
              <button className="btn btn-primary btn-sm" disabled={viewOnly || loading || noSplits || missingCols.length === 0}
                onClick={() => run('missing', () => dataApi.handleMissing({
                  method: missingMethod,
                  columns: missingCols,
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
            const ENC_DESC = {
              'One-Hot Encoding': 'Creates binary columns per category. Best for nominal features with few unique values.',
              'Label Encoding': 'Maps each category to an integer. Simple but implies ordinal relationship.',
              'Target Encoding': 'Replaces categories with mean of target. Powerful but risk of overfitting.',
              'Ordinal Encoding': 'Maps categories to ordered integers. Use when categories have natural order.',
              'Frequency Encoding': 'Replaces categories with their frequency. Preserves information about rarity.',
            };
            return (
            <>
              <div className="cleanup-grid">
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">⚙️</span>
                    <span className="cleanup-panel-title">Encoding Configuration</span>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Method</label>
                    <select className="form-select" value={encMethod} onChange={e => setEncMethod(e.target.value)} disabled={viewOnly}>
                      {['One-Hot Encoding', 'Label Encoding', 'Target Encoding', 'Ordinal Encoding', 'Frequency Encoding'].map(m =>
                        <option key={m} value={m}>{m}</option>
                      )}
                    </select>
                  </div>

                  <div className="cleanup-callout cleanup-callout-info" style={{ marginBottom: 12 }}>
                    <div className="cleanup-callout-icon">📝</div>
                    <div style={{ fontSize: 12 }}>{ENC_DESC[encMethod] || ''}</div>
                  </div>

                  <ColumnMultiSelector
                    label="Columns to encode"
                    columns={categoricalCols}
                    values={encCols}
                    onChange={setEncCols}
                    disabled={viewOnly}
                    placeholder="Choose categorical columns"
                    showSelectAll={categoricalCols.length > 0}
                  />

                  <button
                    className="btn btn-primary btn-sm"
                    style={{ width: '100%', marginTop: 4, justifyContent: 'center' }}
                    disabled={viewOnly || loading || encCols.length === 0}
                    onClick={() => run('encoding', () => dataApi.encode({ method: encMethod, columns: encCols }))}
                  >
                    ✅ Apply Encoding
                  </button>
                </div>

                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">📊</span>
                    <span className="cleanup-panel-title">Encoding Guide</span>
                  </div>

                  <div className="table-container">
                    <table>
                      <thead><tr><th>Method</th><th>Best For</th><th>Pros</th></tr></thead>
                      <tbody>
                        <tr><td style={{ fontWeight: 600 }}>One-Hot</td><td>Nominal, few categories</td><td>No ordinal assumption</td></tr>
                        <tr><td style={{ fontWeight: 600 }}>Label</td><td>Ordinal features</td><td>Simple, compact</td></tr>
                        <tr><td style={{ fontWeight: 600 }}>Target</td><td>High cardinality</td><td>Captures target relationship</td></tr>
                        <tr><td style={{ fontWeight: 600 }}>Ordinal</td><td>Natural ordering</td><td>Preserves order</td></tr>
                        <tr><td style={{ fontWeight: 600 }}>Frequency</td><td>Any categorical</td><td>Handles rare categories</td></tr>
                      </tbody>
                    </table>
                  </div>

                  <div className="cleanup-callout cleanup-callout-warning" style={{ marginTop: 12 }}>
                    <div className="cleanup-callout-icon">⚠️</div>
                    <div style={{ fontSize: 12 }}>Categorical columns remaining: <strong>{categoricalCols.length}</strong></div>
                  </div>
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

                  <button
                    className="btn btn-primary btn-sm"
                    style={{ width: '100%', marginTop: 4, justifyContent: 'center' }}
                    disabled={viewOnly || loading || binningCols.length === 0}
                    onClick={async () => {
                      const data = await run('binning', () => dataApi.binning({
                        columns: binningCols,
                        max_n_bins: binningMaxBins,
                      }));
                      if (data?.results) {
                        setBinningResults(data.results);
                        setSelectedBinningFeature(Object.keys(data.results).find((key) => !data.results[key].error) || '');
                      }
                      if (bCol) await loadDistribution(bCol);
                    }}
                  >
                    ✅ Apply Binning
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
                              <td><span className={`badge ${info.predictive_power === 'Strong' ? 'badge-success' : info.predictive_power === 'Medium' ? 'badge-warning' : 'badge-error'}`}>{info.predictive_power}</span></td>
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
                        {['Strong', 'Medium', 'Weak', 'Useless'].map(p => (
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
                          <td><span className={`badge ${info.predictive_power === 'Strong' ? 'badge-success' : info.predictive_power === 'Medium' ? 'badge-warning' : 'badge-error'}`}>{info.predictive_power}</span></td>
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
            const SCALE_DESC = {
              'StandardScaler': 'Centers to mean=0, std=1. Best for normally distributed features.',
              'MinMaxScaler': 'Scales to [0, 1] range. Sensitive to outliers.',
              'RobustScaler': 'Uses median and IQR. Robust to outliers.',
              'MaxAbsScaler': 'Scales by maximum absolute value. Preserves sparsity.',
            };
            return (
            <>
              <div className="cleanup-grid">
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">⚙️</span>
                    <span className="cleanup-panel-title">Scaling Configuration</span>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Method</label>
                    <select className="form-select" value={scaleMethod} onChange={e => setScaleMethod(e.target.value)} disabled={viewOnly}>
                      {['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler'].map(m =>
                        <option key={m} value={m}>{m}</option>
                      )}
                    </select>
                  </div>

                  <div className="cleanup-callout cleanup-callout-info" style={{ marginBottom: 12 }}>
                    <div className="cleanup-callout-icon">📝</div>
                    <div style={{ fontSize: 12 }}>{SCALE_DESC[scaleMethod] || ''}</div>
                  </div>

                  <button
                    className="btn btn-primary btn-sm"
                    style={{ width: '100%', justifyContent: 'center' }}
                    disabled={viewOnly || loading || noSplits}
                    onClick={() => run('scaling', () => dataApi.scale({ method: scaleMethod }))}
                  >
                    ✅ Scale Features
                  </button>
                </div>

                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">📊</span>
                    <span className="cleanup-panel-title">Scaling Guide</span>
                  </div>

                  <div className="table-container">
                    <table>
                      <thead><tr><th>Scaler</th><th>Formula</th><th>Best When</th></tr></thead>
                      <tbody>
                        <tr><td style={{ fontWeight: 600 }}>Standard</td><td>(x - μ) / σ</td><td>Normal distribution</td></tr>
                        <tr><td style={{ fontWeight: 600 }}>MinMax</td><td>(x - min) / (max - min)</td><td>Bounded range needed</td></tr>
                        <tr><td style={{ fontWeight: 600 }}>Robust</td><td>(x - median) / IQR</td><td>Outliers present</td></tr>
                        <tr><td style={{ fontWeight: 600 }}>MaxAbs</td><td>x / |max|</td><td>Sparse data</td></tr>
                      </tbody>
                    </table>
                  </div>

                  <div className="cleanup-callout cleanup-callout-warning" style={{ marginTop: 12 }}>
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
            const BAL_DESC = {
              'SMOTE': 'Synthetic Minority Over-sampling. Creates synthetic samples by interpolating between nearest neighbors.',
              'ADASYN': 'Adaptive Synthetic. Like SMOTE but focuses on harder-to-learn samples.',
              'SMOTE-ENN': 'SMOTE + Edited Nearest Neighbors. Over-samples then cleans noisy samples.',
              'SMOTE-Tomek': 'SMOTE + Tomek Links. Over-samples then removes borderline samples.',
              'Random Over-sampling': 'Duplicates random minority samples. Simple but risk of overfitting.',
              'Random Under-sampling': 'Removes random majority samples. Fast but loses information.',
            };
            return (
            <>
              <div className="cleanup-grid">
                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">⚙️</span>
                    <span className="cleanup-panel-title">Class Balancing</span>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Method</label>
                    <select className="form-select" value={balanceMethod} onChange={e => setBalanceMethod(e.target.value)} disabled={viewOnly}>
                      {['SMOTE', 'ADASYN', 'SMOTE-ENN', 'SMOTE-Tomek', 'Random Over-sampling', 'Random Under-sampling'].map(m =>
                        <option key={m} value={m}>{m}</option>
                      )}
                    </select>
                  </div>

                  <div className="cleanup-callout cleanup-callout-info" style={{ marginBottom: 12 }}>
                    <div className="cleanup-callout-icon">📝</div>
                    <div style={{ fontSize: 12 }}>{BAL_DESC[balanceMethod] || ''}</div>
                  </div>

                  <button
                    className="btn btn-primary btn-sm"
                    style={{ width: '100%', justifyContent: 'center' }}
                    disabled={viewOnly || loading || noSplits}
                    onClick={() => run('balance', () => dataApi.balance({ method: balanceMethod }))}
                  >
                    ✅ Balance Classes
                  </button>
                </div>

                <div className="cleanup-panel">
                  <div className="cleanup-panel-header">
                    <span className="cleanup-panel-icon">📊</span>
                    <span className="cleanup-panel-title">Balancing Guide</span>
                  </div>

                  <div className="table-container">
                    <table>
                      <thead><tr><th>Method</th><th>Type</th><th>Best When</th></tr></thead>
                      <tbody>
                        <tr><td style={{ fontWeight: 600 }}>SMOTE</td><td>Over-sampling</td><td>General imbalance</td></tr>
                        <tr><td style={{ fontWeight: 600 }}>ADASYN</td><td>Over-sampling</td><td>Complex decision boundary</td></tr>
                        <tr><td style={{ fontWeight: 600 }}>SMOTE-ENN</td><td>Combined</td><td>Noisy data</td></tr>
                        <tr><td style={{ fontWeight: 600 }}>SMOTE-Tomek</td><td>Combined</td><td>Overlapping classes</td></tr>
                        <tr><td style={{ fontWeight: 600 }}>Random Over</td><td>Over-sampling</td><td>Quick baseline</td></tr>
                        <tr><td style={{ fontWeight: 600 }}>Random Under</td><td>Under-sampling</td><td>Large datasets</td></tr>
                      </tbody>
                    </table>
                  </div>

                  <div className="cleanup-callout cleanup-callout-warning" style={{ marginTop: 12 }}>
                    <div className="cleanup-callout-icon">⚠️</div>
                    <div style={{ fontSize: 12 }}>Balancing is applied to training set only. Validation and test sets remain unchanged.</div>
                  </div>
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
                      {['Random Forest', 'Gradient Boosting', 'Mutual Information', 'Correlation'].map(m =>
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
                      const data = await run('importance', () => dataApi.featureImportance({ method: impMethod, top_n: impTopN }));
                      if (data?.importance) setImportanceResults(data.importance);
                    }}
                  >
                    ✅ Calculate Importance
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
                    <div className="table-container">
                      <table>
                        <thead><tr><th>#</th><th>Feature</th><th>Importance</th><th style={{ width: '40%' }}>Bar</th></tr></thead>
                        <tbody>
                          {Object.entries(importanceResults)
                            .sort(([, a], [, b]) => b - a)
                            .map(([feature, score], i) => (
                              <tr key={feature}>
                                <td>{i + 1}</td>
                                <td style={{ fontWeight: 600 }}>{feature}</td>
                                <td>{typeof score === 'number' ? score.toFixed(4) : score}</td>
                                <td>
                                  <div style={{
                                    height: 8, borderRadius: 4,
                                    background: `linear-gradient(90deg, var(--accent), var(--accent-hover))`,
                                    width: `${Math.min(100, (typeof score === 'number' ? score : 0) * 100 / Math.max(...Object.values(importanceResults)))}%`,
                                  }} />
                                </td>
                              </tr>
                            ))
                          }
                        </tbody>
                      </table>
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
