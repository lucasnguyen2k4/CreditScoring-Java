import { useState, useEffect } from 'react';
import { dataApi } from '../api/client';
import { isViewOnly } from '../utils/permissions';
import { useAuth } from '../context/AuthContext';
import ColumnMultiSelector from '../components/ColumnMultiSelector';
import DataDistributionCard from '../components/charts/DataDistributionCard';
import ComparisonMetricChart from '../components/charts/ComparisonMetricChart';
import MetricRankingChart from '../components/charts/MetricRankingChart';
import WoeBinsChart from '../components/charts/WoeBinsChart';

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

  const [missingMethod, setMissingMethod] = useState('Mean Imputation');
  const [missingCols, setMissingCols] = useState([]);

  const [outlierMethod, setOutlierMethod] = useState('Winsorization');
  const [outlierCols, setOutlierCols] = useState([]);

  const [skewMethod, setSkewMethod] = useState('Log');
  const [skewCols, setSkewCols] = useState([]);

  const [encMethod, setEncMethod] = useState('Label Encoding');
  const [encCols, setEncCols] = useState([]);

  const [binningCols, setBinningCols] = useState([]);
  const [binningMaxBins, setBinningMaxBins] = useState(10);

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
            <>
              <div className="flex gap-md" style={{ flexWrap: 'wrap', marginBottom: 12 }}>
                <button
                  className="btn btn-secondary btn-sm"
                  disabled={viewOnly || loading || noSplits}
                  onClick={() => run('cleanup', () => dataApi.removeCategorical(false, true))}
                >
                  Remove Categorical Columns
                </button>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', alignSelf: 'center' }}>
                  Runs on split data only (train/valid/test), not EDA raw data.
                </div>
              </div>

              <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
                <div className="form-group" style={{ flex: 1, minWidth: 220 }}>
                  <label className="form-label">Invalid Number Strategy</label>
                  <select className="form-select" value={cleanupStrategy} onChange={e => setCleanupStrategy(e.target.value)} disabled={viewOnly || noSplits}>
                    <option value="drop_rows">Drop rows with invalid values</option>
                    <option value="fill_median">Convert invalid to NaN and fill train median</option>
                  </select>
                </div>
                <ColumnMultiSelector
                  label="Columns (optional)"
                  columns={numericCols}
                  values={cleanupCols}
                  onChange={setCleanupCols}
                  disabled={viewOnly || noSplits}
                  placeholder="Type or choose numeric columns (empty = all numeric split columns)"
                  showSelectAll={numericCols.length > 0}
                />
              </div>
              <button
                className="btn btn-primary btn-sm"
                disabled={viewOnly || loading || noSplits}
                onClick={() => run('cleanup', () => dataApi.cleanInvalidNumbers({
                  columns: cleanupCols.length ? cleanupCols : null,
                  strategy: cleanupStrategy,
                  processed: false,
                  apply_on_splits: true,
                }))}
              >
                Clean Invalid Numbers
              </button>
            </>
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

          {/* Step 3: Outliers */}
          {key === 'outliers' && (
            <>
              <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
                <div className="form-group" style={{ flex: 1, minWidth: 200 }}>
                  <label className="form-label">Method</label>
                  <select className="form-select" value={outlierMethod} onChange={e => setOutlierMethod(e.target.value)} disabled={viewOnly}>
                    {['Winsorization', 'IQR Method', 'Z-Score', 'Keep All'].map(m =>
                      <option key={m} value={m}>{m}</option>
                    )}
                  </select>
                </div>
                <ColumnMultiSelector
                  label="Columns"
                  columns={numericCols}
                  values={outlierCols}
                  onChange={setOutlierCols}
                  disabled={viewOnly}
                  placeholder="Type or choose numeric columns"
                />
              </div>
              <button className="btn btn-primary btn-sm" disabled={viewOnly || loading || outlierCols.length === 0}
                onClick={async () => {
                  const data = await run('outliers', () => dataApi.handleOutliers({
                    method: outlierMethod,
                    columns: outlierCols,
                  }));
                  if (data?.info) setOutlierResults(data.info);
                  if (distributionColumn) await loadDistribution(distributionColumn);
                }}>
                Handle Outliers
              </button>
              {outlierChartRows.length > 0 && (
                <MetricRankingChart
                  title="Detected Outliers by Column"
                  data={outlierChartRows}
                  valueKey="count"
                  color="#f59e0b"
                />
              )}
              {distributionColumn && (
                <DataDistributionCard
                  title="Current Distribution"
                  description="Inspect the processed numeric column while tuning outlier handling."
                  columns={numericCols}
                  selectedColumn={distributionColumn}
                  onColumnChange={setDistributionColumn}
                  distribution={distributionData}
                  loading={distributionLoading}
                  processed
                />
              )}
            </>
          )}

          {/* Step 4: Skewness / Distribution Transform */}
          {key === 'skewness' && (
            <>
              <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
                <div className="form-group" style={{ flex: 1, minWidth: 200 }}>
                  <label className="form-label">Transform Method</label>
                  <select className="form-select" value={skewMethod} onChange={e => setSkewMethod(e.target.value)} disabled={viewOnly}>
                    {['Log', 'Sqrt', 'Box-Cox', 'Yeo-Johnson', 'Reciprocal'].map(m =>
                      <option key={m} value={m}>{m}</option>
                    )}
                  </select>
                </div>
                <ColumnMultiSelector
                  label="Columns"
                  columns={numericCols}
                  values={skewCols}
                  onChange={setSkewCols}
                  disabled={viewOnly}
                  placeholder="Type or choose numeric columns"
                />
              </div>
              <button className="btn btn-primary btn-sm" disabled={viewOnly || loading || skewCols.length === 0}
                onClick={async () => {
                  const data = await run('skewness', () => dataApi.transformSkewness({
                    method: skewMethod,
                    columns: skewCols,
                  }));
                  if (data?.results) setSkewnessResults(data.results);
                  if (distributionColumn) await loadDistribution(distributionColumn);
                }}>
                Apply Transform
              </button>
              {skewnessResults && (
                <>
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
                              <td>{info.improved ? 'Yes' : 'No'}</td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                  {distributionColumn && (
                    <DataDistributionCard
                      title="Distribution Visualization"
                      description="Use the live histogram to see how the selected feature changes after transformation."
                      columns={numericCols}
                      selectedColumn={distributionColumn}
                      onColumnChange={setDistributionColumn}
                      distribution={distributionData}
                      loading={distributionLoading}
                      processed
                    />
                  )}
                </>
              )}
            </>
          )}

          {/* Step 5: Encoding */}
          {key === 'encoding' && (
            <>
              <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
                <div className="form-group" style={{ flex: 1, minWidth: 200 }}>
                  <label className="form-label">Method</label>
                  <select className="form-select" value={encMethod} onChange={e => setEncMethod(e.target.value)} disabled={viewOnly}>
                    {['One-Hot Encoding', 'Label Encoding', 'Target Encoding', 'Ordinal Encoding', 'Frequency Encoding'].map(m =>
                      <option key={m} value={m}>{m}</option>
                    )}
                  </select>
                </div>
                <ColumnMultiSelector
                  label="Columns"
                  columns={categoricalCols}
                  values={encCols}
                  onChange={setEncCols}
                  disabled={viewOnly}
                  placeholder="Type or choose categorical columns"
                />
              </div>
              <button className="btn btn-primary btn-sm" disabled={viewOnly || loading || encCols.length === 0}
                onClick={() => run('encoding', () => dataApi.encode({
                  method: encMethod,
                  columns: encCols,
                }))}>
                Apply Encoding
              </button>
            </>
          )}

          {/* Step 5: WoE/IV Binning */}
          {key === 'binning' && (
            <>
              <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
                <ColumnMultiSelector
                  label="Columns"
                  columns={numericCols}
                  values={binningCols}
                  onChange={setBinningCols}
                  disabled={viewOnly}
                  placeholder="Type or choose numeric columns"
                />
                <div className="form-group" style={{ flex: 1, minWidth: 120 }}>
                  <label className="form-label">Max Bins</label>
                  <input className="form-input" type="number" min="3" max="20" step="1"
                    value={binningMaxBins} onChange={e => setBinningMaxBins(+e.target.value)} disabled={viewOnly} />
                </div>
              </div>
              <button className="btn btn-primary btn-sm" disabled={viewOnly || loading || binningCols.length === 0}
                onClick={async () => {
                  const data = await run('binning', () => dataApi.binning({
                    columns: binningCols,
                    max_n_bins: binningMaxBins,
                  }));
                  if (data?.results) {
                    setBinningResults(data.results);
                    setSelectedBinningFeature(Object.keys(data.results).find((key) => !data.results[key].error) || '');
                  }
                  if (distributionColumn) await loadDistribution(distributionColumn);
                }}>
                Apply WoE Binning
              </button>
              {binningResults && (
                <>
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
                </>
              )}
            </>
          )}

          {/* Step 7: WoE Analysis */}
          {key === 'woe_analysis' && (
            <>
              <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 12 }}>
                Analyzes all numeric features against the target column. Does NOT transform data — use Step 6 (Binning) to apply WoE transformation.
              </p>
              <button className="btn btn-primary btn-sm" disabled={viewOnly || loading}
                onClick={async () => {
                  const data = await run('woe_analysis', () => dataApi.woeAnalysis({}));
                  if (data?.results) {
                    setWoeResults(data);
                    setSelectedWoeFeature(Object.keys(data.results)[0] || '');
                  }
                }}>
                Run WoE Analysis
              </button>
              {woeResults && (
                <>
                  <MetricRankingChart title="WoE Analysis IV Ranking" data={woeChartData} valueKey="value" color="#10b981" />
                  <div style={{ display: 'flex', gap: 16, marginTop: 16, flexWrap: 'wrap' }}>
                    {['Strong', 'Medium', 'Weak', 'Useless'].map(p => (
                      <div key={p} style={{ padding: '8px 16px', borderRadius: 8, background: 'var(--surface-hover)', textAlign: 'center' }}>
                        <div style={{ fontSize: 20, fontWeight: 700 }}>{woeResults.summary?.[p.toLowerCase()] || 0}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{p}</div>
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
                  <WoeBinsChart title="WoE Analysis by Bin" data={selectedWoeBins} />
                  <div className="table-container" style={{ marginTop: 12 }}>
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
                </>
              )}
            </>
          )}

          {/* Step 8: Multicollinearity */}
          {key === 'multicollinearity' && (
            <>
              <div className="flex gap-md items-center" style={{ flexWrap: 'wrap', marginBottom: 12 }}>
                <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <input type="checkbox" checked={autoRemoveVif} onChange={e => setAutoRemoveVif(e.target.checked)} disabled={viewOnly} />
                  Auto-remove features with high VIF
                </label>
              </div>
              <button className="btn btn-primary btn-sm" disabled={viewOnly || loading}
                onClick={async () => {
                  const data = await run('multicollinearity', () => dataApi.multicollinearity({
                    vif_threshold: 10.0,
                    corr_threshold: 0.8,
                    auto_remove: autoRemoveVif,
                  }));
                  if (data?.vif) setVifResults(data.vif);
                  if (data?.high_correlation_pairs) setCorrPairs(data.high_correlation_pairs);
                }}>
                Check Multicollinearity
              </button>
              {vifResults && (
                <div className="table-container" style={{ marginTop: 16 }}>
                  <div className="card-title" style={{ fontSize: 13, marginBottom: 8 }}>VIF (Variance Inflation Factor)</div>
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
              )}
              {corrPairs && corrPairs.length > 0 && (
                <div className="table-container" style={{ marginTop: 12 }}>
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

          {/* Step 9: Scaling */}
          {key === 'scaling' && (
            <div className="flex gap-md items-center">
              <div className="form-group" style={{ flex: 1, margin: 0 }}>
                <label className="form-label">Method</label>
                <select className="form-select" value={scaleMethod} onChange={e => setScaleMethod(e.target.value)} disabled={viewOnly}>
                  {['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler'].map(m =>
                    <option key={m} value={m}>{m}</option>
                  )}
                </select>
              </div>
              <button className="btn btn-primary btn-sm" style={{ marginTop: 18 }} disabled={viewOnly || loading || noSplits}
                onClick={() => run('scaling', () => dataApi.scale({ method: scaleMethod }))}>
                Scale Features
              </button>
            </div>
          )}

          {/* Step 6: Balance */}
          {key === 'balance' && (
            <div className="flex gap-md items-center">
              <div className="form-group" style={{ flex: 1, margin: 0 }}>
                <label className="form-label">Method</label>
                <select className="form-select" value={balanceMethod} onChange={e => setBalanceMethod(e.target.value)} disabled={viewOnly}>
                  {['SMOTE', 'ADASYN', 'SMOTE-ENN', 'SMOTE-Tomek', 'Random Over-sampling', 'Random Under-sampling'].map(m =>
                    <option key={m} value={m}>{m}</option>
                  )}
                </select>
              </div>
              <button className="btn btn-primary btn-sm" style={{ marginTop: 18 }} disabled={viewOnly || loading || noSplits}
                onClick={() => run('balance', () => dataApi.balance({ method: balanceMethod }))}>
                Balance Classes
              </button>
            </div>
          )}

          {/* Step 7: Feature Importance */}
          {key === 'importance' && (
            <>
              <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
                <div className="form-group" style={{ flex: 1, minWidth: 200 }}>
                  <label className="form-label">Method</label>
                  <select className="form-select" value={impMethod} onChange={e => setImpMethod(e.target.value)} disabled={viewOnly}>
                    {['Random Forest', 'Gradient Boosting', 'Mutual Information', 'Correlation'].map(m =>
                      <option key={m} value={m}>{m}</option>
                    )}
                  </select>
                </div>
                <div className="form-group" style={{ flex: 1, minWidth: 150 }}>
                  <label className="form-label">Top N Features</label>
                  <input className="form-input" type="number" min="5" max="50" step="1"
                    value={impTopN} onChange={e => setImpTopN(+e.target.value)} disabled={viewOnly} />
                </div>
              </div>
              <button className="btn btn-primary btn-sm" disabled={viewOnly || loading || noSplits}
                onClick={async () => {
                  const data = await run('importance', () => dataApi.featureImportance({ method: impMethod, top_n: impTopN }));
                  if (data?.importance) setImportanceResults(data.importance);
                }}>
                Calculate Importance
              </button>
              {importanceResults && (
                <div className="table-container" style={{ marginTop: 16 }}>
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
