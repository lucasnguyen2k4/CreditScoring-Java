import { useState } from 'react';
import { modelApi } from '../api/client';

const SINGLE_MODELS = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost', 'Gradient Boosting'];
const BOOSTING_MODELS = new Set(['XGBoost', 'LightGBM', 'CatBoost', 'Gradient Boosting']);
const BASE_MODELS = [
  { code: 'LR', label: 'Logistic Regression' },
  { code: 'DT', label: 'Decision Tree' },
  { code: 'SVM', label: 'SVM' },
  { code: 'KNN', label: 'KNN' },
  { code: 'RF', label: 'Random Forest' },
  { code: 'GB', label: 'Gradient Boosting' },
];

const DEFAULT_STACK_BASE_GRIDS = {
  LR: { C: [0.1, 1.0, 10.0], max_iter: [200, 500] },
  DT: { max_depth: [5, 10, 15], min_samples_split: [2, 5, 10] },
  SVM: { C: [0.5, 1.0, 2.0], kernel: ['rbf'] },
  KNN: { n_neighbors: [3, 5, 7, 9] },
  RF: { n_estimators: [100, 200], max_depth: [5, 10, 15] },
  GB: { n_estimators: [100, 200], learning_rate: [0.05, 0.1], max_depth: [3, 5] },
};

const DEFAULT_STACK_META_PARAMS = {
  'Random Forest': { n_estimators: 200, max_depth: 12 },
  'Logistic Regression': { C: 1.0, max_iter: 300 },
  XGBoost: { n_estimators: 200, learning_rate: 0.08, max_depth: 4 },
};

function buildDefaultStackBaseGrid(selectedModels) {
  const cfg = {};
  selectedModels.forEach((m) => {
    cfg[m] = DEFAULT_STACK_BASE_GRIDS[m] || {};
  });
  return cfg;
}

function formatMetric(value) {
  if (value == null || Number.isNaN(value)) return '—';
  return Number(value).toFixed(4);
}

export default function ModelTrainingPage() {
  const [tab, setTab] = useState('train');
  const [message, setMessage] = useState('');

  const [modelType, setModelType] = useState('XGBoost');
  const [earlyStopEnabled, setEarlyStopEnabled] = useState(true);
  const [earlyStoppingRounds, setEarlyStoppingRounds] = useState(30);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const [stackBaseModels, setStackBaseModels] = useState(['LR', 'DT']);
  const [stackMetaModel, setStackMetaModel] = useState('Random Forest');
  const [stackLoading, setStackLoading] = useState(false);
  const [stackResult, setStackResult] = useState(null);

  const [cvModel, setCvModel] = useState('XGBoost');
  const [cvFolds, setCvFolds] = useState(5);
  const [cvLoading, setCvLoading] = useState(false);
  const [cvResult, setCvResult] = useState(null);
  const [tuneModel, setTuneModel] = useState('XGBoost');
  const [tuneMethod, setTuneMethod] = useState('Grid Search');
  const [tuneFolds, setTuneFolds] = useState(5);
  const [tuneTrials, setTuneTrials] = useState(50);
  const [tuneLoading, setTuneLoading] = useState(false);
  const [tuneResult, setTuneResult] = useState(null);
  const [autoTrainBest, setAutoTrainBest] = useState(false);
  const [stackTuningMethod, setStackTuningMethod] = useState('Grid Search');
  const [stackCvFolds, setStackCvFolds] = useState(5);
  const [stackBaseConfigText, setStackBaseConfigText] = useState(
    JSON.stringify(buildDefaultStackBaseGrid(['LR', 'DT']), null, 2)
  );
  const [stackMetaParamsText, setStackMetaParamsText] = useState(
    JSON.stringify(DEFAULT_STACK_META_PARAMS['Random Forest'], null, 2)
  );

  const [history, setHistory] = useState([]);
  const [selectingIndex, setSelectingIndex] = useState(null);

  const loadHistory = async () => {
    try {
      const res = await modelApi.getHistory();
      setHistory(res.data.models || []);
    } catch {
      setHistory([]);
    }
  };

  const train = async () => {
    setLoading(true);
    setMessage('');
    try {
      const res = await modelApi.train({
        model_type: modelType,
        early_stopping_rounds: (BOOSTING_MODELS.has(modelType) && earlyStopEnabled) ? earlyStoppingRounds : null,
      });
      setResult(res.data);
      setMessage(res.data.message);
      await loadHistory();
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const toggleBaseModel = (code) => {
    setStackBaseModels((prev) => {
      if (prev.includes(code)) {
        if (prev.length <= 2) return prev;
        return prev.filter((m) => m !== code);
      }
      return [...prev, code];
    });
  };

  const loadStackingDefaults = () => {
    setStackBaseConfigText(
      JSON.stringify(buildDefaultStackBaseGrid(stackBaseModels), null, 2)
    );
    setStackMetaParamsText(
      JSON.stringify(DEFAULT_STACK_META_PARAMS[stackMetaModel] || {}, null, 2)
    );
  };

  const trainStacking = async () => {
    if (stackBaseModels.length < 2) {
      setMessage('Error: Stacking requires at least 2 base models.');
      return;
    }
    setStackLoading(true);
    setMessage('');
    try {
      const res = await modelApi.trainStacking({
        base_models: stackBaseModels,
        meta_model: stackMetaModel,
      });
      setStackResult(res.data);
      setMessage(res.data.message);
      await loadHistory();
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setStackLoading(false);
    }
  };

  const tuneStacking = async () => {
    if (stackBaseModels.length < 2) {
      setMessage('Error: Stacking requires at least 2 base models.');
      return;
    }
    let baseModelsConfig = {};
    let metaModelParams = {};
    try {
      baseModelsConfig = stackBaseConfigText.trim() ? JSON.parse(stackBaseConfigText) : {};
      metaModelParams = stackMetaParamsText.trim() ? JSON.parse(stackMetaParamsText) : {};
      if (typeof baseModelsConfig !== 'object' || baseModelsConfig === null || Array.isArray(baseModelsConfig)) {
        throw new Error('Base config must be a JSON object');
      }
      if (typeof metaModelParams !== 'object' || metaModelParams === null || Array.isArray(metaModelParams)) {
        throw new Error('Meta params must be a JSON object');
      }
    } catch (err) {
      setMessage(`Error: Invalid JSON config for stacking tuning (${err.message})`);
      return;
    }
    setStackLoading(true);
    setMessage('');
    try {
      const res = await modelApi.tuneStacking({
        base_models: stackBaseModels,
        meta_model: stackMetaModel,
        tuning_method: stackTuningMethod,
        cv_folds: stackCvFolds,
        base_models_config: baseModelsConfig,
        meta_model_params: metaModelParams,
      });
      setStackResult(res.data);
      setMessage(res.data.message);
      await loadHistory();
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setStackLoading(false);
    }
  };

  const runCrossValidation = async () => {
    setCvLoading(true);
    setMessage('');
    try {
      const res = await modelApi.crossValidate({
        model_type: cvModel,
        cv_folds: cvFolds,
      });
      setCvResult(res.data);
      setMessage(`Cross-validation completed for ${cvModel}`);
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setCvLoading(false);
    }
  };

  const runTuning = async () => {
    setTuneLoading(true);
    setMessage('');
    try {
      const res = await modelApi.tune({
        model_type: tuneModel,
        method: tuneMethod,
        cv_folds: tuneFolds,
        n_trials: tuneTrials,
        auto_train_best: autoTrainBest,
        early_stopping_rounds: 30,
      });
      setTuneResult(res.data);
      setMessage(res.data.message);
      if (res.data.trained_model_index != null) {
        await loadHistory();
      }
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setTuneLoading(false);
    }
  };

  const selectModel = async (index) => {
    setSelectingIndex(index);
    setMessage('');
    try {
      const res = await modelApi.selectModel(index);
      setMessage(res.data.message);
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setSelectingIndex(null);
    }
  };

  const renderResult = (metrics) => {
    if (!metrics) return null;
    const trainMetrics = metrics.train_metrics;
    const validMetrics = metrics.valid_metrics;
    const testMetrics = metrics.test_metrics;
    const confusion = metrics.confusion_matrix || [];
    const matrixRows = Array.isArray(confusion) && confusion.length === 2 ? confusion : null;

    return (
      <div style={{ marginTop: 20 }}>
        <div className="summary-grid" style={{ marginBottom: 16 }}>
          <div className="summary-card"><div className="summary-label">Accuracy</div><div className="summary-value">{formatMetric(metrics.accuracy)}</div></div>
          <div className="summary-card"><div className="summary-label">Precision</div><div className="summary-value">{formatMetric(metrics.precision)}</div></div>
          <div className="summary-card"><div className="summary-label">Recall</div><div className="summary-value">{formatMetric(metrics.recall)}</div></div>
          <div className="summary-card"><div className="summary-label">F1</div><div className="summary-value">{formatMetric(metrics.f1)}</div></div>
          <div className="summary-card"><div className="summary-label">AUC</div><div className="summary-value">{formatMetric(metrics.auc)}</div></div>
        </div>

        {(trainMetrics || validMetrics || testMetrics) && (
          <div className="table-container" style={{ marginBottom: 16 }}>
            <table>
              <thead>
                <tr><th>Dataset</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr>
              </thead>
              <tbody>
                {trainMetrics && <tr><td style={{ fontWeight: 700 }}>Train</td><td>{formatMetric(trainMetrics.accuracy)}</td><td>{formatMetric(trainMetrics.precision)}</td><td>{formatMetric(trainMetrics.recall)}</td><td>{formatMetric(trainMetrics.f1)}</td><td>{formatMetric(trainMetrics.auc)}</td></tr>}
                {validMetrics && <tr><td style={{ fontWeight: 700 }}>Validation</td><td>{formatMetric(validMetrics.accuracy)}</td><td>{formatMetric(validMetrics.precision)}</td><td>{formatMetric(validMetrics.recall)}</td><td>{formatMetric(validMetrics.f1)}</td><td>{formatMetric(validMetrics.auc)}</td></tr>}
                {testMetrics && <tr><td style={{ fontWeight: 700 }}>Test</td><td>{formatMetric(testMetrics.accuracy)}</td><td>{formatMetric(testMetrics.precision)}</td><td>{formatMetric(testMetrics.recall)}</td><td>{formatMetric(testMetrics.f1)}</td><td>{formatMetric(testMetrics.auc)}</td></tr>}
              </tbody>
            </table>
          </div>
        )}

        {matrixRows && (
          <div className="table-container">
            <table>
              <thead>
                <tr><th>Confusion Matrix</th><th>Pred 0</th><th>Pred 1</th></tr>
              </thead>
              <tbody>
                <tr><td style={{ fontWeight: 700 }}>True 0</td><td>{matrixRows[0][0]}</td><td>{matrixRows[0][1]}</td></tr>
                <tr><td style={{ fontWeight: 700 }}>True 1</td><td>{matrixRows[1][0]}</td><td>{matrixRows[1][1]}</td></tr>
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Model Training</h1>
        <p className="page-description">Single model, tuning, stacking, cross-validation, and model history</p>
      </div>

      {message && <div className={`alert ${message.startsWith('Error') ? 'alert-error' : 'alert-success'}`}>{message}</div>}

      <div className="tabs">
        <button className={`tab ${tab === 'train' ? 'active' : ''}`} onClick={() => setTab('train')}>Single Model</button>
        <button className={`tab ${tab === 'tuning' ? 'active' : ''}`} onClick={() => setTab('tuning')}>Tuning</button>
        <button className={`tab ${tab === 'stacking' ? 'active' : ''}`} onClick={() => setTab('stacking')}>Stacking</button>
        <button className={`tab ${tab === 'cv' ? 'active' : ''}`} onClick={() => setTab('cv')}>Cross-Validation</button>
        <button className={`tab ${tab === 'history' ? 'active' : ''}`} onClick={() => { setTab('history'); loadHistory(); }}>History</button>
      </div>

      {tab === 'train' && (
        <div className="card">
          <div className="card-title" style={{ marginBottom: 12 }}>Single Model Training</div>
          <div className="flex gap-md" style={{ flexWrap: 'wrap', marginBottom: 16 }}>
            {SINGLE_MODELS.map((m) => (
              <button key={m} className={`btn ${modelType === m ? 'btn-primary' : 'btn-secondary'} btn-sm`} onClick={() => setModelType(m)}>
                {m}
              </button>
            ))}
          </div>

          {BOOSTING_MODELS.has(modelType) && (
            <div className="card" style={{ marginBottom: 14 }}>
              <div className="flex items-center gap-md" style={{ flexWrap: 'wrap' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input type="checkbox" checked={earlyStopEnabled} onChange={(e) => setEarlyStopEnabled(e.target.checked)} />
                  Enable Early Stopping
                </label>
                <div className="form-group" style={{ margin: 0, maxWidth: 180 }}>
                  <label className="form-label">Rounds</label>
                  <input
                    className="form-input"
                    type="number"
                    min="5"
                    max="200"
                    step="1"
                    value={earlyStoppingRounds}
                    onChange={(e) => setEarlyStoppingRounds(+e.target.value)}
                    disabled={!earlyStopEnabled}
                  />
                </div>
              </div>
            </div>
          )}

          <button className="btn btn-primary" onClick={train} disabled={loading}>
            {loading ? <><div className="spinner" /> Training...</> : 'Train Model'}
          </button>

          {renderResult(result?.metrics)}
        </div>
      )}

      {tab === 'tuning' && (
        <div className="card">
          <div className="card-title" style={{ marginBottom: 12 }}>Hyperparameter Tuning</div>
          <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
            <div className="form-group" style={{ flex: 1, minWidth: 220 }}>
              <label className="form-label">Model</label>
              <select className="form-select" value={tuneModel} onChange={(e) => setTuneModel(e.target.value)}>
                {SINGLE_MODELS.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <div className="form-group" style={{ flex: 1, minWidth: 220 }}>
              <label className="form-label">Method</label>
              <select className="form-select" value={tuneMethod} onChange={(e) => setTuneMethod(e.target.value)}>
                {['Grid Search', 'Random Search', 'Optuna'].map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <div className="form-group" style={{ width: 140 }}>
              <label className="form-label">CV Folds</label>
              <input className="form-input" type="number" min="3" max="10" value={tuneFolds} onChange={(e) => setTuneFolds(+e.target.value)} />
            </div>
            <div className="form-group" style={{ width: 160 }}>
              <label className="form-label">Trials (Optuna)</label>
              <input className="form-input" type="number" min="10" max="200" value={tuneTrials} onChange={(e) => setTuneTrials(+e.target.value)} />
            </div>
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '6px 0 14px' }}>
            <input type="checkbox" checked={autoTrainBest} onChange={(e) => setAutoTrainBest(e.target.checked)} />
            Auto-train model with best params
          </label>

          <button className="btn btn-primary" onClick={runTuning} disabled={tuneLoading}>
            {tuneLoading ? <><div className="spinner" /> Tuning...</> : 'Run Tuning'}
          </button>

          {tuneResult && (
            <div style={{ marginTop: 16 }}>
              <div className="summary-grid" style={{ marginBottom: 16 }}>
                <div className="summary-card"><div className="summary-label">Method</div><div className="summary-value" style={{ fontSize: 14 }}>{tuneResult.method || tuneMethod}</div></div>
                <div className="summary-card"><div className="summary-label">Best AUC</div><div className="summary-value">{formatMetric(tuneResult.best_score)}</div></div>
                <div className="summary-card"><div className="summary-label">Total Fits</div><div className="summary-value">{tuneResult.total_fits ?? '—'}</div></div>
                <div className="summary-card"><div className="summary-label">Auto-trained</div><div className="summary-value" style={{ fontSize: 14 }}>{tuneResult.trained_model_index != null ? 'Yes' : 'No'}</div></div>
              </div>

              <div className="table-container" style={{ marginBottom: 14 }}>
                <table>
                  <thead><tr><th>Best Params</th><th>Value</th></tr></thead>
                  <tbody>
                    {Object.entries(tuneResult.best_params || {}).map(([k, v]) => (
                      <tr key={k}><td style={{ fontWeight: 700 }}>{k}</td><td>{String(v)}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {(tuneResult.top_results || []).length > 0 && (
                <div className="table-container">
                  <table>
                    <thead><tr><th>#</th><th>Mean AUC</th><th>Std</th><th>Params</th></tr></thead>
                    <tbody>
                      {tuneResult.top_results.map((r, idx) => (
                        <tr key={idx}>
                          <td>{idx + 1}</td>
                          <td>{formatMetric(r.mean_test_score)}</td>
                          <td>{formatMetric(r.std_test_score)}</td>
                          <td style={{ fontSize: 12 }}>{JSON.stringify(r.params)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {tuneResult.trained_metrics && renderResult(tuneResult.trained_metrics)}
            </div>
          )}
        </div>
      )}

      {tab === 'stacking' && (
        <div className="card">
          <div className="card-title" style={{ marginBottom: 12 }}>Stacking Ensemble</div>
          <p style={{ color: 'var(--text-muted)', marginBottom: 12 }}>Select at least 2 base models, then choose a meta model.</p>

          <div className="flex gap-sm" style={{ flexWrap: 'wrap', marginBottom: 16 }}>
            {BASE_MODELS.map((m) => (
              <button
                key={m.code}
                className={`btn ${stackBaseModels.includes(m.code) ? 'btn-primary' : 'btn-secondary'} btn-sm`}
                onClick={() => toggleBaseModel(m.code)}
              >
                {m.code}
              </button>
            ))}
          </div>

          <div className="form-group" style={{ maxWidth: 280 }}>
            <label className="form-label">Meta Model</label>
            <select className="form-select" value={stackMetaModel} onChange={(e) => setStackMetaModel(e.target.value)}>
              <option>Random Forest</option>
              <option>Logistic Regression</option>
              <option>XGBoost</option>
            </select>
          </div>
          <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
            <div className="form-group" style={{ maxWidth: 220 }}>
              <label className="form-label">Tuning Method</label>
              <select className="form-select" value={stackTuningMethod} onChange={(e) => setStackTuningMethod(e.target.value)}>
                <option>Grid Search</option>
                <option>Random Search</option>
                <option>Default</option>
              </select>
            </div>
            <div className="form-group" style={{ width: 140 }}>
              <label className="form-label">OOF Folds</label>
              <input className="form-input" type="number" min="3" max="10" value={stackCvFolds} onChange={(e) => setStackCvFolds(+e.target.value)} />
            </div>
          </div>

          <div className="form-group" style={{ marginTop: 8 }}>
            <label className="form-label">Base Models Param Grid (JSON)</label>
            <textarea
              className="form-input"
              rows={8}
              value={stackBaseConfigText}
              onChange={(e) => setStackBaseConfigText(e.target.value)}
              placeholder='{"LR":{"C":[0.1,1,10]},"RF":{"n_estimators":[100,200]}}'
              style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace' }}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Meta Model Params (JSON)</label>
            <textarea
              className="form-input"
              rows={4}
              value={stackMetaParamsText}
              onChange={(e) => setStackMetaParamsText(e.target.value)}
              placeholder='{"n_estimators":200,"max_depth":8}'
              style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace' }}
            />
          </div>

          <div className="flex gap-sm" style={{ marginTop: 8 }}>
            <button className="btn btn-primary" onClick={trainStacking} disabled={stackLoading || stackBaseModels.length < 2}>
              {stackLoading ? <><div className="spinner" /> Training Stacking...</> : 'Train Stacking'}
            </button>
            <button className="btn btn-secondary" onClick={tuneStacking} disabled={stackLoading || stackBaseModels.length < 2}>
              {stackLoading ? <><div className="spinner" /> Tuning...</> : 'Tune + Train (OOF)'}
            </button>
            <button className="btn btn-secondary" onClick={loadStackingDefaults} disabled={stackLoading}>
              Load Default Grids
            </button>
          </div>

          {renderResult(stackResult?.metrics)}
        </div>
      )}

      {tab === 'cv' && (
        <div className="card">
          <div className="card-title" style={{ marginBottom: 12 }}>Cross-Validation</div>
          <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
            <div className="form-group" style={{ flex: 1, minWidth: 220 }}>
              <label className="form-label">Model</label>
              <select className="form-select" value={cvModel} onChange={(e) => setCvModel(e.target.value)}>
                {SINGLE_MODELS.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <div className="form-group" style={{ width: 140 }}>
              <label className="form-label">Folds</label>
              <input className="form-input" type="number" min="3" max="10" value={cvFolds} onChange={(e) => setCvFolds(+e.target.value)} />
            </div>
          </div>

          <button className="btn btn-primary" onClick={runCrossValidation} disabled={cvLoading}>
            {cvLoading ? <><div className="spinner" /> Running CV...</> : 'Run Cross-Validation'}
          </button>

          {cvResult && (
            <div className="table-container" style={{ marginTop: 16 }}>
              <table>
                <thead><tr><th>Metric</th><th>Mean</th><th>Std</th></tr></thead>
                <tbody>
                  {['accuracy', 'precision', 'recall', 'f1', 'auc'].map((m) => (
                    <tr key={m}>
                      <td style={{ fontWeight: 700 }}>{m.toUpperCase()}</td>
                      <td>{formatMetric(cvResult?.[m]?.mean)}</td>
                      <td>{formatMetric(cvResult?.[m]?.std)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {tab === 'history' && (
        <div className="card">
          {history.length === 0 ? (
            <div className="empty-state">No models trained yet</div>
          ) : (
            <div className="table-container">
              <table>
                <thead><tr><th>#</th><th>Model</th><th>Accuracy</th><th>AUC</th><th>F1</th><th>Timestamp</th><th>Action</th></tr></thead>
                <tbody>
                  {history.map((m, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td style={{ fontWeight: 600 }}>
                        {m.model_type}
                        {m.is_approved && <span className="badge badge-success" style={{ marginLeft: 8 }}>Approved</span>}
                      </td>
                      <td>{formatMetric(m.metrics?.accuracy)}</td>
                      <td>{formatMetric(m.metrics?.auc || m.metrics?.auc_roc || m.metrics?.roc_auc)}</td>
                      <td>{formatMetric(m.metrics?.f1 || m.metrics?.f1_score)}</td>
                      <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>{m.timestamp?.substring(0, 19) || '—'}</td>
                      <td>
                        <button className="btn btn-secondary btn-sm" onClick={() => selectModel(m.index ?? i)} disabled={selectingIndex === (m.index ?? i)}>
                          {selectingIndex === (m.index ?? i) ? 'Selecting...' : 'Set Active'}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
