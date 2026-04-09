import { useState, useEffect } from 'react';
import { predictApi } from '../api/client';

export default function PredictionPage() {
  const [features, setFeatures] = useState([]);
  const [inputs, setInputs] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    predictApi.getFeatures().then(res => {
      setFeatures(res.data.features || []);
      const defaults = {};
      (res.data.features || []).forEach(f => { defaults[f.name] = f.mean ?? 0; });
      setInputs(defaults);
    }).catch(() => {});
  }, []);

  const predict = async () => {
    setLoading(true); setMessage(''); setResult(null);
    try {
      const res = await predictApi.single(inputs);
      setResult(res.data);
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    }
    setLoading(false);
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Credit Score Prediction</h1>
        <p className="page-description">Enter customer data to get a credit score prediction</p>
      </div>

      {message && <div className="alert alert-error">{message}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        {/* Input Form */}
        <div className="card">
          <div className="card-title" style={{ marginBottom: 16 }}>Customer Data</div>

          {features.length === 0 ? (
            <div className="empty-state">
              <p>No model trained yet. Train a model first to enable prediction.</p>
            </div>
          ) : (
            <>
              <div style={{ maxHeight: 500, overflowY: 'auto', marginBottom: 16 }}>
                {features.map(f => (
                  <div className="form-group" key={f.name}>
                    <label className="form-label" style={{ fontSize: 11 }}>{f.name}</label>
                    <input className="form-input" type="number" step="any"
                      value={inputs[f.name] ?? ''}
                      onChange={e => setInputs({ ...inputs, [f.name]: +e.target.value })}
                      placeholder={`Range: ${f.min?.toFixed(2)} - ${f.max?.toFixed(2)}`}
                    />
                  </div>
                ))}
              </div>
              <button className="btn btn-primary" onClick={predict} disabled={loading} style={{ width: '100%', justifyContent: 'center' }}>
                {loading ? <><div className="spinner" /> Predicting...</> : 'Predict Credit Score'}
              </button>
            </>
          )}
        </div>

        {/* Result */}
        <div className="card">
          <div className="card-title" style={{ marginBottom: 16 }}>Prediction Result</div>

          {!result ? (
            <div className="empty-state">
              <p>Fill in customer data and click Predict</p>
            </div>
          ) : (
            <div>
              {/* Score */}
              <div style={{
                textAlign: 'center', padding: 32, background: 'var(--bg-input)',
                borderRadius: 'var(--radius)', marginBottom: 20,
              }}>
                <div style={{ fontSize: 14, color: 'var(--text-muted)', marginBottom: 4 }}>Credit Score</div>
                <div style={{
                  fontSize: 48, fontWeight: 800,
                  color: (result.credit_score || 0) >= 700 ? 'var(--success)' : (result.credit_score || 0) >= 500 ? 'var(--warning)' : 'var(--error)',
                }}>
                  {result.credit_score || result.prediction}
                </div>
                <div style={{ marginTop: 8 }}>
                  <span className={`badge ${result.prediction === 0 ? 'badge-success' : 'badge-error'}`}>
                    {result.risk_level || (result.prediction === 0 ? 'Low Risk' : 'High Risk')}
                  </span>
                </div>
                {result.probability != null && (
                  <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 8 }}>
                    Default Probability: {(result.probability * 100).toFixed(2)}%
                  </div>
                )}
              </div>

              {/* Top contributions */}
              {result.contributions?.length > 0 && (
                <div>
                  <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 8 }}>Top Contributing Features</div>
                  {result.contributions.map(([feat, val], i) => (
                    <div key={i} className="flex justify-between items-center" style={{
                      padding: '8px 0', borderBottom: '1px solid var(--border)',
                    }}>
                      <span style={{ fontSize: 13 }}>{feat}</span>
                      <span className={`badge ${val > 0 ? 'badge-error' : 'badge-success'}`} style={{ fontSize: 11 }}>
                        {val > 0 ? '+' : ''}{val.toFixed(4)}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Recommendations */}
              {result.recommendations?.length > 0 && (
                <div style={{ marginTop: 20 }}>
                  <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 8 }}>Recommendations</div>
                  {result.recommendations.map((rec, i) => (
                    <div key={i} style={{
                      padding: 10, background: 'var(--bg-input)',
                      borderRadius: 'var(--radius-sm)', marginBottom: 6, fontSize: 13,
                    }}>• {rec}</div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
