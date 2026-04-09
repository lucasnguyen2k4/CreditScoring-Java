import { useEffect, useState } from 'react';
import { modelApi } from '../api/client';
import { isViewOnly } from '../utils/permissions';
import { useAuth } from '../context/AuthContext';

function formatMetric(value) {
  if (value == null || Number.isNaN(value)) return '—';
  return Number(value).toFixed(4);
}

export default function ModelApprovalPage() {
  const { user } = useAuth();
  const viewOnly = isViewOnly(user?.role, '/model-approval');
  const [history, setHistory] = useState([]);
  const [approvals, setApprovals] = useState([]);
  const [approvedModel, setApprovedModel] = useState(null);
  const [approvedModelIndex, setApprovedModelIndex] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [decision, setDecision] = useState('approved');
  const [notes, setNotes] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const loadData = async () => {
    try {
      const [historyRes, approvalsRes] = await Promise.all([
        modelApi.getHistory(),
        modelApi.getApprovals(),
      ]);
      const models = historyRes.data.models || [];
      setHistory(models);
      setApprovals(approvalsRes.data.approvals || []);
      setApprovedModel(approvalsRes.data.approved_model || null);
      setApprovedModelIndex(approvalsRes.data.approved_model_index ?? null);
      if (selectedIndex == null && models.length > 0) {
        setSelectedIndex(models[0].index ?? 0);
      }
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const submitApproval = async () => {
    if (selectedIndex == null) return;
    setLoading(true);
    setMessage('');
    try {
      const res = await modelApi.approveModel({
        model_index: selectedIndex,
        decision,
        notes,
        approved_by: user?.username || user?.displayName || 'unknown',
      });
      setMessage(res.data.message);
      setNotes('');
      await loadData();
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Model Approval</h1>
        <p className="page-description">Approve, conditionally accept, or reject trained models for production usage</p>
      </div>

      {viewOnly && <div className="alert alert-info">You are in view-only mode (Validator role)</div>}
      {message && <div className={`alert ${message.startsWith('Error') ? 'alert-error' : 'alert-success'}`}>{message}</div>}

      <div className="card section">
        <div className="card-title" style={{ marginBottom: 12 }}>Current Approved Model</div>
        {!approvedModel ? (
          <div className="empty-state">No approved model yet</div>
        ) : (
          <div className="summary-grid">
            <div className="summary-card"><div className="summary-label">Model</div><div className="summary-value" style={{ fontSize: 14 }}>{approvedModel.model_type}</div></div>
            <div className="summary-card"><div className="summary-label">Accuracy</div><div className="summary-value">{formatMetric(approvedModel.metrics?.accuracy)}</div></div>
            <div className="summary-card"><div className="summary-label">AUC</div><div className="summary-value">{formatMetric(approvedModel.metrics?.auc)}</div></div>
            <div className="summary-card"><div className="summary-label">F1</div><div className="summary-value">{formatMetric(approvedModel.metrics?.f1)}</div></div>
          </div>
        )}
      </div>

      <div className="card section">
        <div className="card-title" style={{ marginBottom: 12 }}>Approve / Reject Model</div>
        {history.length === 0 ? (
          <div className="empty-state">No trained models available. Train models first.</div>
        ) : (
          <>
            <div className="table-container" style={{ marginBottom: 14 }}>
              <table>
                <thead><tr><th>Select</th><th>#</th><th>Model</th><th>Accuracy</th><th>AUC</th><th>F1</th><th>Timestamp</th></tr></thead>
                <tbody>
                  {history.map((m, i) => (
                    <tr key={m.index ?? i} style={{ background: (m.index ?? i) === approvedModelIndex ? 'rgba(16,185,129,0.08)' : 'transparent' }}>
                      <td>
                        <input
                          type="radio"
                          checked={selectedIndex === (m.index ?? i)}
                          onChange={() => setSelectedIndex(m.index ?? i)}
                        />
                      </td>
                      <td>{i + 1}</td>
                      <td style={{ fontWeight: 700 }}>
                        {m.model_type}
                        {(m.index ?? i) === approvedModelIndex && <span className="badge badge-success" style={{ marginLeft: 8 }}>Approved</span>}
                      </td>
                      <td>{formatMetric(m.metrics?.accuracy)}</td>
                      <td>{formatMetric(m.metrics?.auc || m.metrics?.auc_roc || m.metrics?.roc_auc)}</td>
                      <td>{formatMetric(m.metrics?.f1 || m.metrics?.f1_score)}</td>
                      <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>{m.timestamp?.substring(0, 19) || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
              <div className="form-group" style={{ minWidth: 220 }}>
                <label className="form-label">Decision</label>
                <select className="form-select" value={decision} onChange={(e) => setDecision(e.target.value)} disabled={viewOnly}>
                  <option value="approved">Approved</option>
                  <option value="conditional">Conditional</option>
                  <option value="rejected">Rejected</option>
                </select>
              </div>
              <div className="form-group" style={{ flex: 1, minWidth: 320 }}>
                <label className="form-label">Notes</label>
                <input
                  className="form-input"
                  type="text"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Optional reviewer notes..."
                  disabled={viewOnly}
                />
              </div>
            </div>

            <button className="btn btn-primary" onClick={submitApproval} disabled={viewOnly || loading || selectedIndex == null}>
              {loading ? <><div className="spinner" /> Saving...</> : 'Submit Decision'}
            </button>
          </>
        )}
      </div>

      <div className="card section">
        <div className="card-title" style={{ marginBottom: 12 }}>Approval History</div>
        {approvals.length === 0 ? (
          <div className="empty-state">No approval decisions recorded yet</div>
        ) : (
          <div className="table-container">
            <table>
              <thead><tr><th>Time</th><th>Model</th><th>Decision</th><th>By</th><th>Notes</th></tr></thead>
              <tbody>
                {[...approvals].reverse().map((a, idx) => (
                  <tr key={idx}>
                    <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>{a.timestamp?.substring(0, 19) || '—'}</td>
                    <td style={{ fontWeight: 700 }}>{a.model_type}</td>
                    <td>
                      <span className={`badge ${a.decision === 'approved' ? 'badge-success' : a.decision === 'conditional' ? 'badge-warning' : 'badge-error'}`}>
                        {a.decision}
                      </span>
                    </td>
                    <td>{a.approved_by || '—'}</td>
                    <td style={{ maxWidth: 360 }}>{a.notes || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
