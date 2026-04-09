import { useMemo, useState } from 'react';
import { shapApi, llmApi } from '../api/client';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Sparkles } from 'lucide-react';

export default function ShapExplanationPage() {
  const [globalData, setGlobalData] = useState(null);
  const [localData, setLocalData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [tab, setTab] = useState('global');
  const [sampleIdx, setSampleIdx] = useState(0);
  const [aiAnalysis, setAiAnalysis] = useState('');
  const [aiLoading, setAiLoading] = useState(false);
  const [aiMode, setAiMode] = useState('global');
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);

  const initShap = async () => {
    setLoading(true);
    setMessage('');
    try {
      await shapApi.init();
      const res = await shapApi.getGlobal();
      setGlobalData(res.data);
      setMessage('SHAP initialized successfully');
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const getLocal = async () => {
    setLoading(true);
    try {
      const res = await shapApi.getLocal(sampleIdx);
      setLocalData(res.data);
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const runAiAnalysis = async () => {
    setAiLoading(true);
    try {
      const res = aiMode === 'local'
        ? await llmApi.analyzeShapLocal(sampleIdx)
        : await llmApi.analyzeShapGlobal();
      setAiAnalysis(res.data.analysis);
    } catch (err) {
      setAiAnalysis('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setAiLoading(false);
    }
  };

  const sendChat = async () => {
    const question = chatInput.trim();
    if (!question) return;

    const nextHistory = [...chatHistory, { role: 'user', content: question }];
    setChatHistory(nextHistory);
    setChatInput('');
    setChatLoading(true);

    try {
      const res = await llmApi.chat(question, nextHistory);
      const answer = res.data.response || 'No response';
      setChatHistory((prev) => [...prev, { role: 'assistant', content: answer }]);
    } catch (err) {
      setChatHistory((prev) => [...prev, { role: 'assistant', content: 'Error: ' + (err.response?.data?.detail || err.message) }]);
    } finally {
      setChatLoading(false);
    }
  };

  const chartData = useMemo(() => (
    globalData?.features?.map((f, i) => ({
      name: f.length > 22 ? `${f.substring(0, 22)}...` : f,
      value: globalData.mean_abs_shap?.[i] || 0,
    })).sort((a, b) => b.value - a.value).slice(0, 15)
  ), [globalData]);

  const localChartData = useMemo(() => (
    (localData?.contributions || [])
      .map((c) => ({
        feature: c.feature,
        shap: Number(c.shap_value || 0),
      }))
      .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
      .slice(0, 15)
  ), [localData]);

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">SHAP Explanation</h1>
        <p className="page-description">Global + Local SHAP explanations with AI interpretation and chat</p>
      </div>

      {message && <div className={`alert ${message.startsWith('Error') ? 'alert-error' : 'alert-success'}`}>{message}</div>}

      {!globalData && (
        <div className="card">
          <div className="empty-state">
            <p style={{ marginBottom: 16 }}>Initialize SHAP to compute feature importance</p>
            <button className="btn btn-primary" onClick={initShap} disabled={loading}>
              {loading ? <><div className="spinner" /> Computing SHAP...</> : 'Initialize SHAP'}
            </button>
          </div>
        </div>
      )}

      {globalData && (
        <>
          <div className="tabs">
            <button className={`tab ${tab === 'global' ? 'active' : ''}`} onClick={() => setTab('global')}>Global Importance</button>
            <button className={`tab ${tab === 'local' ? 'active' : ''}`} onClick={() => setTab('local')}>Local Explanation</button>
            <button className={`tab ${tab === 'ai' ? 'active' : ''}`} onClick={() => setTab('ai')}>✨ AI Analysis</button>
          </div>

          {tab === 'global' && chartData && (
            <div className="card">
              <div className="card-title" style={{ marginBottom: 16 }}>Mean |SHAP| Feature Importance</div>
              <ResponsiveContainer width="100%" height={420}>
                <BarChart data={chartData} layout="vertical" margin={{ left: 120 }}>
                  <XAxis type="number" stroke="var(--text-muted)" fontSize={12} />
                  <YAxis type="category" dataKey="name" stroke="var(--text-muted)" fontSize={12} width={120} />
                  <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8 }} />
                  <Bar dataKey="value" fill="#6366f1" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {tab === 'local' && (
            <div className="card">
              <div className="flex gap-md items-center" style={{ marginBottom: 16 }}>
                <div className="form-group" style={{ margin: 0 }}>
                  <label className="form-label">Sample Index</label>
                  <input
                    className="form-input"
                    type="number"
                    min="0"
                    value={sampleIdx}
                    onChange={(e) => setSampleIdx(+e.target.value)}
                    style={{ width: 140 }}
                  />
                </div>
                <button className="btn btn-primary btn-sm" onClick={getLocal} disabled={loading} style={{ marginTop: 18 }}>
                  Explain Sample
                </button>
              </div>

              {localChartData.length > 0 && (
                <div className="chart-surface">
                  <div className="card-title" style={{ marginBottom: 12 }}>Local Waterfall-style Impact (Top Features)</div>
                  <ResponsiveContainer width="100%" height={360}>
                    <BarChart data={localChartData} layout="vertical" margin={{ left: 140 }}>
                      <XAxis type="number" stroke="var(--text-muted)" fontSize={12} />
                      <YAxis type="category" dataKey="feature" stroke="var(--text-muted)" fontSize={12} width={140} />
                      <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8 }} />
                      <Bar dataKey="shap">
                        {localChartData.map((row) => (
                          <Cell key={row.feature} fill={row.shap >= 0 ? '#ef4444' : '#10b981'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {localData?.contributions && (
                <div className="table-container">
                  <table>
                    <thead><tr><th>Feature</th><th>Value</th><th>SHAP Value</th><th>Effect</th></tr></thead>
                    <tbody>
                      {localData.contributions.map((c, i) => (
                        <tr key={i}>
                          <td style={{ fontWeight: 600 }}>{c.feature}</td>
                          <td>{typeof c.value === 'number' ? c.value.toFixed(4) : c.value}</td>
                          <td style={{ color: c.shap_value > 0 ? 'var(--error)' : 'var(--success)' }}>{c.shap_value?.toFixed(4)}</td>
                          <td>
                            <span className={c.shap_value > 0 ? 'badge badge-error' : 'badge badge-success'}>
                              {c.shap_value > 0 ? '↑ Risk' : '↓ Risk'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {tab === 'ai' && (
            <div className="card">
              <div className="chart-header">
                <div>
                  <div className="card-title">AI SHAP Interpretation</div>
                  <div className="chart-subtitle">Run global or local AI explanation, then ask follow-up questions.</div>
                </div>
                <div className="chart-controls">
                  <div className="form-group" style={{ margin: 0, minWidth: 220 }}>
                    <label className="form-label">Analysis Type</label>
                    <select className="form-select" value={aiMode} onChange={(e) => setAiMode(e.target.value)}>
                      <option value="global">Global</option>
                      <option value="local">Local (by sample)</option>
                    </select>
                  </div>
                  {aiMode === 'local' && (
                    <div className="form-group" style={{ margin: 0, width: 140 }}>
                      <label className="form-label">Sample</label>
                      <input className="form-input" type="number" min="0" value={sampleIdx} onChange={(e) => setSampleIdx(+e.target.value)} />
                    </div>
                  )}
                </div>
              </div>

              <button className="btn btn-primary" onClick={runAiAnalysis} disabled={aiLoading}>
                {aiLoading ? <><div className="spinner" /> Analyzing...</> : <><Sparkles size={16} /> Run AI Analysis</>}
              </button>

              {aiAnalysis && (
                <div className="card" style={{ marginTop: 14, whiteSpace: 'pre-wrap', fontSize: 14, lineHeight: 1.7 }}>
                  {aiAnalysis}
                </div>
              )}

              <div className="card" style={{ marginTop: 14 }}>
                <div className="card-title" style={{ marginBottom: 10 }}>Ask AI About This Model</div>
                <div style={{ maxHeight: 320, overflowY: 'auto', marginBottom: 10, border: '1px solid var(--border)', borderRadius: 8, padding: 10 }}>
                  {chatHistory.length === 0 && (
                    <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>No chat yet. Ask a question about feature impact, risk drivers, or model behavior.</div>
                  )}
                  {chatHistory.map((msg, idx) => (
                    <div key={`${msg.role}-${idx}`} style={{ marginBottom: 10 }}>
                      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 2 }}>{msg.role === 'user' ? 'You' : 'AI'}</div>
                      <div style={{ background: 'var(--bg-input)', borderRadius: 8, padding: '8px 10px', whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                    </div>
                  ))}
                </div>
                <div className="flex gap-sm">
                  <input
                    className="form-input"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Ask about this model..."
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendChat();
                      }
                    }}
                  />
                  <button className="btn btn-secondary" onClick={sendChat} disabled={chatLoading || !chatInput.trim()}>
                    {chatLoading ? 'Sending...' : 'Send'}
                  </button>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
