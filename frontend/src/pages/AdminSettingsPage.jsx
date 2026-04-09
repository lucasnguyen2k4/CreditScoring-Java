import { useState, useEffect } from 'react';
import { userApi } from '../api/client';
import { Trash2, Plus } from 'lucide-react';

export default function AdminSettingsPage() {
  const [users, setUsers] = useState([]);
  const [showAdd, setShowAdd] = useState(false);
  const [newUser, setNewUser] = useState({ username: '', password: '', displayName: '', role: 'SCORER' });
  const [message, setMessage] = useState('');

  const loadUsers = async () => {
    try {
      const res = await userApi.getAll();
      setUsers(res.data);
    } catch (err) {
      setMessage('Error loading users: ' + (err.response?.data?.error || err.message));
    }
  };

  useEffect(() => { loadUsers(); }, []);

  const addUser = async () => {
    try {
      await userApi.create(newUser);
      setMessage('User created');
      setShowAdd(false);
      setNewUser({ username: '', password: '', displayName: '', role: 'SCORER' });
      loadUsers();
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.error || err.message));
    }
  };

  const deleteUser = async (id, username) => {
    if (!confirm(`Delete user "${username}"?`)) return;
    try {
      await userApi.delete(id);
      setMessage('User deleted');
      loadUsers();
    } catch (err) {
      setMessage('Error: ' + (err.response?.data?.error || err.message));
    }
  };

  const toggleEnabled = async (id, enabled) => {
    try {
      await userApi.update(id, { enabled: String(!enabled) });
      loadUsers();
    } catch (err) {}
  };

  return (
    <div>
      <div className="page-header">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="page-title">Admin Settings</h1>
            <p className="page-description">Manage users and roles</p>
          </div>
          <button className="btn btn-primary" onClick={() => setShowAdd(!showAdd)}>
            <Plus size={16} /> Add User
          </button>
        </div>
      </div>

      {message && <div className={`alert ${message.startsWith('Error') ? 'alert-error' : 'alert-success'}`}>{message}</div>}

      {showAdd && (
        <div className="card section">
          <div className="card-title" style={{ marginBottom: 16 }}>Create New User</div>
          <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
            <div className="form-group" style={{ flex: 1, minWidth: 150 }}>
              <label className="form-label">Username</label>
              <input className="form-input" value={newUser.username}
                onChange={e => setNewUser({ ...newUser, username: e.target.value })} />
            </div>
            <div className="form-group" style={{ flex: 1, minWidth: 150 }}>
              <label className="form-label">Password</label>
              <input className="form-input" type="password" value={newUser.password}
                onChange={e => setNewUser({ ...newUser, password: e.target.value })} />
            </div>
            <div className="form-group" style={{ flex: 1, minWidth: 150 }}>
              <label className="form-label">Display Name</label>
              <input className="form-input" value={newUser.displayName}
                onChange={e => setNewUser({ ...newUser, displayName: e.target.value })} />
            </div>
            <div className="form-group" style={{ flex: 1, minWidth: 150 }}>
              <label className="form-label">Role</label>
              <select className="form-select" value={newUser.role}
                onChange={e => setNewUser({ ...newUser, role: e.target.value })}>
                {['ADMIN', 'MODEL_BUILDER', 'VALIDATOR', 'SCORER'].map(r =>
                  <option key={r} value={r}>{r.replace('_', ' ')}</option>
                )}
              </select>
            </div>
          </div>
          <div className="flex gap-sm">
            <button className="btn btn-primary btn-sm" onClick={addUser}>Create</button>
            <button className="btn btn-secondary btn-sm" onClick={() => setShowAdd(false)}>Cancel</button>
          </div>
        </div>
      )}

      <div className="card">
        <div className="table-container">
          <table>
            <thead><tr><th>Username</th><th>Display Name</th><th>Role</th><th>Status</th><th>Actions</th></tr></thead>
            <tbody>
              {users.map(u => (
                <tr key={u.id}>
                  <td style={{ fontWeight: 600 }}>{u.username}</td>
                  <td>{u.displayName}</td>
                  <td><span className="badge badge-info">{u.role?.replace('_', ' ')}</span></td>
                  <td>
                    <span className={`badge ${u.enabled ? 'badge-success' : 'badge-error'}`}
                      onClick={() => toggleEnabled(u.id, u.enabled)} style={{ cursor: 'pointer' }}>
                      {u.enabled ? 'Active' : 'Disabled'}
                    </span>
                  </td>
                  <td>
                    <button className="btn btn-danger btn-sm" onClick={() => deleteUser(u.id, u.username)}>
                      <Trash2 size={14} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
