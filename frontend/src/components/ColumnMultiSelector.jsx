import { useId, useMemo, useState } from 'react';

function unique(values) {
  return [...new Set(values.map((v) => String(v).trim()).filter(Boolean))];
}

export default function ColumnMultiSelector({
  label,
  columns = [],
  values = [],
  onChange,
  disabled = false,
  placeholder = 'Type or choose a column',
  showSelectAll = true,
}) {
  const [draft, setDraft] = useState('');
  const listId = useId();
  const normalizedValues = useMemo(() => unique(values), [values]);

  const addColumns = (raw) => {
    const parsed = raw
      .split(',')
      .map((v) => v.trim())
      .filter(Boolean);
    if (!parsed.length) return;
    onChange(unique([...normalizedValues, ...parsed]));
    setDraft('');
  };

  const removeColumn = (column) => {
    onChange(normalizedValues.filter((v) => v !== column));
  };

  return (
    <div className="form-group" style={{ flex: 2, minWidth: 240 }}>
      <label className="form-label">{label}</label>
      <div className="flex gap-sm" style={{ alignItems: 'center' }}>
        <input
          className="form-input"
          list={listId}
          value={draft}
          disabled={disabled}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ',') {
              e.preventDefault();
              addColumns(draft);
            }
          }}
          placeholder={placeholder}
        />
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          disabled={disabled || !draft.trim()}
          onClick={() => addColumns(draft)}
        >
          Add
        </button>
        {showSelectAll && columns.length > 0 && (
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            disabled={disabled}
            onClick={() => onChange(unique(columns))}
          >
            All
          </button>
        )}
      </div>

      <datalist id={listId}>
        {columns.map((column) => (
          <option key={column} value={column} />
        ))}
      </datalist>

      <div className="column-tags">
        {normalizedValues.length === 0 && (
          <span className="column-tag column-tag-muted">No columns selected</span>
        )}
        {normalizedValues.map((column) => (
          <span key={column} className="column-tag">
            {column}
            <button
              type="button"
              className="column-tag-remove"
              disabled={disabled}
              onClick={() => removeColumn(column)}
              aria-label={`Remove ${column}`}
            >
              x
            </button>
          </span>
        ))}
      </div>
    </div>
  );
}
