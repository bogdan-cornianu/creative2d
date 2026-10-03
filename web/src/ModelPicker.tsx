import { useEffect, useMemo, useState } from "react";
import { api, formatPrice, ModelKind, ORModel } from "./api";

const cache = new Map<ModelKind, Promise<ORModel[]>>();

export function useModels(kind: ModelKind, enabled: boolean) {
  const [models, setModels] = useState<ORModel[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    if (nonce > 0) cache.delete(kind);
    if (!cache.has(kind)) cache.set(kind, api.models(kind, nonce > 0));
    cache
      .get(kind)!
      .then((m) => alive && (setModels(m), setError(null)))
      .catch((e) => {
        cache.delete(kind);
        if (alive) setError(String(e.message ?? e));
      });
    return () => {
      alive = false;
    };
  }, [kind, enabled, nonce]);

  return { models, error, refresh: () => setNonce((n) => n + 1) };
}

interface Props {
  kind: ModelKind;
  label: string;
  value: string | null;
  onChange: (id: string | null) => void;
  optional?: boolean;
}

export function ModelPicker({ kind, label, value, onChange, optional }: Props) {
  const { models, error, refresh } = useModels(kind, true);
  const [filter, setFilter] = useState("");
  const shown = useMemo(() => {
    const f = filter.trim().toLowerCase();
    const list = models ?? [];
    const hits = f ? list.filter((m) => m.id.toLowerCase().includes(f) || m.name.toLowerCase().includes(f)) : list;
    // Keep the current selection visible even when filtered out.
    const cur = list.find((m) => m.id === value);
    return cur && !hits.includes(cur) ? [cur, ...hits] : hits;
  }, [models, filter, value]);

  const selected = models?.find((m) => m.id === value);

  return (
    <div className="field model-picker">
      <div className="field-head">
        <label htmlFor={`mp-${kind}-${label}`}>{label}</label>
        <button type="button" className="link" onClick={refresh} title="Reload the model list from OpenRouter">
          Reload list
        </button>
      </div>
      {error && <p className="hint error">Could not load models: {error}</p>}
      {models && models.length > 12 && (
        <input
          type="search"
          className="filter"
          placeholder={`Filter ${models.length} models`}
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label={`Filter ${label}`}
        />
      )}
      <select
        id={`mp-${kind}-${label}`}
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value || null)}
        disabled={!models}
      >
        <option value="">{models ? (optional ? "None" : "Choose a model") : "Loading models…"}</option>
        {shown.map((m) => (
          <option key={m.id} value={m.id}>
            {m.name}
          </option>
        ))}
      </select>
      {selected && (
        <p className="hint">
          <code>{selected.id}</code>
          {formatPrice(selected.pricing) && <span className="price">{formatPrice(selected.pricing)}</span>}
        </p>
      )}
    </div>
  );
}
