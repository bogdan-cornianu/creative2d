import { useCallback, useEffect, useState } from "react";
import { api, LocalModel, LocalModels as LocalModelsData } from "./api";

const KIND_LABEL: Record<LocalModel["kind"], string> = { image: "Image", video: "Image to video", matting: "Background removal" };

function gb(bytes: number) {
  return (bytes / 1e9).toFixed(1);
}

function statusText(m: LocalModel): string {
  switch (m.status) {
    case "ready":
      return "Ready";
    case "queued":
      return "Waiting…";
    case "downloading":
      return `Downloading ${gb(m.downloaded_bytes ?? 0)}${m.size_gb ? ` of ~${m.size_gb}` : ""} GB`;
    case "error":
      return "Download failed";
    default:
      return m.size_gb ? `Not downloaded · ~${m.size_gb} GB` : "Not downloaded";
  }
}

export function LocalModels({ open }: { open: boolean }) {
  const [data, setData] = useState<LocalModelsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    () =>
      api
        .localModels()
        .then((d) => (setData(d), setError(null)))
        .catch((e) => setError(String(e.message ?? e))),
    [],
  );

  useEffect(() => {
    if (open) load();
  }, [open, load]);

  const busy = data?.models.some((m) => m.status === "queued" || m.status === "downloading") ?? false;
  useEffect(() => {
    if (!open || !busy) return;
    const t = setInterval(load, 1500);
    return () => clearInterval(t);
  }, [open, busy, load]);

  async function download(ids: string[]) {
    if (!ids.length) return;
    try {
      const r = await api.downloadModels(ids);
      setData((d) => (d ? { ...d, models: r.models } : d));
      setError(null);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  }

  const pendingDefaults =
    data?.models.filter((m) => m.default && (m.status === "missing" || m.status === "error")).map((m) => m.id) ?? [];

  return (
    <section className="local-models">
      <div className="local-models-head">
        <h3>Local models</h3>
        {data?.ml_available && (
          <button className="primary" onClick={() => download(pendingDefaults)} disabled={!pendingDefaults.length}>
            Download models
          </button>
        )}
      </div>
      {error && <p className="notice error">{error}</p>}
      {!data && !error && <p className="hint">Checking local models…</p>}
      {data && !data.ml_available && (
        <p className="hint">
          Local generation needs the ML dependencies: <code>uv sync --extra ml</code>, then restart the server.
        </p>
      )}
      {data?.ml_available && (
        <>
          <p className="hint">
            Models download only when you ask. <b>Download models</b> gets the defaults; files go to the Hugging Face cache.
          </p>
          <ul className="model-list">
            {data.models.map((m) => (
              <li key={m.id} className={`model-row ${m.status}`}>
                <div className="model-info">
                  <span className="model-name">
                    {m.label}
                    {m.default && <span className="tag">default</span>}
                  </span>
                  <span className="hint">
                    {KIND_LABEL[m.kind]} · {statusText(m)}
                    {m.gated && m.status !== "ready" && " · needs Hugging Face login"}
                  </span>
                  {m.status === "error" && m.error && <span className="hint error">{m.error}</span>}
                </div>
                {m.status !== "ready" && (
                  <button
                    className="ghost"
                    onClick={() => download([m.id])}
                    disabled={m.status === "queued" || m.status === "downloading"}
                  >
                    {m.status === "error" ? "Retry" : "Download"}
                  </button>
                )}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
