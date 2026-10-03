import { useEffect, useState } from "react";
import { assetUrl, Job } from "./api";
import { PhaserPreview } from "./PhaserPreview";

type Tab = "preview" | "sheets" | "frames" | "code";

const STAGE_LABELS: Record<string, string> = {
  start: "Starting",
  enhance: "Writing prompt",
  generate: "Generating images",
  animate: "Animating",
  pack: "Packing",
  done: "Done",
};

interface Props {
  job: Job;
  onCancel: () => void;
  onDelete: () => void;
  onReuse: () => void;
}

export function AssetView({ job, onCancel, onDelete, onReuse }: Props) {
  const [tab, setTab] = useState<Tab>("preview");
  const [zoom, setZoom] = useState(3);
  const [code, setCode] = useState<string>("");
  const m = job.result;
  const pixel = job.spec.style === "pixel";

  useEffect(() => {
    if (tab !== "code" || !m) return;
    fetch(assetUrl(job.id, "phaser-loader.js"))
      .then((r) => r.text())
      .then(setCode)
      .catch(() => setCode("// loader snippet unavailable"));
  }, [tab, job.id, m]);

  useEffect(() => {
    setZoom(m?.result.kind === "background" ? 1 : m?.result.kind === "tiles" ? 2 : 3);
  }, [m?.result.kind, job.id]);

  if (job.status === "queued" || job.status === "running") {
    const pct = Math.round(job.progress * 100);
    return (
      <div className="stage-empty">
        <div className="progress-card" aria-live="polite">
          <p className="progress-title">{STAGE_LABELS[job.stage] ?? (job.status === "queued" ? "Waiting in queue" : "Working")}</p>
          <div className="bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <span style={{ width: `${pct}%` }} />
          </div>
          <p className="progress-msg">
            <span>{job.message}</span>
            <span className="pct">{pct}%</span>
          </p>
          <button className="ghost" onClick={onCancel}>
            Cancel job
          </button>
        </div>
      </div>
    );
  }

  if (job.status !== "done" || !m) {
    return (
      <div className="stage-empty">
        <div className="progress-card">
          <p className="progress-title">{job.status === "cancelled" ? "Job cancelled" : "Job failed"}</p>
          {job.error && <p className="error-text">{job.error}</p>}
          <div className="actions">
            <button className="ghost" onClick={onReuse}>
              Load settings into form
            </button>
            <button className="ghost danger" onClick={onDelete}>
              Delete job
            </button>
          </div>
        </div>
      </div>
    );
  }

  const r = m.result;
  const sheets = r.entries.flatMap((e) => (e.texture ? [e.texture] : e.textures ?? []));

  return (
    <div className="asset-view">
      <div className="asset-head">
        <div>
          <h2>{job.spec.name}</h2>
          <p className="meta">
            {r.frames?.length ?? r.entries.length} {r.kind === "background" ? "layers" : r.kind === "tiles" ? "tiles" : "frames"}
            {r.anims?.length ? `, ${r.anims.length} animation${r.anims.length > 1 ? "s" : ""}` : ""}
            {` in ${Math.round(m.seconds)} s`}
            {m.cost_usd > 0 ? `, $${m.cost_usd.toFixed(3)}` : ""}
            {`, seed ${m.seed}`}
          </p>
        </div>
        <div className="actions">
          <button className="ghost" onClick={onReuse}>
            Reuse settings
          </button>
          <button className="ghost danger" onClick={onDelete}>
            Delete
          </button>
          <a className="button primary" href={`/api/jobs/${job.id}/download`}>
            Download .zip
          </a>
        </div>
      </div>

      <div className="tabs" role="tablist">
        {(["preview", "sheets", "frames", "code"] as Tab[]).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>
            {{ preview: "In Phaser", sheets: "Sheets", frames: "Frames", code: "Loader code" }[t]}
          </button>
        ))}
        {tab === "preview" && (
          <label className="zoom">
            Zoom
            <input type="range" min={1} max={8} step={1} value={zoom} onChange={(e) => setZoom(Number(e.target.value))} />
            <span>{zoom}×</span>
          </label>
        )}
      </div>

      <div className="stage checker">
        {tab === "preview" && <PhaserPreview manifest={m} zoom={zoom} pixelArt={pixel} />}
        {tab === "sheets" && (
          <div className="sheet-list">
            {sheets.map((s) => (
              <figure key={s}>
                <img src={assetUrl(job.id, s)} alt={s} className={pixel ? "pixelated" : ""} />
                <figcaption>{s}</figcaption>
              </figure>
            ))}
          </div>
        )}
        {tab === "frames" && (
          <div className="frame-grid">
            {(r.frames ?? r.entries.map((e) => e.key)).map((f) => (
              <figure key={f}>
                <img
                  src={assetUrl(job.id, r.kind === "background" ? `${f}.png` : `frames/${f}.png`)}
                  alt={f}
                  className={pixel ? "pixelated" : ""}
                />
                <figcaption>{f}</figcaption>
              </figure>
            ))}
          </div>
        )}
        {tab === "code" && <pre className="code">{code}</pre>}
      </div>

      {m.log.length > 0 && (
        <details className="log">
          <summary>Pipeline notes</summary>
          <ul>
            {m.subject !== job.spec.prompt && <li>Prompt used: {m.subject}</li>}
            {m.log.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
