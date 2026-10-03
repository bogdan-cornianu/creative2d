import { assetUrl, JobRow } from "./api";

const TYPE_GLYPH: Record<string, string> = { character: "C", prop: "P", tile: "T", background: "B" };

function ago(ts: number) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

interface Props {
  jobs: JobRow[];
  selected: string | null;
  onSelect: (id: string) => void;
}

export function JobList({ jobs, selected, onSelect }: Props) {
  if (!jobs.length) {
    return <p className="empty-list">Jobs you start show up here, newest first.</p>;
  }
  return (
    <ol className="job-list">
      {jobs.map((j) => (
        <li key={j.id}>
          <button className={`job ${j.status} ${selected === j.id ? "on" : ""}`} onClick={() => onSelect(j.id)}>
            {j.status === "done" && j.thumb ? (
              <img
                className={`thumb checker ${j.spec.style === "pixel" ? "pixelated" : ""}`}
                src={assetUrl(j.id, j.thumb)}
                alt=""
                loading="lazy"
              />
            ) : (
              <span className={`glyph ${j.spec.asset_type}`} aria-hidden>
                {TYPE_GLYPH[j.spec.asset_type]}
              </span>
            )}
            <span className="job-text">
              <span className="job-prompt">{j.spec.prompt}</span>
              <span className="job-meta">
                {j.status === "running"
                  ? `${Math.round(j.progress * 100)}%, ${j.stage}`
                  : j.status === "done"
                    ? `${j.spec.style}, ${j.spec.view}${j.spec.animate.enabled ? `, ${j.spec.animate.action}` : ""}`
                    : j.status}
                {`, ${ago(j.created)}`}
              </span>
            </span>
            {j.status === "running" && (
              <span className="mini-bar" aria-hidden>
                <span style={{ width: `${j.progress * 100}%` }} />
              </span>
            )}
          </button>
        </li>
      ))}
    </ol>
  );
}
