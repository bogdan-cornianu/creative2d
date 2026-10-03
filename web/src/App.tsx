import { useCallback, useEffect, useState } from "react";
import { api, Job, JobRow, JobSpec, Options, Settings } from "./api";
import { AssetView } from "./AssetView";
import { JobForm } from "./JobForm";
import { JobList } from "./JobList";
import { SettingsDialog } from "./SettingsDialog";

export function App() {
  const [options, setOptions] = useState<Options | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [device, setDevice] = useState<string>("");
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [bootError, setBootError] = useState<string | null>(null);
  const [formKey, setFormKey] = useState(0);

  const refreshJobs = useCallback(() => api.jobs().then(setJobs).catch(() => {}), []);

  useEffect(() => {
    Promise.all([api.options(), api.settings(), api.health()])
      .then(([o, s, h]) => {
        setOptions(o);
        setSettings(s);
        setDevice(h.device);
      })
      .catch((e) => setBootError(String(e.message ?? e)));
    refreshJobs();
  }, [refreshJobs]);

  // Keep the list fresh while anything is queued or running.
  const active = jobs.some((j) => j.status === "queued" || j.status === "running");
  useEffect(() => {
    if (!active) return;
    const t = setInterval(refreshJobs, 2500);
    return () => clearInterval(t);
  }, [active, refreshJobs]);

  // Selected job: fetch details, then follow live events until it finishes.
  useEffect(() => {
    if (!selected) {
      setJob(null);
      return;
    }
    let es: EventSource | null = null;
    let alive = true;
    api.job(selected).then((j) => {
      if (!alive) return;
      setJob(j);
      if (j.status === "queued" || j.status === "running") {
        es = new EventSource(`/api/jobs/${selected}/events`);
        es.onmessage = (msg) => {
          const ev = JSON.parse(msg.data);
          setJob((cur) => (cur && cur.id === ev.id ? { ...cur, ...ev } : cur));
          setJobs((list) => list.map((row) => (row.id === ev.id ? { ...row, ...ev } : row)));
          if (["done", "failed", "cancelled"].includes(ev.status)) {
            es?.close();
            api.job(selected).then((full) => alive && setJob(full));
            refreshJobs();
          }
        };
      }
    });
    return () => {
      alive = false;
      es?.close();
    };
  }, [selected, refreshJobs]);

  async function submit(spec: JobSpec) {
    const { id } = await api.createJob(spec);
    await refreshJobs();
    setSelected(id);
  }

  function reuse() {
    if (!job) return;
    try {
      localStorage.setItem("creative2d.form.v1", JSON.stringify(job.spec));
    } catch {
      /* storage unavailable */
    }
    setFormKey((k) => k + 1);
  }

  async function remove() {
    if (!job) return;
    if (!confirm(`Delete "${job.spec.name}" and its files?`)) return;
    await api.deleteJob(job.id);
    setSelected(null);
    refreshJobs();
  }

  if (bootError) {
    return (
      <main className="boot">
        <h1>creative2d</h1>
        <p>The asset server is not reachable: {bootError}</p>
        <p>
          Start it with <code>uv run creative2d serve</code>, then reload this page.
        </p>
      </main>
    );
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>
          creative<span>2d</span>
        </h1>
        <p className="device" title="Where local models run">
          {device && `Local models on ${device === "mps" ? "Apple GPU" : device.toUpperCase()}`}
        </p>
        <button className="ghost" onClick={() => setSettingsOpen(true)}>
          Settings
        </button>
      </header>

      <aside className="panel compose">
        <h2 className="panel-title">New asset</h2>
        {options ? (
          <JobForm key={formKey} options={options} settings={settings} onSubmit={submit} onOpenSettings={() => setSettingsOpen(true)} />
        ) : (
          <p className="hint">Loading options…</p>
        )}
      </aside>

      <main className="panel stage-panel">
        {job ? (
          <AssetView job={job} onCancel={() => api.cancelJob(job.id).then(refreshJobs)} onDelete={remove} onReuse={reuse} />
        ) : (
          <div className="stage-empty checker">
            <div className="welcome">
              <h2>Describe an asset, pick a size, press Generate.</h2>
              <p>
                Finished assets play here inside a real Phaser scene, using the same atlas, spritesheet and tileset files
                you download.
              </p>
            </div>
          </div>
        )}
      </main>

      <aside className="panel history">
        <h2 className="panel-title">Jobs</h2>
        <JobList jobs={jobs} selected={selected} onSelect={setSelected} />
      </aside>

      <SettingsDialog open={settingsOpen} settings={settings} onClose={() => setSettingsOpen(false)} onSaved={setSettings} />
    </div>
  );
}
