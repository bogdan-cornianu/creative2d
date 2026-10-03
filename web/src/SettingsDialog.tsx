import { FormEvent, useEffect, useRef, useState } from "react";
import { api, Settings } from "./api";
import { LocalModels } from "./LocalModels";
import { ModelPicker } from "./ModelPicker";

interface Props {
  open: boolean;
  settings: Settings | null;
  onClose: () => void;
  onSaved: (s: Settings) => void;
}

export function SettingsDialog({ open, settings, onClose, onSaved }: Props) {
  const ref = useRef<HTMLDialogElement>(null);
  const [key, setKey] = useState("");
  const [status, setStatus] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  async function saveKey(e: FormEvent) {
    e.preventDefault();
    if (!key.trim()) return;
    setBusy(true);
    setStatus(null);
    try {
      const s = await api.saveSettings({ openrouter_api_key: key.trim() });
      onSaved(s);
      setKey("");
      const t = await api.testKey();
      setStatus({ ok: true, text: `Key saved and working${t.label ? ` (${t.label})` : ""}.` });
    } catch (err) {
      setStatus({ ok: false, text: String((err as Error).message) });
    } finally {
      setBusy(false);
    }
  }

  async function removeKey() {
    const s = await api.saveSettings({ openrouter_api_key: "" });
    onSaved(s);
    setStatus({ ok: true, text: "Key removed from this machine." });
  }

  async function test() {
    setBusy(true);
    try {
      const t = await api.testKey();
      const used = t.usage != null ? ` Used $${Number(t.usage).toFixed(2)}${t.limit ? ` of $${t.limit}` : ""}.` : "";
      setStatus({ ok: true, text: `Key works.${used}` });
    } catch (err) {
      setStatus({ ok: false, text: String((err as Error).message) });
    } finally {
      setBusy(false);
    }
  }

  async function setPref(k: keyof Settings["prefs"], v: string | null) {
    onSaved(await api.saveSettings({ prefs: { [k]: v ?? undefined } as Settings["prefs"] }));
  }

  return (
    <dialog ref={ref} className="settings" onClose={onClose} onCancel={onClose}>
      <header>
        <h2>Settings</h2>
        <button className="icon" onClick={onClose} aria-label="Close settings">
          ×
        </button>
      </header>
      <h3>OpenRouter</h3>
      <p className="hint">
        Your key stays on this computer in <code>~/.config/creative2d/secrets.env</code>. The browser never receives it back.
      </p>

      {settings?.openrouter_key_set ? (
        <div className="key-row">
          <span>
            Current key <code>{settings.openrouter_key_masked}</code>
            {settings.openrouter_key_source === "env" && " from OPENROUTER_API_KEY"}
          </span>
          <button className="ghost" onClick={test} disabled={busy}>
            Test key
          </button>
          {settings.openrouter_key_source === "file" && (
            <button className="ghost danger" onClick={removeKey}>
              Remove key
            </button>
          )}
        </div>
      ) : null}

      <form onSubmit={saveKey} className="key-form">
        <label className="field">
          <span>{settings?.openrouter_key_set ? "Replace key" : "API key"}</span>
          <input
            type="password"
            autoComplete="off"
            placeholder="sk-or-v1-…"
            value={key}
            onChange={(e) => setKey(e.target.value)}
          />
        </label>
        <button className="primary" type="submit" disabled={busy || !key.trim()}>
          Save key
        </button>
      </form>
      {status && <p className={`notice ${status.ok ? "ok" : "error"}`}>{status.text}</p>}

      {settings?.openrouter_key_set && open && (
        <>
          <h3>Default OpenRouter models</h3>
          <p className="hint">Used when a job does not pick its own.</p>
          <ModelPicker kind="image" label="Image" value={settings.prefs.image_model ?? null} onChange={(v) => setPref("image_model", v)} optional />
          <ModelPicker kind="video" label="Image to video" value={settings.prefs.video_model ?? null} onChange={(v) => setPref("video_model", v)} optional />
        </>
      )}

      <LocalModels open={open} />
    </dialog>
  );
}
