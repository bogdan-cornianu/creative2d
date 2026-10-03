import { FormEvent, ReactNode, useEffect, useState } from "react";
import { AssetType, JobSpec, Options, Settings } from "./api";
import { ModelPicker } from "./ModelPicker";

const STORAGE_KEY = "creative2d.form.v1";

const TYPE_LABELS: Record<AssetType, string> = {
  character: "Character",
  prop: "Prop",
  tile: "Tile",
  background: "Background",
};

const VIEW_LABELS = { side: "Side", topdown: "Top-down", iso: "Isometric" };

const PROMPT_HINTS: Record<AssetType, string> = {
  character: "A goblin archer with a green hood and a short bow",
  prop: "A wooden treasure chest with iron bands",
  tile: "Mossy cobblestone floor",
  background: "Misty pine forest at dawn with distant mountains",
};

function loadSaved(defaults: JobSpec): JobSpec {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const saved = JSON.parse(raw);
      return {
        ...defaults,
        ...saved,
        animate: { ...defaults.animate, ...saved.animate },
        output: { ...defaults.output, ...saved.output },
      };
    }
  } catch {
    /* storage unavailable */
  }
  return defaults;
}

function Segmented<T extends string | number>(props: {
  label: string;
  value: T;
  options: { value: T; label: string; disabled?: boolean }[];
  onChange: (v: T) => void;
}) {
  return (
    <fieldset className="field">
      <legend>{props.label}</legend>
      <div className="segmented" role="radiogroup">
        {props.options.map((o) => (
          <button
            key={String(o.value)}
            type="button"
            role="radio"
            aria-checked={props.value === o.value}
            disabled={o.disabled}
            className={props.value === o.value ? "on" : ""}
            onClick={() => props.onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

function Num(props: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
}) {
  return (
    <label className="field num">
      <span>{props.label}</span>
      <span className="num-input">
        <input
          type="number"
          value={props.value}
          min={props.min}
          max={props.max}
          step={props.step ?? 1}
          onChange={(e) => props.onChange(Number(e.target.value))}
        />
        {props.suffix && <em>{props.suffix}</em>}
      </span>
    </label>
  );
}

function Section(props: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="form-section">
      <header>
        <h3>{props.title}</h3>
        {props.aside}
      </header>
      {props.children}
    </section>
  );
}

interface Props {
  options: Options;
  settings: Settings | null;
  onSubmit: (spec: JobSpec) => Promise<void>;
  onOpenSettings: () => void;
}

export function JobForm({ options, settings, onSubmit, onOpenSettings }: Props) {
  const defaults: JobSpec = { prompt: "", ...options.defaults, seed: null } as JobSpec;
  const [spec, setSpec] = useState<JobSpec>(() => loadSaved(defaults));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(spec));
    } catch {
      /* storage unavailable */
    }
  }, [spec]);

  const set = <K extends keyof JobSpec>(k: K, v: JobSpec[K]) => setSpec((s) => ({ ...s, [k]: v }));
  const setAnim = <K extends keyof JobSpec["animate"]>(k: K, v: JobSpec["animate"][K]) =>
    setSpec((s) => ({ ...s, animate: { ...s.animate, [k]: v } }));
  const setOut = <K extends keyof JobSpec["output"]>(k: K, v: JobSpec["output"][K]) =>
    setSpec((s) => ({ ...s, output: { ...s.output, [k]: v } }));

  const isSprite = spec.asset_type === "character" || spec.asset_type === "prop";
  const needsKey =
    spec.backend === "openrouter" ||
    (isSprite && spec.animate.enabled && spec.animate.backend === "openrouter") ||
    spec.enhance_prompt ||
    !!spec.vision_model;
  const keyMissing = needsKey && !settings?.openrouter_key_set;
  const localMissing = !options.ml_available && (spec.backend === "local" || (spec.animate.enabled && spec.animate.backend === "local"));

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    const payload: JobSpec = {
      ...spec,
      prompt: spec.prompt.trim(),
      animate: { ...spec.animate, enabled: isSprite && spec.animate.enabled },
      enhance_prompt: spec.enhance_prompt && !!spec.text_model,
      vision_model: spec.candidates > 1 ? spec.vision_model : null,
      image_model: spec.image_model,
    };
    // Model ids are backend specific; drop one left over from the other backend.
    if (payload.backend === "local" && payload.image_model && !(payload.image_model in options.image_profiles)) {
      payload.image_model = null;
    }
    if (payload.animate.backend === "local" && payload.animate.model && !(payload.animate.model in options.video_profiles)) {
      payload.animate.model = null;
    }
    setBusy(true);
    try {
      await onSubmit(payload);
    } catch (err) {
      setError(String((err as Error).message ?? err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="job-form" onSubmit={submit}>
      <div className="field">
        <label htmlFor="prompt">What should it look like?</label>
        <textarea
          id="prompt"
          rows={3}
          value={spec.prompt}
          placeholder={PROMPT_HINTS[spec.asset_type]}
          onChange={(e) => set("prompt", e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) (e.currentTarget.form as HTMLFormElement).requestSubmit();
          }}
        />
      </div>

      <Segmented
        label="Asset"
        value={spec.asset_type}
        options={options.asset_types.map((t) => ({ value: t, label: TYPE_LABELS[t] }))}
        onChange={(v) => set("asset_type", v)}
      />

      <div className="row">
        <label className="field">
          <span>Style</span>
          <select value={spec.style} onChange={(e) => set("style", e.target.value)}>
            {Object.entries(options.styles).map(([k, label]) => (
              <option key={k} value={k}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>File name</span>
          <input
            value={spec.name}
            pattern="[A-Za-z0-9_\-]{1,48}"
            title="Letters, numbers, - and _ only"
            onChange={(e) => set("name", e.target.value)}
          />
        </label>
      </div>

      <Segmented
        label="Camera"
        value={spec.view}
        options={options.views.map((v) => ({ value: v, label: VIEW_LABELS[v] }))}
        onChange={(v) => set("view", v)}
      />

      <div className="row">
        {isSprite && <Num label="Frame size" value={spec.frame_size} onChange={(v) => set("frame_size", v)} min={8} max={1024} suffix="px" />}
        {spec.asset_type === "tile" && (
          <Num label={spec.view === "iso" ? "Tile width" : "Tile size"} value={spec.tile_size} onChange={(v) => set("tile_size", v)} min={8} max={512} suffix="px" />
        )}
        {spec.asset_type === "tile" && (
          <label className="field">
            <span>Tiles per texture</span>
            <select
              value={spec.tile_chunk ?? ""}
              onChange={(e) => set("tile_chunk", e.target.value === "" ? null : Number(e.target.value))}
              title="Small tiles are cut from one larger seamless texture so details stay readable"
            >
              <option value="">Auto</option>
              <option value="1">1 × 1</option>
              <option value="2">2 × 2</option>
              <option value="4">4 × 4</option>
            </select>
          </label>
        )}
        {spec.asset_type === "tile" && spec.view === "iso" && (
          <Num label="Block height" value={spec.iso_depth} onChange={(v) => set("iso_depth", v)} min={0} max={512} suffix="px" />
        )}
        {spec.asset_type === "background" && (
          <>
            <Num label="Width" value={spec.background_width} onChange={(v) => set("background_width", v)} min={128} max={4096} suffix="px" />
            <Num label="Height" value={spec.background_height} onChange={(v) => set("background_height", v)} min={128} max={4096} suffix="px" />
          </>
        )}
        {spec.asset_type !== "background" && (
          <Num label="Variants" value={spec.variants} onChange={(v) => set("variants", v)} min={1} max={16} />
        )}
        {spec.asset_type === "background" && (
          <Num label="Parallax layers" value={spec.parallax_layers} onChange={(v) => set("parallax_layers", v)} min={1} max={4} />
        )}
      </div>

      {isSprite && (
        <Segmented
          label="Facing directions"
          value={spec.directions}
          options={[
            { value: 1, label: "One" },
            { value: 4, label: "Four" },
            { value: 8, label: "Eight" },
          ]}
          onChange={(v) => set("directions", v)}
        />
      )}

      <Section title="Image model">
        <Segmented
          label="Runs on"
          value={spec.backend}
          options={[
            { value: "local", label: "This Mac" },
            { value: "openrouter", label: "OpenRouter" },
          ]}
          onChange={(v) => set("backend", v)}
        />
        {spec.backend === "local" ? (
          <label className="field">
            <span>Local model</span>
            <select value={spec.image_model && spec.image_model in options.image_profiles ? spec.image_model : options.default_image_profile} onChange={(e) => set("image_model", e.target.value)}>
              {Object.entries(options.image_profiles).map(([k, label]) => (
                <option key={k} value={k}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        ) : settings?.openrouter_key_set ? (
          <ModelPicker kind="image" label="OpenRouter image model" value={spec.image_model} onChange={(v) => set("image_model", v)} />
        ) : null}
      </Section>

      {isSprite && (
        <Section
          title="Animation"
          aside={
            <label className="switch">
              <input type="checkbox" checked={spec.animate.enabled} onChange={(e) => setAnim("enabled", e.target.checked)} />
              <span>{spec.animate.enabled ? "On" : "Off"}</span>
            </label>
          }
        >
          {spec.animate.enabled && (
            <>
              <div className="row">
                <label className="field">
                  <span>Action</span>
                  <select value={spec.animate.action} onChange={(e) => setAnim("action", e.target.value)}>
                    {options.actions.map((a) => (
                      <option key={a} value={a}>
                        {a[0].toUpperCase() + a.slice(1)}
                      </option>
                    ))}
                  </select>
                </label>
                <Num label="Frames" value={spec.animate.frames} onChange={(v) => setAnim("frames", v)} min={2} max={64} />
                <Num label="Speed" value={spec.animate.fps} onChange={(v) => setAnim("fps", v)} min={1} max={60} suffix="fps" />
              </div>
              {spec.animate.action === "custom" && (
                <label className="field">
                  <span>Describe the motion</span>
                  <input value={spec.animate.custom_motion} onChange={(e) => setAnim("custom_motion", e.target.value)} placeholder="Spins once and bows" />
                </label>
              )}
              <label className="check">
                <input type="checkbox" checked={spec.animate.loop} onChange={(e) => setAnim("loop", e.target.checked)} />
                Loop seamlessly
              </label>
              <Segmented
                label="Video model runs on"
                value={spec.animate.backend}
                options={[
                  { value: "local", label: "This Mac" },
                  { value: "openrouter", label: "OpenRouter" },
                ]}
                onChange={(v) => setAnim("backend", v)}
              />
              {spec.animate.backend === "local" ? (
                <label className="field">
                  <span>Local video model</span>
                  <select
                    value={spec.animate.model && spec.animate.model in options.video_profiles ? spec.animate.model : options.default_video_profile}
                    onChange={(e) => setAnim("model", e.target.value)}
                  >
                    {Object.entries(options.video_profiles).map(([k, label]) => (
                      <option key={k} value={k}>
                        {label}
                      </option>
                    ))}
                  </select>
                  <small className="hint">Local video takes several minutes per clip.</small>
                </label>
              ) : settings?.openrouter_key_set ? (
                <ModelPicker kind="video" label="OpenRouter video model" value={spec.animate.model} onChange={(v) => setAnim("model", v)} />
              ) : null}
            </>
          )}
        </Section>
      )}

      <details className="form-section more">
        <summary>Prompt help and quality checks</summary>
        <label className="check">
          <input type="checkbox" checked={spec.enhance_prompt} onChange={(e) => set("enhance_prompt", e.target.checked)} />
          Expand my prompt with an LLM
        </label>
        {spec.enhance_prompt && settings?.openrouter_key_set && (
          <ModelPicker kind="text" label="Text model" value={spec.text_model} onChange={(v) => set("text_model", v)} />
        )}
        <Num label="Candidates per image" value={spec.candidates} onChange={(v) => set("candidates", v)} min={1} max={8} />
        {spec.candidates > 1 && settings?.openrouter_key_set && (
          <ModelPicker kind="vision" label="Pick the best with" value={spec.vision_model} onChange={(v) => set("vision_model", v)} optional />
        )}
        {spec.candidates > 1 && !spec.vision_model && <p className="hint">Without a vision model the first candidate is kept.</p>}
      </details>

      <details className="form-section more">
        <summary>Export</summary>
        <div className="checks">
          {spec.asset_type !== "background" && (
            <label className="check">
              <input type="checkbox" checked={spec.output.atlas} onChange={(e) => setOut("atlas", e.target.checked)} />
              Texture atlas (JSON hash)
            </label>
          )}
          {isSprite && (
            <label className="check">
              <input type="checkbox" checked={spec.output.spritesheet} onChange={(e) => setOut("spritesheet", e.target.checked)} />
              Spritesheet grid
            </label>
          )}
          {spec.asset_type === "tile" && (
            <label className="check">
              <input type="checkbox" checked={spec.output.tilesheet} onChange={(e) => setOut("tilesheet", e.target.checked)} />
              Tile sheet and Tiled tileset
            </label>
          )}
        </div>
        <div className="row">
          <Num label="Padding" value={spec.output.padding} onChange={(v) => setOut("padding", v)} min={0} max={32} suffix="px" />
          <Num label="Edge extrude" value={spec.output.extrude} onChange={(v) => setOut("extrude", v)} min={0} max={8} suffix="px" />
          <label className="field">
            <span>Max atlas</span>
            <select value={spec.output.max_atlas_size} onChange={(e) => setOut("max_atlas_size", Number(e.target.value))}>
              {[1024, 2048, 4096].map((s) => (
                <option key={s} value={s}>
                  {s}px
                </option>
              ))}
            </select>
          </label>
        </div>
        <label className="field num">
          <span>Seed</span>
          <input
            type="number"
            placeholder="Random"
            value={spec.seed ?? ""}
            onChange={(e) => set("seed", e.target.value === "" ? null : Number(e.target.value))}
          />
        </label>
      </details>

      {keyMissing && (
        <p className="notice">
          OpenRouter needs an API key.{" "}
          <button type="button" className="link" onClick={onOpenSettings}>
            Add your key
          </button>
        </p>
      )}
      {localMissing && <p className="notice">Local models are not installed. Run <code>uv sync --extra ml</code>, then restart the server.</p>}
      {error && <p className="notice error">{error}</p>}

      <button className="primary" type="submit" disabled={busy || !spec.prompt.trim() || keyMissing || localMissing}>
        {busy ? "Queuing…" : "Generate"}
      </button>
    </form>
  );
}
