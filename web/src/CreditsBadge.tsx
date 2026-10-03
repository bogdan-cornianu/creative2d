import { ReactNode, useCallback, useEffect, useState } from "react";
import { api, Credits } from "./api";

function usd(n: number) {
  return n > 0 && n < 0.01 ? "<$0.01" : `$${n.toFixed(2)}`;
}

function describe(c: Credits): string {
  if (c.source === "account") return `Account balance: ${usd(c.remaining!)} of ${usd(c.total!)} purchased.`;
  if (c.source === "key")
    return `Left under this key's ${usd(c.total ?? 0)} limit. The account balance needs an OpenRouter management key.`;
  return "This key has no spending limit and can not read the account balance, so only its spend is shown.";
}

interface Props {
  /** Changes whenever the key changes or spending may have happened; triggers a reload. */
  refreshKey: string;
  onOpenSettings: () => void;
}

/** OpenRouter credits left, for the top bar. Render it only when a key is set. */
export function CreditsBadge({ refreshKey, onOpenSettings }: Props) {
  const [credits, setCredits] = useState<Credits | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    () =>
      api
        .credits()
        .then((c) => (setCredits(c), setError(null)))
        .catch((e) => setError(String(e.message ?? e))),
    [],
  );

  useEffect(() => {
    load();
  }, [refreshKey, load]);

  // Coming back to the tab is when a top-up on openrouter.ai shows up.
  useEffect(() => {
    window.addEventListener("focus", load);
    return () => window.removeEventListener("focus", load);
  }, [load]);

  const low =
    !error &&
    credits?.remaining != null &&
    (credits.remaining < 1 || (credits.total ? credits.remaining / credits.total < 0.1 : false));

  let text: ReactNode = "…";
  if (error) text = "unavailable";
  else if (credits?.remaining != null) text = <><b>{usd(credits.remaining)}</b> left</>;
  else if (credits) text = <><b>{usd(credits.used)}</b> used</>;

  return (
    <button
      type="button"
      className={`credits ${low ? "low" : ""} ${error ? "error" : ""}`}
      onClick={onOpenSettings}
      title={error ? `Could not read OpenRouter credits: ${error}` : credits ? describe(credits) : "Loading OpenRouter credits"}
      aria-live="polite"
    >
      <span className="credits-label">OpenRouter</span> {text}
    </button>
  );
}
