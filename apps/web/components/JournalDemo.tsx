"use client";

import { FormEvent, useState } from "react";
import { createJournal, getJournal, Journal } from "../lib/api";

function pct(value: number) {
  return `${Math.round(value * 100)}%`;
}

export default function JournalDemo() {
  const [text, setText] = useState("Work was exhausting today, but I am hopeful about tomorrow.");
  const [journal, setJournal] = useState<Journal | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setJournal(null);
    try {
      const created = await createJournal(text);
      for (let attempt = 0; attempt < 30; attempt++) {
        const current = await getJournal(created.id);
        setJournal(current);
        if (current.status === "COMPLETED" || current.status === "FAILED") break;
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <form className="card" onSubmit={submit}>
        <div className="eyebrow">M0 · TEXT JOURNAL</div>
        <h2>How are things today?</h2>
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={6} minLength={1} />
        <button disabled={busy || !text.trim()}>{busy ? "Processing…" : "Save journal"}</button>
      </form>

      {error && <div className="card error">{error}</div>}

      {journal && (
        <section className="card">
          <div className="row"><strong>Journal</strong><span className={`status ${journal.status.toLowerCase()}`}>{journal.status}</span></div>
          <p className="muted">{journal.id}</p>
          {journal.state && (
            <div className="metrics">
              <div className="metric"><span>Energy</span><strong>{pct(journal.state.energy)}</strong></div>
              <div className="metric"><span>Stress</span><strong>{pct(journal.state.stress)}</strong></div>
              <div className="metric"><span>Confidence</span><strong>{pct(journal.state.confidence)}</strong></div>
              <div className="model">{journal.state.model_version}</div>
            </div>
          )}
          {journal.status === "FAILED" && <p className="errorText">Your journal was saved, but analysis failed.</p>}
        </section>
      )}
    </div>
  );
}
