"use client";

import { FormEvent, useEffect, useState } from "react";

import {
  createJournal,
  deleteJournal,
  getJournal,
  listJournals,
  updateJournal,
  type Journal,
  type User,
} from "../lib/api";

type Props = {
  user: User;
  onLogout: () => Promise<void>;
};

const PAGE_SIZE = 20;

export default function JournalWorkspace({
  user,
  onLogout,
}: Props) {
  const [journals, setJournals] = useState<Journal[]>([]);
  const [total, setTotal] = useState(0);

  const [text, setText] = useState("");
  const [selected, setSelected] = useState<Journal | null>(null);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadHistory() {
    try {
      const result = await listJournals(PAGE_SIZE, 0);

      setJournals(result.items);
      setTotal(result.total);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not load journal history"
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadHistory();
  }, []);

  async function pollJournal(id: string) {
    const maxAttempts = 20;

    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 1000));

      const journal = await getJournal(id);

      if (
        journal.status === "COMPLETED" ||
        journal.status === "FAILED"
      ) {
        setSelected(journal);
        await loadHistory();
        return;
      }
    }

    await loadHistory();
  }

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();

    const cleanedText = text.trim();

    if (!cleanedText) {
      return;
    }

    setSaving(true);
    setError(null);

    try {
      if (selected) {
        const result = await updateJournal(
          selected.id,
          cleanedText
        );

        setSelected({
          ...selected,
          text: cleanedText,
          status: result.status as Journal["status"],
          state: null,
        });

        await loadHistory();
        void pollJournal(selected.id);
      } else {
        const result = await createJournal(cleanedText);

        setText("");

        await loadHistory();
        void pollJournal(result.id);
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not save journal"
      );
    } finally {
      setSaving(false);
    }
  }

  function handleSelect(journal: Journal) {
    setSelected(journal);
    setText(journal.text);
    setError(null);
  }

  function handleNewJournal() {
    setSelected(null);
    setText("");
    setError(null);
  }

  async function handleDelete() {
    if (!selected) {
      return;
    }

    const confirmed = window.confirm(
      "Delete this journal entry? This cannot be undone."
    );

    if (!confirmed) {
      return;
    }

    setSaving(true);
    setError(null);

    try {
      await deleteJournal(selected.id);

      setSelected(null);
      setText("");

      await loadHistory();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not delete journal"
      );
    } finally {
      setSaving(false);
    }
  }

  return (
    <main>
      <header>
        <div>
          <h1>Continuum</h1>
          <p>
            Signed in as{" "}
            <strong>{user.display_name || user.email}</strong>
          </p>
        </div>

        <button type="button" onClick={() => void onLogout()}>
          Log out
        </button>
      </header>

      <hr />

      <section>
        <div>
          <h2>Your journal</h2>

          <button type="button" onClick={handleNewJournal}>
            New entry
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="What's on your mind today?"
            rows={10}
            maxLength={10000}
            required
          />

          <div>
            <button
              type="submit"
              disabled={saving || !text.trim()}
            >
              {saving
                ? "Saving..."
                : selected
                  ? "Save changes"
                  : "Save journal"}
            </button>

            {selected && (
              <button
                type="button"
                onClick={() => void handleDelete()}
                disabled={saving}
              >
                Delete
              </button>
            )}
          </div>
        </form>

        {error && <p role="alert">{error}</p>}
      </section>

      <hr />

      <section>
        <h2>History</h2>

        <p>{total} journal entries</p>

        {loading ? (
          <p>Loading...</p>
        ) : journals.length === 0 ? (
          <p>No journal entries yet.</p>
        ) : (
          <div>
            {journals.map((journal) => (
              <button
                key={journal.id}
                type="button"
                onClick={() => handleSelect(journal)}
              >
                <div>
                  <strong>
                    {new Date(
                      journal.created_at
                    ).toLocaleString()}
                  </strong>
                </div>

                <div>
                  {journal.text.length > 100
                    ? `${journal.text.slice(0, 100)}...`
                    : journal.text}
                </div>

                <div>Status: {journal.status}</div>
              </button>
            ))}
          </div>
        )}
      </section>

      {selected && (
        <>
          <hr />

          <section>
            <h2>Entry analysis</h2>

            <p>Status: {selected.status}</p>

            {selected.status === "QUEUED" ||
            selected.status === "PROCESSING" ? (
              <p>Continuum is analysing this entry...</p>
            ) : null}

            {selected.status === "FAILED" ? (
              <p>
                Analysis failed. Please edit and save the entry to
                try again.
              </p>
            ) : null}

            {selected.status === "COMPLETED" &&
            selected.state ? (
              <div>
                <p>
                  Energy:{" "}
                  {Math.round(selected.state.energy * 100)}%
                </p>

                <p>
                  Stress:{" "}
                  {Math.round(selected.state.stress * 100)}%
                </p>

                <p>
                  Confidence:{" "}
                  {Math.round(
                    selected.state.confidence * 100
                  )}
                  %
                </p>

                <small>
                  Model: {selected.state.model_version}
                </small>
              </div>
            ) : null}
          </section>
        </>
      )}
    </main>
  );
}