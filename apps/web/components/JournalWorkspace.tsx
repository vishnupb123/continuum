"use client";

import {
  FormEvent,
  useEffect,
  useState,
} from "react";

import {
  createJournal,
  createVoiceJournal,
  deleteJournal,
  getJournal,
  getJournalAudio,
  listJournals,
  updateJournal,
  type Journal,
  type User,
} from "../lib/api";

import VoiceRecorder from "./VoiceRecorder";

type Props = {
  user: User;
  onLogout: () => Promise<void>;
};

type EntryMode = "TEXT" | "VOICE";

const PAGE_SIZE = 20;

export default function JournalWorkspace({
  user,
  onLogout,
}: Props) {
  const [journals, setJournals] =
    useState<Journal[]>([]);

  const [total, setTotal] =
    useState(0);

  const [text, setText] =
    useState("");

  const [selected, setSelected] =
    useState<Journal | null>(null);

  const [entryMode, setEntryMode] =
    useState<EntryMode>("TEXT");

  const [loading, setLoading] =
    useState(true);

  const [saving, setSaving] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);
  
  const [savedAudioUrl, setSavedAudioUrl] =
    useState<string | null>(null);

  const [audioLoading, setAudioLoading] =
    useState(false);

  const [audioError, setAudioError] =
    useState<string | null>(null);

  async function loadHistory() {
    try {
      const result = await listJournals(
        PAGE_SIZE,
        0
      );

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

  useEffect(() => {
  let cancelled = false;
  let objectUrl: string | null = null;

  async function loadSavedAudio() {
    setSavedAudioUrl(null);
    setAudioError(null);

    if (
      !selected ||
      selected.entry_type !== "VOICE" ||
      !selected.audio
    ) {
      setAudioLoading(false);
      return;
    }

    setAudioLoading(true);

    try {
      const blob =
        await getJournalAudio(
          selected.id
        );

      if (cancelled) {
        return;
      }

      objectUrl =
        URL.createObjectURL(blob);

      setSavedAudioUrl(
        objectUrl
      );
    } catch (err) {
      if (cancelled) {
        return;
      }

      setAudioError(
        err instanceof Error
          ? err.message
          : "Could not load journal audio"
      );
    } finally {
      if (!cancelled) {
        setAudioLoading(false);
      }
    }
  }

  void loadSavedAudio();

  return () => {
    cancelled = true;

    if (objectUrl) {
      URL.revokeObjectURL(
        objectUrl
      );
    }
  };
  }, [
    selected?.id,
    selected?.entry_type,
    selected?.audio?.id,
]);

  async function pollJournal(id: string) {
    const maxAttempts = 30;

    for (
      let attempt = 0;
      attempt < maxAttempts;
      attempt += 1
    ) {
      await new Promise((resolve) =>
        setTimeout(resolve, 1000)
      );

      try {
        const journal =
          await getJournal(id);

        setSelected(journal);

        if (
          journal.status ===
            "COMPLETED" ||
          journal.status === "FAILED"
        ) {
          await loadHistory();
          return;
        }
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Could not refresh journal status"
        );

        return;
      }
    }

    await loadHistory();
  }

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();

    const cleanedText =
      text.trim();

    if (!cleanedText) {
      return;
    }

    if (
      selected &&
      selected.entry_type === "VOICE"
    ) {
      return;
    }

    setSaving(true);
    setError(null);

    try {
      if (selected) {
        const result =
          await updateJournal(
            selected.id,
            cleanedText
          );

        setSelected({
          ...selected,
          text: cleanedText,
          status: result.status,
          state: null,
        });

        await loadHistory();

        void pollJournal(
          selected.id
        );
      } else {
        const result =
          await createJournal(
            cleanedText
          );

        setText("");

        const journal =
          await getJournal(
            result.id
          );

        setSelected(journal);

        await loadHistory();

        void pollJournal(
          result.id
        );
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

  async function handleVoiceRecording(
    audio: Blob,
    filename: string
  ) {
    setSaving(true);
    setError(null);

    try {
      const result =
        await createVoiceJournal(
          audio,
          filename
        );

      /*
       * Fetch immediately so the user can see
       * the newly created voice entry while
       * transcription is running.
       */
      const journal =
        await getJournal(result.id);

      setSelected(journal);

      await loadHistory();

      void pollJournal(result.id);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Could not save voice journal";

      setError(message);

      /*
       * Re-throw so VoiceRecorder knows the
       * upload failed and keeps the recording
       * available for retry.
       */
      throw err;
    } finally {
      setSaving(false);
    }
  }

  function handleSelect(
    journal: Journal
  ) {
    setSelected(journal);

    if (
      journal.entry_type === "TEXT"
    ) {
      setText(
        journal.text ?? ""
      );

      setEntryMode("TEXT");
    } else {
      setText("");
      setEntryMode("VOICE");
    }

    setError(null);
  }

  function handleNewJournal() {
    setSelected(null);
    setText("");
    setEntryMode("TEXT");
    setError(null);
  }

  function selectTextMode() {
    if (saving) {
      return;
    }

    setEntryMode("TEXT");
    setError(null);
  }

  function selectVoiceMode() {
    if (saving) {
      return;
    }

    setEntryMode("VOICE");
    setError(null);
  }

  async function handleDelete() {
    if (!selected) {
      return;
    }

    const confirmed =
      window.confirm(
        "Delete this journal entry? This cannot be undone."
      );

    if (!confirmed) {
      return;
    }

    setSaving(true);
    setError(null);

    try {
      await deleteJournal(
        selected.id
      );

      setSelected(null);
      setText("");
      setEntryMode("TEXT");

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
            <strong>
              {user.display_name ||
                user.email}
            </strong>
          </p>
        </div>

        <button
          type="button"
          onClick={() =>
            void onLogout()
          }
        >
          Log out
        </button>
      </header>

      <hr />

      <section>
        <div>
          <h2>Your journal</h2>

          <button
            type="button"
            onClick={
              handleNewJournal
            }
          >
            New entry
          </button>
        </div>

        {!selected && (
          <div>
            <button
              type="button"
              onClick={
                selectTextMode
              }
              disabled={
                saving ||
                entryMode === "TEXT"
              }
            >
              Text
            </button>

            <button
              type="button"
              onClick={
                selectVoiceMode
              }
              disabled={
                saving ||
                entryMode === "VOICE"
              }
            >
              Voice
            </button>
          </div>
        )}

        {!selected &&
        entryMode === "TEXT" ? (
          <form
            onSubmit={
              handleSubmit
            }
          >
            <textarea
              value={text}
              onChange={(event) =>
                setText(
                  event.target.value
                )
              }
              placeholder="What's on your mind today?"
              rows={10}
              maxLength={10000}
              required
            />

            <div>
              <button
                type="submit"
                disabled={
                  saving ||
                  !text.trim()
                }
              >
                {saving
                  ? "Saving..."
                  : "Save journal"}
              </button>
            </div>
          </form>
        ) : null}

        {!selected &&
        entryMode === "VOICE" ? (
          <VoiceRecorder
            disabled={saving}
            onRecordingReady={
              handleVoiceRecording
            }
          />
        ) : null}

        {selected?.entry_type ===
        "TEXT" ? (
          <form
            onSubmit={
              handleSubmit
            }
          >
            <textarea
              value={text}
              onChange={(event) =>
                setText(
                  event.target.value
                )
              }
              placeholder="What's on your mind today?"
              rows={10}
              maxLength={10000}
              required
            />

            <div>
              <button
                type="submit"
                disabled={
                  saving ||
                  !text.trim()
                }
              >
                {saving
                  ? "Saving..."
                  : "Save changes"}
              </button>

              <button
                type="button"
                onClick={() =>
                  void handleDelete()
                }
                disabled={saving}
              >
                Delete
              </button>
            </div>
          </form>
        ) : null}

        {selected?.entry_type ===
        "VOICE" ? (
          <div>
            <h3>Voice journal</h3>
            <div>
            <p>
           <strong>
           Recording
          </strong>
         </p>

        {audioLoading && (
         <p>
         Loading recording...
        </p>
  )}

  {audioError && (
    <p role="alert">
      {audioError}
    </p>
  )}

  {savedAudioUrl && (
    <audio
      controls
      preload="metadata"
      src={savedAudioUrl}
    >
      Your browser does not
      support audio playback.
    </audio>
  )}
</div>

            {selected.text ? (
              <>
                <p>
                  <strong>
                    Transcript
                  </strong>
                </p>

                <p>
                  {selected.text}
                </p>
              </>
            ) : (
              <p>
                {selected.audio
                  ?.transcription_status ===
                "TRANSCRIBING"
                  ? "Your voice journal is being transcribed..."
                  : selected.audio
                        ?.transcription_status ===
                      "TRANSCRIPTION_QUEUED"
                    ? "Your voice journal is waiting for transcription..."
                    : selected.audio
                          ?.transcription_status ===
                        "TRANSCRIPTION_FAILED"
                      ? "Voice transcription failed."
                      : "No transcript is available yet."}
              </p>
            )}

            {selected.audio && (
              <div>
                <p>
                  Audio type:{" "}
                  {
                    selected.audio
                      .mime_type
                  }
                </p>

                <p>
                  Size:{" "}
                  {Math.round(
                    selected.audio
                      .size_bytes /
                      1024
                  )}{" "}
                  KB
                </p>

                {selected.audio
                  .duration_seconds !==
                  null && (
                  <p>
                    Duration:{" "}
                    {selected.audio.duration_seconds.toFixed(
                      1
                    )}{" "}
                    seconds
                  </p>
                )}

                <p>
                  Transcription:{" "}
                  {
                    selected.audio
                      .transcription_status
                  }
                </p>
              </div>
            )}

            <button
              type="button"
              onClick={() =>
                void handleDelete()
              }
              disabled={saving}
            >
              {saving
                ? "Deleting..."
                : "Delete voice journal"}
            </button>
          </div>
        ) : null}

        {error && (
          <p role="alert">
            {error}
          </p>
        )}
      </section>

      <hr />

      <section>
        <h2>History</h2>

        <p>
          {total} journal entries
        </p>

        {loading ? (
          <p>Loading...</p>
        ) : journals.length ===
          0 ? (
          <p>
            No journal entries yet.
          </p>
        ) : (
          <div>
            {journals.map(
              (journal) => (
                <button
                  key={journal.id}
                  type="button"
                  onClick={() =>
                    handleSelect(
                      journal
                    )
                  }
                >
                  <div>
                    <strong>
                      {new Date(
                        journal.created_at
                      ).toLocaleString()}
                    </strong>
                  </div>

                  <div>
                    {journal.entry_type ===
                    "VOICE"
                      ? "Voice entry"
                      : "Text entry"}
                  </div>

                  <div>
                    {journal.entry_type ===
                      "VOICE" &&
                    !journal.text ? (
                      <span>
                        Voice journal
                        {journal.audio
                          ?.transcription_status ===
                        "TRANSCRIBING"
                          ? " · Transcribing..."
                          : journal.audio
                                ?.transcription_status ===
                              "TRANSCRIPTION_QUEUED"
                            ? " · Waiting for transcription..."
                            : journal.audio
                                  ?.transcription_status ===
                                "TRANSCRIPTION_FAILED"
                              ? " · Transcription failed"
                              : ""}
                      </span>
                    ) : (
                      <span>
                        {(journal.text ??
                          "").length >
                        100
                          ? `${(
                              journal.text ??
                              ""
                            ).slice(
                              0,
                              100
                            )}...`
                          : journal.text ??
                            "No transcript available"}
                      </span>
                    )}
                  </div>

                  <div>
                    Status:{" "}
                    {journal.status}
                  </div>
                </button>
              )
            )}
          </div>
        )}
      </section>

      {selected && (
        <>
          <hr />

          <section>
            <h2>
              Entry analysis
            </h2>

            <p>
              Type:{" "}
              {selected.entry_type ===
              "VOICE"
                ? "Voice"
                : "Text"}
            </p>

            <p>
              Status:{" "}
              {selected.status}
            </p>

            {selected.status ===
              "QUEUED" ||
            selected.status ===
              "PROCESSING" ? (
              <p>
                {selected.entry_type ===
                  "VOICE" &&
                selected.audio
                  ?.transcription_status !==
                  "TRANSCRIPTION_COMPLETED"
                  ? "Continuum is processing your voice journal..."
                  : "Continuum is analysing this entry..."}
              </p>
            ) : null}

            {selected.status ===
            "FAILED" ? (
              <p>
                {selected.entry_type ===
                "VOICE"
                  ? "Continuum could not process this voice journal."
                  : "Analysis failed. Please edit and save the entry to try again."}
              </p>
            ) : null}

            {selected.status ===
              "COMPLETED" &&
            selected.state ? (
              <div>
                <p>
                  Energy:{" "}
                  {Math.round(
                    selected.state
                      .energy * 100
                  )}
                  %
                </p>

                <p>
                  Stress:{" "}
                  {Math.round(
                    selected.state
                      .stress * 100
                  )}
                  %
                </p>

                <p>
                  Confidence:{" "}
                  {Math.round(
                    selected.state
                      .confidence *
                      100
                  )}
                  %
                </p>

                <small>
                  Model:{" "}
                  {
                    selected.state
                      .model_version
                  }
                </small>
              </div>
            ) : null}
          </section>
        </>
      )}
    </main>
  );
}