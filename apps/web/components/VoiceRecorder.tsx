"use client";

import {
  useEffect,
  useRef,
  useState,
} from "react";

type Props = {
  disabled?: boolean;
  onRecordingReady: (
    audio: Blob,
    filename: string
  ) => Promise<void> | void;
};

type RecorderState =
  | "IDLE"
  | "REQUESTING_PERMISSION"
  | "RECORDING"
  | "RECORDED"
  | "UPLOADING";

const MAX_RECORDING_SECONDS = 5 * 60;

function formatTime(totalSeconds: number) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;

  return `${minutes}:${seconds
    .toString()
    .padStart(2, "0")}`;
}

function getSupportedMimeType(): string {
  if (
    typeof MediaRecorder === "undefined"
  ) {
    return "";
  }

  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
  ];

  for (const candidate of candidates) {
    if (
      MediaRecorder.isTypeSupported(candidate)
    ) {
      return candidate;
    }
  }

  return "";
}

function extensionForMimeType(
  mimeType: string
): string {
  if (mimeType.includes("mp4")) {
    return "m4a";
  }

  if (mimeType.includes("mpeg")) {
    return "mp3";
  }

  if (mimeType.includes("wav")) {
    return "wav";
  }

  return "webm";
}

function getMicrophoneErrorMessage(
  error: unknown
): string {
  if (
    error instanceof DOMException
  ) {
    if (
      error.name === "NotAllowedError" ||
      error.name === "SecurityError"
    ) {
      return "Microphone access was denied. Please allow microphone access in your browser and try again.";
    }

    if (
      error.name === "NotFoundError" ||
      error.name === "DevicesNotFoundError"
    ) {
      return "No microphone was found on this device.";
    }

    if (
      error.name === "NotReadableError" ||
      error.name === "TrackStartError"
    ) {
      return "Your microphone could not be accessed. It may already be in use by another application.";
    }
  }

  return "Continuum could not access your microphone.";
}

export default function VoiceRecorder({
  disabled = false,
  onRecordingReady,
}: Props) {
  const mediaRecorderRef =
    useRef<MediaRecorder | null>(null);

  const streamRef =
    useRef<MediaStream | null>(null);

  const chunksRef =
    useRef<Blob[]>([]);

  const timerRef =
    useRef<ReturnType<
      typeof setInterval
    > | null>(null);

  const previewUrlRef =
    useRef<string | null>(null);

  const [recorderState, setRecorderState] =
    useState<RecorderState>("IDLE");

  const [recordingSeconds, setRecordingSeconds] =
    useState(0);

  const [audioBlob, setAudioBlob] =
    useState<Blob | null>(null);

  const [previewUrl, setPreviewUrl] =
    useState<string | null>(null);

  const [filename, setFilename] =
    useState<string | null>(null);

  const [error, setError] =
    useState<string | null>(null);

  const isRecording =
    recorderState === "RECORDING";

  const isUploading =
    recorderState === "UPLOADING";

  const isBusy =
    recorderState ===
      "REQUESTING_PERMISSION" ||
    isUploading;

  function clearTimer() {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  function stopMediaStream() {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }

      streamRef.current = null;
    }
  }

  function revokePreviewUrl() {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(
        previewUrlRef.current
      );

      previewUrlRef.current = null;
    }

    setPreviewUrl(null);
  }

  function cleanupRecorder() {
    clearTimer();
    stopMediaStream();

    mediaRecorderRef.current = null;
    chunksRef.current = [];
  }

  useEffect(() => {
    return () => {
      clearTimer();
      stopMediaStream();

      if (previewUrlRef.current) {
        URL.revokeObjectURL(
          previewUrlRef.current
        );
      }
    };
  }, []);

  async function startRecording() {
    setError(null);

    if (
      typeof window === "undefined" ||
      typeof navigator === "undefined"
    ) {
      setError(
        "Voice recording is not available in this environment."
      );
      return;
    }

    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices.getUserMedia
    ) {
      setError(
        "This browser does not support microphone recording."
      );
      return;
    }

    if (
      typeof MediaRecorder === "undefined"
    ) {
      setError(
        "This browser does not support audio recording."
      );
      return;
    }

    setRecorderState(
      "REQUESTING_PERMISSION"
    );

    try {
      const stream =
        await navigator.mediaDevices.getUserMedia({
          audio: true,
        });

      streamRef.current = stream;

      const mimeType =
        getSupportedMimeType();

      let recorder: MediaRecorder;

      if (mimeType) {
        recorder = new MediaRecorder(
          stream,
          {
            mimeType,
          }
        );
      } else {
        recorder = new MediaRecorder(
          stream
        );
      }

      mediaRecorderRef.current =
        recorder;

      chunksRef.current = [];

      revokePreviewUrl();

      setAudioBlob(null);
      setFilename(null);
      setRecordingSeconds(0);

      recorder.ondataavailable = (
        event: BlobEvent
      ) => {
        if (event.data.size > 0) {
          chunksRef.current.push(
            event.data
          );
        }
      };

      recorder.onerror = () => {
        setError(
          "An error occurred while recording audio."
        );

        cleanupRecorder();

        setRecorderState("IDLE");
      };

      recorder.onstop = () => {
        clearTimer();
        stopMediaStream();

        const actualMimeType =
          recorder.mimeType ||
          mimeType ||
          "audio/webm";

        const blob = new Blob(
          chunksRef.current,
          {
            type: actualMimeType,
          }
        );

        chunksRef.current = [];
        mediaRecorderRef.current = null;

        if (blob.size === 0) {
          setError(
            "The recording was empty. Please try again."
          );

          setRecorderState("IDLE");
          return;
        }

        const extension =
          extensionForMimeType(
            actualMimeType
          );

        const recordingFilename =
          `continuum-${Date.now()}.${extension}`;

        const objectUrl =
          URL.createObjectURL(blob);

        if (previewUrlRef.current) {
          URL.revokeObjectURL(
            previewUrlRef.current
          );
        }

        previewUrlRef.current =
          objectUrl;

        setAudioBlob(blob);
        setFilename(recordingFilename);
        setPreviewUrl(objectUrl);

        setRecorderState("RECORDED");
      };

      recorder.start(1000);

      setRecorderState("RECORDING");

      timerRef.current = setInterval(
        () => {
          setRecordingSeconds(
            (current) => current + 1
          );
        },
        1000
      );
    } catch (err) {
      cleanupRecorder();

      setRecorderState("IDLE");

      setError(
        getMicrophoneErrorMessage(err)
      );
    }
  }

  function stopRecording() {
    const recorder =
      mediaRecorderRef.current;

    if (
      !recorder ||
      recorder.state === "inactive"
    ) {
      return;
    }

    recorder.stop();
  }

  function discardRecording() {
    if (isRecording) {
      const recorder =
        mediaRecorderRef.current;

      if (
        recorder &&
        recorder.state !== "inactive"
      ) {
        recorder.onstop = null;
        recorder.stop();
      }
    }

    cleanupRecorder();
    revokePreviewUrl();

    setAudioBlob(null);
    setFilename(null);
    setRecordingSeconds(0);
    setError(null);

    setRecorderState("IDLE");
  }

  async function uploadRecording() {
    if (
      !audioBlob ||
      !filename ||
      isUploading
    ) {
      return;
    }

    setError(null);
    setRecorderState("UPLOADING");

    try {
      await onRecordingReady(
        audioBlob,
        filename
      );

      revokePreviewUrl();

      setAudioBlob(null);
      setFilename(null);
      setRecordingSeconds(0);

      setRecorderState("IDLE");
    } catch (err) {
      setRecorderState("RECORDED");

      setError(
        err instanceof Error
          ? err.message
          : "Could not upload the voice journal."
      );
    }
  }

  useEffect(() => {
    if (
      recorderState !== "RECORDING"
    ) {
      return;
    }

    if (
      recordingSeconds <
      MAX_RECORDING_SECONDS
    ) {
      return;
    }

    stopRecording();
  }, [
    recordingSeconds,
    recorderState,
  ]);

  return (
    <section>
      <h3>Voice journal</h3>

      <p>
        Record a private voice journal.
        You can listen to it before
        uploading.
      </p>

      {recorderState === "IDLE" && (
        <button
          type="button"
          onClick={() =>
            void startRecording()
          }
          disabled={disabled}
        >
          Start recording
        </button>
      )}

      {recorderState ===
        "REQUESTING_PERMISSION" && (
        <p>
          Waiting for microphone
          permission...
        </p>
      )}

      {recorderState ===
        "RECORDING" && (
        <div>
          <p>
            Recording ·{" "}
            {formatTime(
              recordingSeconds
            )}
          </p>

          <button
            type="button"
            onClick={stopRecording}
          >
            Stop recording
          </button>

          <button
            type="button"
            onClick={
              discardRecording
            }
          >
            Cancel
          </button>
        </div>
      )}

      {recorderState ===
        "RECORDED" &&
        audioBlob &&
        previewUrl && (
          <div>
            <p>
              Recording ready ·{" "}
              {formatTime(
                recordingSeconds
              )}
            </p>

            <audio
              controls
              src={previewUrl}
            >
              Your browser does not
              support audio playback.
            </audio>

            <div>
              <button
                type="button"
                onClick={() =>
                  void uploadRecording()
                }
                disabled={disabled}
              >
                Save voice journal
              </button>

              <button
                type="button"
                onClick={
                  discardRecording
                }
              >
                Discard
              </button>
            </div>
          </div>
        )}

      {recorderState ===
        "UPLOADING" && (
        <p>
          Uploading voice journal...
        </p>
      )}

      {isBusy &&
      recorderState !==
        "UPLOADING" &&
      recorderState !==
        "REQUESTING_PERMISSION" ? (
        <p>Please wait...</p>
      ) : null}

      {error && (
        <p role="alert">
          {error}
        </p>
      )}
    </section>
  );
}