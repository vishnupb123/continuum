export type JournalStatus =
  | "QUEUED"
  | "PROCESSING"
  | "COMPLETED"
  | "FAILED";

export type TranscriptionStatus =
  | "TRANSCRIPTION_QUEUED"
  | "TRANSCRIBING"
  | "TRANSCRIPTION_COMPLETED"
  | "TRANSCRIPTION_FAILED";

export type JournalAudio = {
  id: string;
  original_filename: string | null;
  mime_type: string;
  size_bytes: number;
  duration_seconds: number | null;
  transcription_status: TranscriptionStatus;
};

export type JournalState = {
  energy: number;
  stress: number;
  confidence: number;
  model_version: string;
};

export type Journal = {
  id: string;
  entry_type: "TEXT" | "VOICE";
  text: string | null;
  status: JournalStatus;
  error_message?: string | null;
  created_at: string;
  updated_at: string;
  state?: JournalState | null;
  audio?: JournalAudio | null;
};

export type JournalListResponse = {
  items: Journal[];
  total: number;
  limit: number;
  offset: number;
};

export type User = {
  id: string;
  email: string;
  display_name: string | null;
  is_active: boolean;
  created_at: string;
};

export type RegisterPayload = {
  email: string;
  password: string;
  display_name?: string;
};

export type LoginPayload = {
  email: string;
  password: string;
};

export type JournalAccepted = {
  id: string;
  status: JournalStatus;
};

declare const process: {
  env: {
    NEXT_PUBLIC_API_BASE_URL?: string;
  };
};

const API =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  "http://localhost:8000";


async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const headers = new Headers(options.headers);

  /*
   * Do not manually set Content-Type for FormData.
   *
   * The browser must generate:
   *
   * multipart/form-data; boundary=...
   *
   * Setting it manually would omit the boundary and break
   * FastAPI's UploadFile parsing.
   */
  if (
    options.body !== undefined &&
    !(options.body instanceof FormData) &&
    !headers.has("Content-Type")
  ) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API}${path}`, {
    ...options,
    credentials: "include",
    headers,
  });

  if (!response.ok) {
    let message = "Something went wrong";

    try {
      const body = await response.json();

      if (typeof body.detail === "string") {
        message = body.detail;
      }
    } catch {
      // Keep the fallback error message.
    }

    throw new Error(message);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}


/* -------------------------------------------------------------------------- */
/* Authentication                                                             */
/* -------------------------------------------------------------------------- */

export function register(
  payload: RegisterPayload
): Promise<User> {
  return apiFetch<User>("/v1/auth/register", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}


export function login(
  payload: LoginPayload
): Promise<{ message: string }> {
  return apiFetch<{ message: string }>(
    "/v1/auth/login",
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}


export function getCurrentUser(): Promise<User> {
  return apiFetch<User>("/v1/auth/me");
}


export function logout(): Promise<{ message: string }> {
  return apiFetch<{ message: string }>(
    "/v1/auth/logout",
    {
      method: "POST",
    }
  );
}


/* -------------------------------------------------------------------------- */
/* Journals                                                                   */
/* -------------------------------------------------------------------------- */

export function createJournal(
  text: string
): Promise<JournalAccepted> {
  return apiFetch<JournalAccepted>(
    "/v1/journals",
    {
      method: "POST",
      body: JSON.stringify({
        entry_type: "TEXT",
        text,
      }),
    }
  );
}


export function createVoiceJournal(
  audio: Blob,
  filename = "recording.webm"
): Promise<JournalAccepted> {
  const formData = new FormData();

  formData.append(
    "file",
    audio,
    filename
  );

  return apiFetch<JournalAccepted>(
    "/v1/journals/voice",
    {
      method: "POST",
      body: formData,
    }
  );
}

export async function getJournalAudio(
  id: string
): Promise<Blob> {
  const response = await fetch(
    `${API}/v1/journals/${id}/audio`,
    {
      method: "GET",
      credentials: "include",
      cache: "no-store",
    }
  );

  if (!response.ok) {
    let message =
      "Could not load journal audio";

    try {
      const body = await response.json();

      if (
        typeof body.detail ===
        "string"
      ) {
        message = body.detail;
      }
    } catch {}

    throw new Error(message);
  }

  return response.blob();
}


export function getJournal(
  id: string
): Promise<Journal> {
  return apiFetch<Journal>(
    `/v1/journals/${id}`,
    {
      cache: "no-store",
    }
  );
}


export function listJournals(
  limit = 20,
  offset = 0
): Promise<JournalListResponse> {
  return apiFetch<JournalListResponse>(
    `/v1/journals?limit=${limit}&offset=${offset}`,
    {
      cache: "no-store",
    }
  );
}


export function updateJournal(
  id: string,
  text: string
): Promise<JournalAccepted> {
  return apiFetch<JournalAccepted>(
    `/v1/journals/${id}`,
    {
      method: "PATCH",
      body: JSON.stringify({ text }),
    }
  );
}


export function deleteJournal(
  id: string
): Promise<void> {
  return apiFetch<void>(
    `/v1/journals/${id}`,
    {
      method: "DELETE",
    }
  );
}