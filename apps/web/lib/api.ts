export type Journal = {
  id: string;
  entry_type: "TEXT";
  text: string;
  status: "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED";
  error_message?: string | null;
  created_at: string;
  updated_at: string;
  state?: {
    energy: number;
    stress: number;
    confidence: number;
    model_version: string;
  } | null;
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

declare const process: {
  env: {
    NEXT_PUBLIC_API_BASE_URL?: string;
  };
};

const API = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...options,
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
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

export function register(payload: RegisterPayload): Promise<User> {
  return apiFetch<User>("/v1/auth/register", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function login(
  payload: LoginPayload
): Promise<{ message: string }> {
  return apiFetch<{ message: string }>("/v1/auth/login", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getCurrentUser(): Promise<User> {
  return apiFetch<User>("/v1/auth/me");
}

export function logout(): Promise<{ message: string }> {
  return apiFetch<{ message: string }>("/v1/auth/logout", {
    method: "POST",
  });
}

/* -------------------------------------------------------------------------- */
/* Journals                                                                   */
/* -------------------------------------------------------------------------- */

export function createJournal(
  text: string
): Promise<{ id: string; status: string }> {
  return apiFetch<{ id: string; status: string }>("/v1/journals", {
    method: "POST",
    body: JSON.stringify({
      entry_type: "TEXT",
      text,
    }),
  });
}

export function getJournal(id: string): Promise<Journal> {
  return apiFetch<Journal>(`/v1/journals/${id}`, {
    cache: "no-store",
  });
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
): Promise<{ id: string; status: string }> {
  return apiFetch<{ id: string; status: string }>(
    `/v1/journals/${id}`,
    {
      method: "PATCH",
      body: JSON.stringify({ text }),
    }
  );
}

export function deleteJournal(id: string): Promise<void> {
  return apiFetch<void>(`/v1/journals/${id}`, {
    method: "DELETE",
  });
}