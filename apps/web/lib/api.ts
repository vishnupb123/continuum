export type Journal = {
  id: string;
  entry_type: "TEXT";
  text: string;
  status: "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED";
  error_message?: string | null;
  state?: {
    energy: number;
    stress: number;
    confidence: number;
    model_version: string;
  } | null;
};

const API = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function createJournal(text: string): Promise<{ id: string; status: string }> {
  const response = await fetch(`${API}/v1/journals`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entry_type: "TEXT", text }),
  });
  if (!response.ok) throw new Error("Could not create journal");
  return response.json();
}

export async function getJournal(id: string): Promise<Journal> {
  const response = await fetch(`${API}/v1/journals/${id}`, { cache: "no-store" });
  if (!response.ok) throw new Error("Could not load journal");
  return response.json();
}
