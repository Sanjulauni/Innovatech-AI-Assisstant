import type {
  ChatMessageIn,
  ChatResponse,
  Health,
  Instructions,
  Models,
  ReindexResult,
  StoredDocument,
  StreamEvent,
  UploadLimit,
  UploadResult,
} from "./types";

const BASE = "/api";

/** An API failure with a message that is safe to show to the user. */
export class ApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function errorFrom(response: Response): Promise<ApiError> {
  let detail: unknown;
  try {
    detail = (await response.json()).detail;
  } catch {
    detail = undefined;
  }
  if (typeof detail === "string" && detail) return new ApiError(detail, response.status);
  if (response.status === 422) {
    return new ApiError("The request was not valid. Please check your input.", 422);
  }
  return new ApiError(
    `The server returned an error (${response.status}). Please try again.`,
    response.status,
  );
}

async function send(path: string, init: RequestInit = {}): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(BASE + path, { credentials: "same-origin", ...init });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError("Can't reach the assistant server. Check that the API is running.");
  }
  if (!response.ok) throw await errorFrom(response);
  return response;
}

async function json<T>(path: string, init: RequestInit = {}): Promise<T> {
  return (await send(path, init)).json() as Promise<T>;
}

function jsonBody(method: string, body: unknown): RequestInit {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

// --- Employee --------------------------------------------------------------------

export const getHealth = () => json<Health>("/health");

export const chat = (question: string, history: ChatMessageIn[] = []) =>
  json<ChatResponse>("/chat", jsonBody("POST", { question, history }));

/**
 * Stream an answer. Calls `onEvent` for every event as it arrives and resolves once
 * the stream ends. Pass an AbortSignal to stop generating.
 */
export async function streamChat(
  question: string,
  history: ChatMessageIn[],
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await send("/chat/stream", { ...jsonBody("POST", { question, history }), signal });
  if (!response.body) throw new ApiError("The server sent an empty response.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line) as StreamEvent);
    }
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer) as StreamEvent);
}

// --- Admin (session cookie set by login) -------------------------------------------

export const login = (password: string) =>
  json<{ authenticated: boolean; expires_in: number | null }>(
    "/admin/login",
    jsonBody("POST", { password }),
  );

export const logout = () => send("/admin/logout", { method: "POST" }).then(() => undefined);

/** Resolves to true if the admin session cookie is still valid. */
export async function checkSession(): Promise<boolean> {
  try {
    await send("/admin/session");
    return true;
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) return false;
    throw error;
  }
}

export const listDocuments = () => json<StoredDocument[]>("/admin/documents");

export function uploadDocument(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  return json<UploadResult>("/admin/documents", { method: "POST", body: form });
}

export const deleteDocument = (docId: string) =>
  send(`/admin/documents/${encodeURIComponent(docId)}`, { method: "DELETE" }).then(() => undefined);

export const reindexDocuments = () =>
  json<ReindexResult>("/admin/documents/reindex", { method: "POST" });

export const getUploadLimit = () => json<UploadLimit>("/admin/upload-limit");

export const saveUploadLimit = (mb: number) =>
  json<UploadLimit>("/admin/upload-limit", jsonBody("PUT", { max_upload_size_mb: mb }));

export const getModels = () => json<Models>("/admin/model");

export const selectModel = (model: string) => json<Models>("/admin/model", jsonBody("PUT", { model }));

export const getInstructions = () => json<Instructions>("/admin/instructions");

export const saveInstructions = (text: string) =>
  json<Instructions>("/admin/instructions", jsonBody("PUT", { text }));
