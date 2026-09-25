// Shapes returned by the FastAPI backend (see src/api/schemas.py).

export type Role = "user" | "assistant";

export interface ChatMessageIn {
  role: Role;
  content: string;
}

export interface Source {
  index: number;
  doc_id: string;
  source: string;
  page: number | null;
  snippet: string;
  score: number;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
}

export type StreamEvent =
  | { type: "token"; text: string }
  | ({ type: "done" } & ChatResponse)
  | { type: "error"; message: string };

export interface Health {
  status: "ok" | "degraded";
  vector_store: "ok" | "error";
  documents: number;
  llm_provider: string;
  llm_model: string;
  embedding_model: string;
  admin_enabled: boolean;
}

export interface StoredDocument {
  doc_id: string;
  source: string;
  chunk_count: number;
  ingested_at: string;
}

export interface UploadResult {
  doc_id: string;
  source: string;
  status: "ingested" | "duplicate";
  chunk_count: number;
  message: string;
}

export interface Instructions {
  text: string;
  updated_at: string | null;
}

export type ModelStatus = "stopped" | "starting" | "ready" | "error";

export interface ModelOption {
  id: string;
  label: string;
  description: string;
  /** "cloud" = Groq; "local" = runs on the API's machine. */
  kind: "cloud" | "local";
  /** Cloud models are always "ready"; the local model starts when selected. */
  status: ModelStatus;
  /** Why the local model is in error. */
  detail: string;
}

export interface Models {
  current: string;
  options: ModelOption[];
}

export interface UploadLimit {
  max_upload_size_mb: number;
  default_mb: number;
  max_allowed_mb: number;
}

export interface ReindexResult {
  indexed: number;
  skipped: number;
  failed: Record<string, string>;
}
