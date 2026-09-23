import type { QueryClient } from "@tanstack/react-query";

import { ApiError } from "../../lib/api";

export const SESSION_KEY = ["admin", "session"] as const;
export const DOCUMENTS_KEY = ["admin", "documents"] as const;
export const INSTRUCTIONS_KEY = ["admin", "instructions"] as const;
export const MODELS_KEY = ["admin", "models"] as const;

/** Any admin call answered with 401 means the session ended: show the login form. */
export function handleUnauthorized(client: QueryClient, error: unknown) {
  if (error instanceof ApiError && error.status === 401) {
    client.setQueryData(SESSION_KEY, false);
  }
}
