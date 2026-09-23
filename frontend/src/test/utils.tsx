import { QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { vi } from "vitest";

import { AppRoutes } from "../App";
import { createQueryClient } from "../lib/queryClient";

export function renderApp(route = "/") {
  const client = createQueryClient();
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[route]}>
        <AppRoutes />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

export const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

/** A streaming NDJSON response; each string is sent as its own network chunk. */
export function streamResponse(chunks: string[]) {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(body, { headers: { "Content-Type": "application/x-ndjson" } });
}

export const ndjson = (...events: object[]) => events.map((e) => JSON.stringify(e) + "\n");

export const HEALTH = {
  status: "ok",
  vector_store: "ok",
  documents: 2,
  llm_provider: "groq",
  llm_model: "openai/gpt-oss-120b",
  embedding_model: "BAAI/bge-small-en-v1.5",
  admin_enabled: true,
};

type Handler = (url: string, init: RequestInit) => Response | Promise<Response>;

/**
 * Replace fetch with a router: keys are "METHOD /path" (path without /api).
 * Returns the mock so tests can inspect calls.
 */
export function mockApi(routes: Record<string, Handler>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = typeof input === "string" ? input : input.toString();
    const path = url.replace(/^\/api/, "");
    const key = `${(init.method ?? "GET").toUpperCase()} ${path}`;
    const handler = routes[key] ?? (key === "GET /health" ? () => jsonResponse(HEALTH) : undefined);
    if (!handler) throw new Error(`Unexpected request: ${key}`);
    return handler(url, init);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

/** Calls made to one route, with their parsed JSON bodies. */
export function callsTo(fetchMock: ReturnType<typeof mockApi>, method: string, path: string) {
  return fetchMock.mock.calls
    .filter(([url, init]) => url === `/api${path}` && (init?.method ?? "GET") === method)
    .map(([, init]) =>
      typeof init?.body === "string" ? (JSON.parse(init.body) as unknown) : init?.body,
    );
}
