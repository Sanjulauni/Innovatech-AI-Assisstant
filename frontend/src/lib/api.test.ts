import { describe, expect, it, vi } from "vitest";

import { jsonResponse, mockApi, ndjson, streamResponse } from "../test/utils";
import { ApiError, chat, checkSession, streamChat } from "./api";
import type { StreamEvent } from "./types";

describe("streamChat", () => {
  it("parses events split across network chunks", async () => {
    const lines = ndjson(
      { type: "token", text: "Leave is " },
      { type: "token", text: "18 days [1]." },
      { type: "done", answer: "Leave is 18 days [1].", sources: [] },
    ).join("");
    // Cut the stream at awkward places, including inside a JSON object.
    mockApi({ "POST /chat/stream": () => streamResponse([lines.slice(0, 7), lines.slice(7, 40), lines.slice(40)]) });

    const events: StreamEvent[] = [];
    await streamChat("q", [], (e) => events.push(e));

    expect(events.map((e) => e.type)).toEqual(["token", "token", "done"]);
    expect(events[1]).toEqual({ type: "token", text: "18 days [1]." });
  });

  it("sends the question and history", async () => {
    const fetchMock = mockApi({
      "POST /chat/stream": () => streamResponse(ndjson({ type: "done", answer: "a", sources: [] })),
    });

    await streamChat("follow-up", [{ role: "user", content: "first" }], () => {});

    const body = JSON.parse(fetchMock.mock.calls[0][1]!.body as string);
    expect(body).toEqual({ question: "follow-up", history: [{ role: "user", content: "first" }] });
  });

  it("reports HTTP errors with the server's message", async () => {
    mockApi({
      "POST /chat/stream": () => jsonResponse({ detail: "The AI model is busy." }, 503),
    });

    const error = await streamChat("q", [], () => {}).catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).toBe("The AI model is busy.");
    expect((error as ApiError).status).toBe(503);
  });
});

describe("error messages", () => {
  it("explains when the server can't be reached", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(chat("q")).rejects.toThrow("Can't reach the assistant server");
  });

  it("falls back to a generic message without a detail", async () => {
    mockApi({ "POST /chat": () => new Response("<html>oops</html>", { status: 500 }) });
    await expect(chat("q")).rejects.toThrow("server returned an error (500)");
  });

  it("explains validation errors", async () => {
    mockApi({ "POST /chat": () => jsonResponse({ detail: [{ msg: "bad" }] }, 422) });
    await expect(chat("q")).rejects.toThrow("request was not valid");
  });
});

describe("checkSession", () => {
  it("is true with a valid session and false on 401", async () => {
    mockApi({ "GET /admin/session": () => jsonResponse({ authenticated: true }) });
    await expect(checkSession()).resolves.toBe(true);

    mockApi({ "GET /admin/session": () => jsonResponse({ detail: "Incorrect" }, 401) });
    await expect(checkSession()).resolves.toBe(false);
  });

  it("rethrows other errors (e.g. admin disabled)", async () => {
    mockApi({ "GET /admin/session": () => jsonResponse({ detail: "disabled" }, 503) });
    await expect(checkSession()).rejects.toThrow("disabled");
  });
});
