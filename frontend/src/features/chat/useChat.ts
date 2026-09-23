import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, streamChat } from "../../lib/api";
import type { ChatMessageIn, Role, Source } from "../../lib/types";

export type MessageStatus = "streaming" | "done" | "stopped" | "error";

export interface Message {
  id: string;
  role: Role;
  content: string;
  sources?: Source[];
  status: MessageStatus;
  error?: string;
}

const STORAGE_KEY = "innovatech.chat";

function load(): Message[] {
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) ?? "[]") as Message[];
    // An answer that was still streaming when the page closed is incomplete.
    return saved.map((m) => (m.status === "streaming" ? { ...m, status: "stopped" } : m));
  } catch {
    return [];
  }
}

function save(messages: Message[]) {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  } catch {
    // Storage full or blocked: the chat still works, it just won't survive a reload.
  }
}

let counter = 0;
const newId = () => `${Date.now().toString(36)}-${(counter++).toString(36)}`;

/** Earlier completed turns, in the shape the API expects. */
function historyOf(messages: Message[]): ChatMessageIn[] {
  return messages
    .filter((m) => m.status !== "error" && m.content.trim())
    .map((m) => ({ role: m.role, content: m.content }));
}

/** Chat state for this browser tab: messages, streaming, stop, retry and clear. */
export function useChat() {
  const [messages, setMessages] = useState<Message[]>(load);
  const abortRef = useRef<AbortController | null>(null);
  const isStreaming = messages.some((m) => m.status === "streaming");

  useEffect(() => save(messages), [messages]);
  useEffect(() => () => abortRef.current?.abort(), []);

  const update = useCallback((id: string, change: (m: Message) => Partial<Message>) => {
    setMessages((all) => all.map((m) => (m.id === id ? { ...m, ...change(m) } : m)));
  }, []);

  const send = useCallback(
    async (question: string, previous: Message[]) => {
      const text = question.trim();
      if (!text) return;

      const assistantId = newId();
      setMessages([
        ...previous,
        { id: newId(), role: "user", content: text, status: "done" },
        { id: assistantId, role: "assistant", content: "", status: "streaming" },
      ]);

      const controller = new AbortController();
      abortRef.current = controller;
      try {
        await streamChat(
          text,
          historyOf(previous),
          (event) => {
            if (event.type === "token") {
              update(assistantId, (m) => ({ content: m.content + event.text }));
            } else if (event.type === "done") {
              update(assistantId, () => ({
                content: event.answer,
                sources: event.sources,
                status: "done",
              }));
            } else {
              update(assistantId, () => ({ status: "error", error: event.message }));
            }
          },
          controller.signal,
        );
        // A stream that ended without a "done" event was cut off.
        update(assistantId, (m) =>
          m.status === "streaming"
            ? { status: "error", error: "The answer was interrupted. Please try again." }
            : {},
        );
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          update(assistantId, () => ({ status: "stopped" }));
        } else {
          const message =
            error instanceof ApiError ? error.message : "Something went wrong. Please try again.";
          update(assistantId, () => ({ status: "error", error: message }));
        }
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
      }
    },
    [update],
  );

  const ask = useCallback(
    (question: string) => {
      if (!isStreaming) void send(question, messages);
    },
    [isStreaming, messages, send],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  /** Re-ask the last question after a failed answer. */
  const retry = useCallback(() => {
    const last = messages.at(-1);
    const question = messages.at(-2);
    if (isStreaming || last?.status !== "error" || question?.role !== "user") return;
    void send(question.content, messages.slice(0, -2));
  }, [isStreaming, messages, send]);

  const clear = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
  }, []);

  return { messages, isStreaming, ask, stop, retry, clear };
}
