import { FileSearch, SquarePen } from "lucide-react";
import { useEffect, useRef } from "react";

import { Button } from "../../components/ui";
import { Composer } from "./Composer";
import { MessageItem } from "./MessageItem";
import { useChat } from "./useChat";

const SUGGESTIONS = [
  "How many days of annual leave do I get?",
  "How do I request to work remotely?",
  "What should I do if I lose my laptop?",
  "What are the password requirements?",
];

function EmptyState({ onPick }: { onPick: (question: string) => void }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-4 py-12 text-center">
      <div className="flex size-12 items-center justify-center rounded-2xl bg-brand-50 text-brand-600 ring-1 ring-brand-100 dark:bg-brand-500/15 dark:text-brand-100 dark:ring-brand-500/30">
        <FileSearch className="size-6" aria-hidden />
      </div>
      <h1 className="mt-4 text-2xl font-semibold tracking-tight">How can I help?</h1>
      <p className="mt-2 max-w-md text-sm text-slate-500 dark:text-slate-400">
        Ask about company policies, manuals and guidelines. Every answer comes from the
        company documents and shows its sources.
      </p>
      <div className="mt-8 grid w-full max-w-xl gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map((question) => (
          <button
            key={question}
            type="button"
            onClick={() => onPick(question)}
            className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-left text-sm text-slate-700 shadow-sm transition-colors hover:border-brand-500/50 hover:bg-brand-50/40 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-brand-500/10"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}

export function ChatPage() {
  const { messages, isStreaming, ask, stop, retry, clear } = useChat();
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  // Follow new text unless the reader has scrolled up to re-read something.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (el) stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  const lastIndex = messages.length - 1;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div ref={scrollRef} onScroll={onScroll} className="flex min-h-0 flex-1 flex-col overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState onPick={ask} />
        ) : (
          <div className="mx-auto w-full max-w-3xl space-y-6 px-4 py-8" aria-live="polite">
            {messages.map((message, index) => (
              <MessageItem
                key={message.id}
                message={message}
                onRetry={index === lastIndex ? retry : undefined}
              />
            ))}
          </div>
        )}
      </div>

      <div className="border-t border-slate-200 bg-slate-50/90 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
        <div className="mx-auto w-full max-w-3xl px-4 pb-4 pt-3">
          <Composer isStreaming={isStreaming} onSend={ask} onStop={stop} />
          <div className="mt-2 flex items-center justify-between gap-2 px-1 text-xs text-slate-500 dark:text-slate-400">
            <span>Answers can be wrong. Check the sources for important decisions.</span>
            {messages.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={clear}
                icon={<SquarePen className="size-3.5" />}
                className="h-7 px-2 text-xs"
              >
                New chat
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
