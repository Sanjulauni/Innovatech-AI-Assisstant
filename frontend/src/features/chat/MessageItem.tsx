import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { Bot, RotateCcw } from "lucide-react";
import { useState } from "react";

import { Alert, Button } from "../../components/ui";
import { getHealth } from "../../lib/api";
import { AnswerMarkdown } from "./AnswerMarkdown";
import { SourceList } from "./SourceList";
import type { Message } from "./useChat";

interface Props {
  message: Message;
  onRetry?: () => void;
}

export function MessageItem({ message, onRetry }: Props) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [highlighted, setHighlighted] = useState<number | null>(null);
  // Shares the header's cached health; tells us whether the local model is answering.
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth });
  const localModel = health.data?.llm_provider === "local";

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-brand-600 px-4 py-2.5 text-[15px] text-white shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  const streaming = message.status === "streaming";
  const cite = (index: number) => {
    setSourcesOpen(true);
    setHighlighted(index);
  };

  return (
    <div className="flex gap-3" aria-busy={streaming}>
      <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-brand-50 text-brand-600 ring-1 ring-brand-100 dark:bg-brand-500/15 dark:text-brand-100 dark:ring-brand-500/30">
        <Bot className="size-4" aria-hidden />
      </div>

      <div className="min-w-0 flex-1 pt-1">
        {message.content ? (
          <div className={clsx(streaming && "typing-cursor")}>
            <AnswerMarkdown text={message.content} onCite={cite} />
          </div>
        ) : streaming ? (
          <div className="flex h-7 items-center gap-1" aria-label="Searching the documents">
            {[0, 150, 300].map((delay) => (
              <span
                key={delay}
                className="size-1.5 animate-bounce rounded-full bg-slate-400"
                style={{ animationDelay: `${delay}ms` }}
              />
            ))}
            <span className="ml-2 text-sm text-slate-500 dark:text-slate-400">
              Searching the documents…
            </span>
          </div>
        ) : null}
        {streaming && !message.content && localModel && (
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            The local model runs on this computer, so the first words can take about a minute.
          </p>
        )}

        {message.status === "stopped" && (
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">Stopped.</p>
        )}

        {message.status === "error" && (
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <Alert tone="error">{message.error}</Alert>
            {onRetry && (
              <Button variant="secondary" size="sm" onClick={onRetry} icon={<RotateCcw className="size-3.5" />}>
                Try again
              </Button>
            )}
          </div>
        )}

        {message.sources && (
          <SourceList
            sources={message.sources}
            open={sourcesOpen}
            onToggle={() => setSourcesOpen((open) => !open)}
            highlighted={highlighted}
          />
        )}
      </div>
    </div>
  );
}
