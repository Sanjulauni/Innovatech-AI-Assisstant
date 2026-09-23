import { ArrowUp, Square } from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";

export const MAX_QUESTION_LENGTH = 2000;

interface Props {
  isStreaming: boolean;
  onSend: (question: string) => void;
  onStop: () => void;
}

export function Composer({ isStreaming, onSend, onStop }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  const canSend = text.trim().length > 0 && !isStreaming;

  // Grow with the text, up to about 8 lines.
  const fit = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, []);

  useLayoutEffect(fit, [text, fit]);

  // Re-measure when the width changes (and once styles have loaded after mount).
  useEffect(() => {
    const frame = requestAnimationFrame(fit);
    window.addEventListener("resize", fit);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", fit);
    };
  }, [fit]);

  const submit = () => {
    if (!canSend) return;
    onSend(text);
    setText("");
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  };

  const nearLimit = text.length > MAX_QUESTION_LENGTH * 0.9;

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
      className="rounded-2xl border border-slate-200 bg-white p-2 shadow-sm focus-within:border-brand-500/60 focus-within:ring-4 focus-within:ring-brand-500/10 dark:border-slate-700 dark:bg-slate-900"
    >
      <div className="flex items-end gap-2">
        <label htmlFor="question" className="sr-only">
          Your question
        </label>
        <textarea
          id="question"
          ref={ref}
          rows={1}
          value={text}
          maxLength={MAX_QUESTION_LENGTH}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about policies, manuals or guidelines…"
          className="max-h-[200px] min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-[15px] placeholder:text-slate-400 focus:outline-none"
          autoFocus
        />
        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-slate-900 text-white hover:bg-slate-700 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-300"
            aria-label="Stop generating"
          >
            <Square className="size-3.5 fill-current" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={!canSend}
            className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-brand-600 text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 dark:disabled:bg-slate-800 dark:disabled:text-slate-600"
            aria-label="Send question"
          >
            <ArrowUp className="size-5" />
          </button>
        )}
      </div>
      {nearLimit && (
        <p className="px-2 pt-1 text-right text-xs text-slate-500">
          {text.length} / {MAX_QUESTION_LENGTH}
        </p>
      )}
    </form>
  );
}
