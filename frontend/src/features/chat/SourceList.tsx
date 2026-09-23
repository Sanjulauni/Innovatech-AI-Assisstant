import clsx from "clsx";
import { ChevronDown, FileText } from "lucide-react";
import { useEffect, useRef } from "react";

import type { Source } from "../../lib/types";

interface Props {
  sources: Source[];
  open: boolean;
  onToggle: () => void;
  highlighted: number | null;
}

export function SourceList({ sources, open, onToggle, highlighted }: Props) {
  const refs = useRef(new Map<number, HTMLLIElement>());

  useEffect(() => {
    if (open && highlighted !== null) {
      refs.current.get(highlighted)?.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
    }
  }, [open, highlighted]);

  if (sources.length === 0) return null;

  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="inline-flex items-center gap-1 text-xs font-medium text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200"
      >
        <ChevronDown className={clsx("size-3.5 transition-transform", open && "rotate-180")} />
        {sources.length} source{sources.length === 1 ? "" : "s"}
      </button>

      {open && (
        <ul className="mt-2 grid gap-2">
          {sources.map((source) => (
            <li
              key={source.index}
              ref={(el) => {
                if (el) refs.current.set(source.index, el);
                else refs.current.delete(source.index);
              }}
              className={clsx(
                "rounded-xl border p-3 text-sm transition-colors",
                highlighted === source.index
                  ? "border-brand-500/60 bg-brand-50/60 dark:bg-brand-500/10"
                  : "border-slate-200 bg-slate-50/60 dark:border-slate-800 dark:bg-slate-900/60",
              )}
            >
              <div className="flex items-center gap-2 font-medium">
                <span className="inline-flex size-5 shrink-0 items-center justify-center rounded-md bg-brand-600 text-[11px] font-semibold text-white">
                  {source.index}
                </span>
                <FileText className="size-4 shrink-0 text-slate-400" aria-hidden />
                <span className="truncate">{source.source}</span>
                {source.page !== null && (
                  <span className="shrink-0 text-xs font-normal text-slate-500 dark:text-slate-400">
                    page {source.page}
                  </span>
                )}
              </div>
              <p className="mt-1.5 line-clamp-3 text-slate-600 dark:text-slate-400">
                {source.snippet}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
