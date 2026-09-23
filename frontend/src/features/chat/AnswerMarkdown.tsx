import type { ComponentProps } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { CITE_PREFIX, linkCitations } from "./citations";

interface Props {
  text: string;
  onCite?: (index: number) => void;
}

/**
 * Renders an answer as Markdown. Raw HTML in the text is not rendered (react-markdown
 * escapes it), so document content can't inject markup into the page.
 */
export function AnswerMarkdown({ text, onCite }: Props) {
  const components: ComponentProps<typeof ReactMarkdown>["components"] = {
    a({ href, children }) {
      if (href?.startsWith(CITE_PREFIX)) {
        const index = Number(href.slice(CITE_PREFIX.length));
        return (
          <button
            type="button"
            onClick={() => onCite?.(index)}
            className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-md bg-brand-50 px-1 align-text-top text-[11px] font-semibold text-brand-700 no-underline ring-1 ring-brand-100 hover:bg-brand-100 dark:bg-brand-500/15 dark:text-brand-100 dark:ring-brand-500/30"
            aria-label={`Show source ${index}`}
          >
            {children}
          </button>
        );
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer">
          {children}
        </a>
      );
    },
  };

  return (
    <div className="prose prose-slate max-w-none text-[15px] leading-relaxed dark:prose-invert prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {linkCitations(text)}
      </ReactMarkdown>
    </div>
  );
}
