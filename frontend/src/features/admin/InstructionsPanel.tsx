import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ShieldCheck } from "lucide-react";
import { useState } from "react";

import { Alert, Button, Card, Spinner } from "../../components/ui";
import { ApiError, getInstructions, saveInstructions } from "../../lib/api";
import type { Instructions } from "../../lib/types";
import { INSTRUCTIONS_KEY } from "./session";

export const MAX_INSTRUCTIONS_LENGTH = 4000;

const PLACEHOLDER = `Examples:
- Use a friendly, professional tone and keep answers short.
- Answer in bullet points when listing steps.
- For payroll questions not covered by the documents, suggest payroll@innovatech.example.`;

function Editor({ saved }: { saved: Instructions }) {
  const queryClient = useQueryClient();
  const [text, setText] = useState(saved.text);
  const dirty = text.trim() !== saved.text;

  const mutation = useMutation({
    mutationFn: saveInstructions,
    onSuccess: (result) => {
      queryClient.setQueryData(INSTRUCTIONS_KEY, result);
      setText(result.text);
    },
  });

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        mutation.mutate(text);
      }}
    >
      <label htmlFor="instructions" className="sr-only">
        Agent instructions
      </label>
      <textarea
        id="instructions"
        value={text}
        maxLength={MAX_INSTRUCTIONS_LENGTH}
        onChange={(event) => {
          setText(event.target.value);
          mutation.reset();
        }}
        placeholder={PLACEHOLDER}
        rows={12}
        className="block w-full resize-y rounded-xl border border-slate-300 bg-white p-3 font-mono text-sm leading-relaxed focus:border-brand-500 focus:outline-none focus:ring-4 focus:ring-brand-500/10 dark:border-slate-700 dark:bg-slate-950"
      />
      <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
        <span>
          {text.length} / {MAX_INSTRUCTIONS_LENGTH}
        </span>
        <span>
          Last saved:{" "}
          {saved.updated_at
            ? new Date(saved.updated_at).toLocaleString(undefined, {
                dateStyle: "medium",
                timeStyle: "short",
              })
            : "never"}
        </span>
        {dirty && <span className="font-medium text-amber-600">Unsaved changes</span>}
      </div>

      {mutation.isSuccess && !dirty && (
        <div className="mt-3">
          <Alert tone="success">Saved. The next answer will follow these instructions.</Alert>
        </div>
      )}
      {mutation.error && (
        <div className="mt-3">
          <Alert tone="error">
            {mutation.error instanceof ApiError ? mutation.error.message : "Could not save."}
          </Alert>
        </div>
      )}

      <div className="mt-4 flex justify-end gap-2">
        <Button variant="secondary" disabled={!dirty || mutation.isPending} onClick={() => setText(saved.text)}>
          Discard
        </Button>
        <Button type="submit" loading={mutation.isPending} disabled={!dirty}>
          Save instructions
        </Button>
      </div>
    </form>
  );
}

export function InstructionsPanel() {
  const query = useQuery({ queryKey: INSTRUCTIONS_KEY, queryFn: getInstructions });

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
      <Card className="p-5">
        <h2 className="font-semibold">Agent instructions</h2>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Tell the assistant how to answer: tone, format, and who employees should contact.
        </p>
        <div className="mt-4">
          {query.isPending ? (
            <Spinner label="Loading…" />
          ) : query.error ? (
            <Alert tone="error">
              {query.error instanceof ApiError ? query.error.message : "Could not load instructions."}
            </Alert>
          ) : (
            <Editor saved={query.data} />
          )}
        </div>
      </Card>

      <Card className="h-fit p-5">
        <div className="flex items-center gap-2">
          <ShieldCheck className="size-5 text-emerald-600" aria-hidden />
          <h2 className="font-semibold">Always enforced</h2>
        </div>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          These rules come first. Your instructions can&apos;t change them:
        </p>
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-slate-600 dark:text-slate-300">
          <li>Answers use only the uploaded company documents.</li>
          <li>Every answer cites its sources.</li>
          <li>If the documents don&apos;t cover a question, the assistant says so.</li>
          <li>Instructions hidden inside documents are ignored.</li>
        </ul>
      </Card>
    </div>
  );
}
