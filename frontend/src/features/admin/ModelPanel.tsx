import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Check, Cpu } from "lucide-react";

import { Alert, Card, Spinner } from "../../components/ui";
import { ApiError, getModels, selectModel } from "../../lib/api";
import type { Models } from "../../lib/types";
import { MODELS_KEY } from "./session";

export function ModelPanel() {
  const queryClient = useQueryClient();
  const query = useQuery({ queryKey: MODELS_KEY, queryFn: getModels });

  const mutation = useMutation({
    mutationFn: selectModel,
    onSuccess: (result: Models) => {
      queryClient.setQueryData(MODELS_KEY, result);
      // The header shows the current model.
      void queryClient.invalidateQueries({ queryKey: ["health"] });
    },
  });

  const saving = mutation.isPending ? mutation.variables : null;
  const current = query.data?.current;
  const currentLabel = query.data?.options.find((o) => o.id === current)?.label ?? current;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
      <Card className="p-5">
        <h2 className="font-semibold">Chat model</h2>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Choose the model that writes the answers. The change applies to the next question.
        </p>

        <div className="mt-4">
          {query.isPending ? (
            <Spinner label="Loading models…" />
          ) : query.error ? (
            <Alert tone="error">
              {query.error instanceof ApiError ? query.error.message : "Could not load models."}
            </Alert>
          ) : (
            <div role="radiogroup" aria-label="Chat model" className="grid gap-2">
              {query.data.options.map((option) => {
                const selected = option.id === current;
                return (
                  <button
                    key={option.id}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    disabled={mutation.isPending}
                    onClick={() => !selected && mutation.mutate(option.id)}
                    className={clsx(
                      "flex items-start gap-3 rounded-xl border p-4 text-left transition-colors disabled:cursor-wait",
                      selected
                        ? "border-brand-500 bg-brand-50/60 ring-1 ring-brand-500 dark:bg-brand-500/10"
                        : "border-slate-200 hover:border-brand-500/50 hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-800/50",
                    )}
                  >
                    <span
                      className={clsx(
                        "mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border",
                        selected
                          ? "border-brand-600 bg-brand-600 text-white"
                          : "border-slate-300 dark:border-slate-600",
                      )}
                      aria-hidden
                    >
                      {selected && <Check className="size-3.5" />}
                    </span>
                    <span className="min-w-0">
                      <span className="flex flex-wrap items-center gap-2 font-medium">
                        {option.label}
                        {saving === option.id && (
                          <span className="text-xs font-normal text-slate-500">Switching…</span>
                        )}
                      </span>
                      {option.description && (
                        <span className="mt-0.5 block text-sm text-slate-500 dark:text-slate-400">
                          {option.description}
                        </span>
                      )}
                      <code className="mt-1 block text-xs text-slate-400">{option.id}</code>
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {mutation.isSuccess && (
          <div className="mt-4">
            <Alert tone="success">Now answering with {currentLabel}.</Alert>
          </div>
        )}
        {mutation.error && (
          <div className="mt-4">
            <Alert tone="error">
              {mutation.error instanceof ApiError ? mutation.error.message : "Could not change the model."}
            </Alert>
          </div>
        )}
      </Card>

      <Card className="h-fit p-5">
        <div className="flex items-center gap-2">
          <Cpu className="size-5 text-brand-600" aria-hidden />
          <h2 className="font-semibold">About the models</h2>
        </div>
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-slate-600 dark:text-slate-300">
          <li>Models run on Groq&apos;s free tier, which limits requests per minute and per day.</li>
          <li>Larger models give better answers; smaller ones answer faster.</li>
          <li>Document search runs locally and is the same for every model.</li>
          <li>
            The list comes from <code>GROQ_MODELS</code> in the server&apos;s <code>.env</code>.
          </li>
        </ul>
      </Card>
    </div>
  );
}
