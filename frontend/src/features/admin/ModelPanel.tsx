import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Check, Cpu, Loader2, ShieldCheck } from "lucide-react";

import { Alert, Card, Spinner } from "../../components/ui";
import { ApiError, getModels, selectModel } from "../../lib/api";
import type { ModelOption, Models } from "../../lib/types";
import { MODELS_KEY } from "./session";

/** Poll while the local model loads, so its status updates by itself. */
const LOADING_POLL_MS = 2000;

function LocalStatus({ option }: { option: ModelOption }) {
  switch (option.status) {
    case "starting":
      return (
        <span className="mt-1 flex items-center gap-1.5 text-xs text-sky-700 dark:text-sky-300">
          <Loader2 className="size-3.5 animate-spin" aria-hidden />
          Loading the model… this can take a minute.
        </span>
      );
    case "ready":
      return <span className="mt-1 block text-xs text-emerald-700 dark:text-emerald-400">Running</span>;
    case "error":
      return (
        <span className="mt-1 block text-xs text-red-600 dark:text-red-400">
          {option.detail || "The local model could not be started."} Click to try again.
        </span>
      );
    default:
      return (
        <span className="mt-1 block text-xs text-slate-500 dark:text-slate-400">
          Starts when selected.
        </span>
      );
  }
}

interface OptionButtonProps {
  option: ModelOption;
  selected: boolean;
  saving: boolean;
  disabled: boolean;
  onSelect: () => void;
}

function OptionButton({ option, selected, saving, disabled, onSelect }: OptionButtonProps) {
  const local = option.kind === "local";
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      disabled={disabled}
      onClick={onSelect}
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
          selected ? "border-brand-600 bg-brand-600 text-white" : "border-slate-300 dark:border-slate-600",
        )}
        aria-hidden
      >
        {selected && <Check className="size-3.5" />}
      </span>
      <span className="min-w-0">
        <span className="flex flex-wrap items-center gap-2 font-medium">
          {option.label}
          {local && (
            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-700 ring-1 ring-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 dark:ring-emerald-900/60">
              <ShieldCheck className="size-3" aria-hidden />
              Private
            </span>
          )}
          {saving && <span className="text-xs font-normal text-slate-500">Switching…</span>}
        </span>
        {option.description && (
          <span className="mt-0.5 block text-sm text-slate-500 dark:text-slate-400">
            {option.description}
          </span>
        )}
        <code className="mt-1 block text-xs text-slate-400">{option.id}</code>
        {local && <LocalStatus option={option} />}
      </span>
    </button>
  );
}

function SwitchResult({ current }: { current: ModelOption | undefined }) {
  if (!current) return null;
  if (current.status === "starting") {
    return (
      <Alert tone="info">
        Loading {current.label}. Questions will be answered as soon as it&apos;s ready.
      </Alert>
    );
  }
  if (current.status === "error") return null; // the option shows why
  return <Alert tone="success">Now answering with {current.label}.</Alert>;
}

export function ModelPanel() {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: MODELS_KEY,
    queryFn: getModels,
    refetchInterval: (q) =>
      q.state.data?.options.some((o) => o.status === "starting") ? LOADING_POLL_MS : false,
  });

  const mutation = useMutation({
    mutationFn: selectModel,
    onSuccess: (result: Models) => {
      queryClient.setQueryData(MODELS_KEY, result);
      // The header shows the current model.
      void queryClient.invalidateQueries({ queryKey: ["health"] });
    },
  });

  const saving = mutation.isPending ? mutation.variables : null;
  const options = query.data?.options ?? [];
  const currentOption = options.find((o) => o.id === query.data?.current);
  const groups = [
    { title: "Cloud (Groq)", options: options.filter((o) => o.kind !== "local") },
    { title: "On this server", options: options.filter((o) => o.kind === "local") },
  ].filter((g) => g.options.length > 0);
  const hasLocal = groups.length > 1;

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
            <div role="radiogroup" aria-label="Chat model" className="grid gap-4">
              {groups.map((group) => (
                <div key={group.title} className="grid gap-2">
                  {hasLocal && (
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      {group.title}
                    </h3>
                  )}
                  {group.options.map((option) => {
                    const selected = option.id === query.data.current;
                    return (
                      <OptionButton
                        key={option.id}
                        option={option}
                        selected={selected}
                        saving={saving === option.id}
                        disabled={mutation.isPending}
                        // Re-selecting a local model in error retries starting it.
                        onSelect={() =>
                          (!selected || option.status === "error") && mutation.mutate(option.id)
                        }
                      />
                    );
                  })}
                </div>
              ))}
            </div>
          )}
        </div>

        {mutation.isSuccess && (
          <div className="mt-4">
            <SwitchResult current={currentOption} />
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
          <li>Cloud models run on Groq&apos;s free tier, which limits requests per minute and per day.</li>
          <li>Larger models give better answers; smaller ones answer faster.</li>
          <li>Document search runs locally and is the same for every model.</li>
          {hasLocal && (
            <li>
              The local model runs on this server, so nothing leaves the machine. It uses memory
              only while selected and is stopped when you switch to another model.
            </li>
          )}
          <li>
            The list comes from <code>GROQ_MODELS</code>
            {hasLocal && (
              <>
                {" "}and <code>LOCAL_LLM_MODEL</code>
              </>
            )}{" "}
            in the server&apos;s <code>.env</code>.
          </li>
        </ul>
      </Card>
    </div>
  );
}
