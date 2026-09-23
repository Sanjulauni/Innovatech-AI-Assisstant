import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Cpu, FolderOpen, LogOut, MessageSquareText } from "lucide-react";
import { useState, type ComponentType } from "react";

import { Alert, Button, Spinner } from "../../components/ui";
import { ApiError, checkSession, getHealth, logout } from "../../lib/api";
import { DocumentsPanel } from "./DocumentsPanel";
import { InstructionsPanel } from "./InstructionsPanel";
import { LoginForm } from "./LoginForm";
import { ModelPanel } from "./ModelPanel";
import { DOCUMENTS_KEY, INSTRUCTIONS_KEY, MODELS_KEY, SESSION_KEY } from "./session";

type Tab = "documents" | "instructions" | "model";

const TABS: { id: Tab; label: string; icon: typeof FolderOpen }[] = [
  { id: "documents", label: "Documents", icon: FolderOpen },
  { id: "instructions", label: "Instructions", icon: MessageSquareText },
  { id: "model", label: "Model", icon: Cpu },
];

const PANELS: Record<Tab, ComponentType> = {
  documents: DocumentsPanel,
  instructions: InstructionsPanel,
  model: ModelPanel,
};

export function AdminPage() {
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<Tab>("documents");
  const session = useQuery({ queryKey: SESSION_KEY, queryFn: checkSession, retry: false });
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth, retry: false });

  const signOut = useMutation({
    mutationFn: logout,
    onSettled: () => {
      queryClient.setQueryData(SESSION_KEY, false);
      // Drop cached admin data so the next person to sign in doesn't see it first.
      queryClient.removeQueries({ queryKey: DOCUMENTS_KEY });
      queryClient.removeQueries({ queryKey: INSTRUCTIONS_KEY });
      queryClient.removeQueries({ queryKey: MODELS_KEY });
    },
  });

  if (session.isPending) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Spinner label="Checking your session…" />
      </div>
    );
  }

  if (session.error) {
    // 503 = admin disabled on the server; anything else = server unreachable.
    if (session.error instanceof ApiError && session.error.status === 503) {
      return <LoginForm adminEnabled={false} />;
    }
    return (
      <div className="mx-auto w-full max-w-md px-4 py-12">
        <Alert tone="error">
          {session.error instanceof ApiError ? session.error.message : "Could not reach the server."}
        </Alert>
      </div>
    );
  }

  if (!session.data) return <LoginForm adminEnabled={health.data?.admin_enabled} />;

  const Panel = PANELS[tab];

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-6xl px-4 py-8">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Admin</h1>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              Manage the knowledge base and how the assistant answers.
            </p>
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => signOut.mutate()}
            loading={signOut.isPending}
            icon={<LogOut className="size-4" />}
          >
            Sign out
          </Button>
        </div>

        <div
          role="tablist"
          aria-label="Admin sections"
          className="mt-6 flex gap-1 border-b border-slate-200 dark:border-slate-800"
        >
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              role="tab"
              type="button"
              aria-selected={tab === id}
              onClick={() => setTab(id)}
              className={clsx(
                "-mb-px inline-flex items-center gap-2 border-b-2 px-3 py-2 text-sm font-medium transition-colors",
                tab === id
                  ? "border-brand-600 text-brand-700 dark:text-brand-100"
                  : "border-transparent text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200",
              )}
            >
              <Icon className="size-4" aria-hidden />
              {label}
            </button>
          ))}
        </div>

        <div className="mt-6" role="tabpanel">
          <Panel />
        </div>
      </div>
    </div>
  );
}
