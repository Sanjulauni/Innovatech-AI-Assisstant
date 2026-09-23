import { useMutation, useQueryClient } from "@tanstack/react-query";
import { LockKeyhole } from "lucide-react";
import { useState } from "react";

import { Alert, Button, Card } from "../../components/ui";
import { ApiError, login } from "../../lib/api";
import { SESSION_KEY } from "./session";

export function LoginForm({ adminEnabled }: { adminEnabled: boolean | undefined }) {
  const queryClient = useQueryClient();
  const [password, setPassword] = useState("");

  const mutation = useMutation({
    mutationFn: login,
    onSuccess: () => {
      setPassword("");
      queryClient.setQueryData(SESSION_KEY, true);
    },
  });

  const error = mutation.error instanceof ApiError ? mutation.error.message : null;

  return (
    <div className="flex flex-1 items-center justify-center px-4 py-12">
      <Card className="w-full max-w-sm p-6">
        <div className="flex size-11 items-center justify-center rounded-xl bg-brand-50 text-brand-600 ring-1 ring-brand-100 dark:bg-brand-500/15 dark:text-brand-100 dark:ring-brand-500/30">
          <LockKeyhole className="size-5" aria-hidden />
        </div>
        <h1 className="mt-4 text-xl font-semibold">Admin sign in</h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Manage company documents and how the assistant answers.
        </p>

        {adminEnabled === false ? (
          <div className="mt-5">
            <Alert tone="info">
              Admin access is turned off. Set <code>ADMIN_PASSWORD</code> in the server&apos;s{" "}
              <code>.env</code> file and restart the API.
            </Alert>
          </div>
        ) : (
          <form
            className="mt-5 space-y-4"
            onSubmit={(event) => {
              event.preventDefault();
              if (password) mutation.mutate(password);
            }}
          >
            <div>
              <label htmlFor="admin-password" className="text-sm font-medium">
                Password
              </label>
              <input
                id="admin-password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                className="mt-1.5 block h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm focus:border-brand-500 focus:outline-none focus:ring-4 focus:ring-brand-500/10 dark:border-slate-700 dark:bg-slate-950"
                autoFocus
              />
            </div>
            {error && <Alert tone="error">{error}</Alert>}
            <Button type="submit" className="w-full" loading={mutation.isPending} disabled={!password}>
              Sign in
            </Button>
          </form>
        )}
      </Card>
    </div>
  );
}
