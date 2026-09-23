import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { MessagesSquare, Moon, Settings2, Sun } from "lucide-react";
import { NavLink, Outlet } from "react-router";

import { getHealth } from "../lib/api";
import { useTheme } from "../lib/theme";

function ServerStatus() {
  const { data, isError, isPending } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: false,
  });

  const ok = data?.status === "ok";
  const label = isPending
    ? "Connecting…"
    : isError
      ? "Server offline"
      : ok
        ? `${data.documents} document${data.documents === 1 ? "" : "s"}`
        : "Degraded";

  return (
    <span
      className="hidden items-center gap-1.5 text-xs text-slate-500 sm:flex dark:text-slate-400"
      title={data ? `Model: ${data.llm_model}` : undefined}
    >
      <span
        className={clsx(
          "size-2 rounded-full",
          isPending ? "bg-slate-300" : isError || !ok ? "bg-red-500" : "bg-emerald-500",
        )}
        aria-hidden
      />
      {label}
    </span>
  );
}

const navClass = ({ isActive }: { isActive: boolean }) =>
  clsx(
    "inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors",
    isActive
      ? "bg-brand-50 text-brand-700 dark:bg-brand-500/15 dark:text-brand-100"
      : "text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100",
  );

export function Layout() {
  const [theme, toggleTheme] = useTheme();

  return (
    <div className="flex h-full flex-col">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-slate-950/80">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-4">
          <NavLink to="/" className="flex items-center gap-2 font-semibold">
            <img src="/favicon.svg" alt="" className="size-7" />
            <span>
              InnovaTech <span className="text-brand-600 dark:text-brand-500">Assistant</span>
            </span>
          </NavLink>

          <nav className="ml-2 flex items-center gap-1" aria-label="Main">
            <NavLink to="/" end className={navClass}>
              <MessagesSquare className="size-4" aria-hidden /> Chat
            </NavLink>
            <NavLink to="/admin" className={navClass}>
              <Settings2 className="size-4" aria-hidden /> Admin
            </NavLink>
          </nav>

          <div className="ml-auto flex items-center gap-3">
            <ServerStatus />
            <button
              type="button"
              onClick={toggleTheme}
              className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100"
              aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            >
              {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </button>
          </div>
        </div>
      </header>

      <main className="flex min-h-0 flex-1 flex-col">
        <Outlet />
      </main>
    </div>
  );
}
