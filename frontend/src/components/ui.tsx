import clsx from "clsx";
import { AlertCircle, CheckCircle2, Info, Loader2 } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-brand-600 text-white hover:bg-brand-700 disabled:bg-brand-600/50 shadow-sm",
  secondary:
    "bg-white text-slate-700 ring-1 ring-slate-200 hover:bg-slate-50 dark:bg-slate-900 " +
    "dark:text-slate-200 dark:ring-slate-700 dark:hover:bg-slate-800",
  ghost:
    "text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 " +
    "dark:hover:bg-slate-800 dark:hover:text-slate-100",
  danger: "bg-red-600 text-white hover:bg-red-700 disabled:bg-red-600/50 shadow-sm",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: "sm" | "md";
  loading?: boolean;
  icon?: ReactNode;
}

export function Button({
  variant = "primary",
  size = "md",
  loading = false,
  icon,
  className,
  children,
  disabled,
  type = "button",
  ...props
}: ButtonProps) {
  return (
    <button
      type={type}
      disabled={disabled || loading}
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-60",
        size === "sm" ? "h-8 px-3 text-sm" : "h-10 px-4 text-sm",
        VARIANTS[variant],
        className,
      )}
      {...props}
    >
      {loading ? <Loader2 className="size-4 animate-spin" aria-hidden /> : icon}
      {children}
    </button>
  );
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div
      className={clsx(
        "rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900",
        className,
      )}
    >
      {children}
    </div>
  );
}

type Tone = "error" | "success" | "info";

const TONES: Record<Tone, { box: string; icon: ReactNode }> = {
  error: {
    box: "border-red-200 bg-red-50 text-red-800 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-200",
    icon: <AlertCircle className="mt-0.5 size-4 shrink-0" aria-hidden />,
  },
  success: {
    box: "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-200",
    icon: <CheckCircle2 className="mt-0.5 size-4 shrink-0" aria-hidden />,
  },
  info: {
    box: "border-sky-200 bg-sky-50 text-sky-800 dark:border-sky-900/60 dark:bg-sky-950/40 dark:text-sky-200",
    icon: <Info className="mt-0.5 size-4 shrink-0" aria-hidden />,
  },
};

export function Alert({ tone, children }: { tone: Tone; children: ReactNode }) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={clsx("flex gap-2 rounded-lg border px-3 py-2 text-sm", TONES[tone].box)}
    >
      {TONES[tone].icon}
      <div className="min-w-0">{children}</div>
    </div>
  );
}

export function Spinner({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
      <Loader2 className="size-4 animate-spin" aria-hidden />
      {label}
    </div>
  );
}
