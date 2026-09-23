import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import {
  CheckCircle2,
  CircleAlert,
  CopyCheck,
  FileText,
  Loader2,
  Search,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { useRef, useState, type DragEvent } from "react";

import { Alert, Button, Card, Spinner } from "../../components/ui";
import { ApiError, deleteDocument, listDocuments, uploadDocument } from "../../lib/api";
import type { StoredDocument } from "../../lib/types";
import { DOCUMENTS_KEY } from "./session";

const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"];

type UploadStatus = "waiting" | "uploading" | "ingested" | "duplicate" | "error";

interface UploadItem {
  id: string;
  name: string;
  status: UploadStatus;
  message?: string;
}

const hasAcceptedType = (name: string) =>
  ACCEPTED_EXTENSIONS.some((ext) => name.toLowerCase().endsWith(ext));

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function UploadRow({ item }: { item: UploadItem }) {
  const icon = {
    waiting: <Loader2 className="size-4 text-slate-300" />,
    uploading: <Loader2 className="size-4 animate-spin text-brand-600" />,
    ingested: <CheckCircle2 className="size-4 text-emerald-600" />,
    duplicate: <CopyCheck className="size-4 text-sky-600" />,
    error: <CircleAlert className="size-4 text-red-600" />,
  }[item.status];

  return (
    <li className="flex items-start gap-2 py-2 text-sm">
      <span className="mt-0.5" aria-hidden>
        {icon}
      </span>
      <div className="min-w-0">
        <p className="truncate font-medium">{item.name}</p>
        <p
          className={clsx(
            "text-xs",
            item.status === "error" ? "text-red-600 dark:text-red-400" : "text-slate-500 dark:text-slate-400",
          )}
        >
          {item.status === "waiting" && "Waiting…"}
          {item.status === "uploading" && "Reading and indexing…"}
          {item.message}
        </p>
      </div>
    </li>
  );
}

function Dropzone({ onFiles, busy }: { onFiles: (files: File[]) => void; busy: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    onFiles(Array.from(event.dataTransfer.files));
  };

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={clsx(
        "flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-8 text-center transition-colors",
        dragging
          ? "border-brand-500 bg-brand-50/60 dark:bg-brand-500/10"
          : "border-slate-300 dark:border-slate-700",
      )}
    >
      <UploadCloud className="size-8 text-slate-400" aria-hidden />
      <p className="mt-2 text-sm font-medium">Drag files here, or</p>
      <Button
        variant="secondary"
        size="sm"
        className="mt-2"
        onClick={() => input.current?.click()}
        disabled={busy}
      >
        Choose files
      </Button>
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">PDF, Word, text or Markdown</p>
      <input
        ref={input}
        type="file"
        multiple
        accept={ACCEPTED_EXTENSIONS.join(",")}
        className="hidden"
        data-testid="file-input"
        onChange={(event) => {
          onFiles(Array.from(event.target.files ?? []));
          event.target.value = "";
        }}
      />
    </div>
  );
}

function DocumentRow({ document }: { document: StoredDocument }) {
  const queryClient = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const mutation = useMutation({
    mutationFn: () => deleteDocument(document.doc_id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: DOCUMENTS_KEY }),
  });

  return (
    <li className="flex flex-wrap items-center gap-3 px-4 py-3">
      <FileText className="size-5 shrink-0 text-slate-400" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium">{document.source}</p>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          {document.chunk_count} chunk{document.chunk_count === 1 ? "" : "s"} · added{" "}
          {formatDate(document.ingested_at)}
        </p>
        {mutation.error instanceof ApiError && (
          <p className="text-xs text-red-600">{mutation.error.message}</p>
        )}
      </div>
      {confirming ? (
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">Delete this document?</span>
          <Button variant="ghost" size="sm" onClick={() => setConfirming(false)}>
            Cancel
          </Button>
          <Button variant="danger" size="sm" loading={mutation.isPending} onClick={() => mutation.mutate()}>
            Delete
          </Button>
        </div>
      ) : (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setConfirming(true)}
          aria-label={`Delete ${document.source}`}
          icon={<Trash2 className="size-4" />}
        />
      )}
    </li>
  );
}

export function DocumentsPanel() {
  const queryClient = useQueryClient();
  const documents = useQuery({ queryKey: DOCUMENTS_KEY, queryFn: listDocuments });
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [filter, setFilter] = useState("");
  const busy = uploads.some((u) => u.status === "waiting" || u.status === "uploading");

  const setItem = (id: string, change: Partial<UploadItem>) =>
    setUploads((all) => all.map((u) => (u.id === id ? { ...u, ...change } : u)));

  const upload = async (files: File[]) => {
    if (files.length === 0 || busy) return;
    const items = files.map((file, i) => ({
      file,
      item: { id: `${Date.now()}-${i}`, name: file.name, status: "waiting" as UploadStatus },
    }));
    setUploads(items.map((i) => i.item));

    // One at a time: indexing calls the embedding API, which is rate limited.
    for (const { file, item } of items) {
      if (!hasAcceptedType(file.name)) {
        setItem(item.id, {
          status: "error",
          message: `Not a supported file type. Use ${ACCEPTED_EXTENSIONS.join(", ")}.`,
        });
        continue;
      }
      setItem(item.id, { status: "uploading" });
      try {
        const result = await uploadDocument(file);
        setItem(item.id, { status: result.status, message: result.message });
        await queryClient.invalidateQueries({ queryKey: DOCUMENTS_KEY });
      } catch (error) {
        const message = error instanceof ApiError ? error.message : "Upload failed.";
        setItem(item.id, { status: "error", message });
        if (error instanceof ApiError && error.status === 401) break;
      }
    }
  };

  const visible = (documents.data ?? []).filter((d) =>
    d.source.toLowerCase().includes(filter.trim().toLowerCase()),
  );

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
      <Card className="h-fit p-5">
        <h2 className="font-semibold">Upload documents</h2>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          New documents can be searched as soon as they are indexed. Re-uploading the same file
          is skipped.
        </p>
        <div className="mt-4">
          <Dropzone onFiles={upload} busy={busy} />
        </div>
        {uploads.length > 0 && (
          <ul className="mt-3 divide-y divide-slate-100 dark:divide-slate-800" aria-label="Uploads">
            {uploads.map((item) => (
              <UploadRow key={item.id} item={item} />
            ))}
          </ul>
        )}
      </Card>

      <Card className="overflow-hidden">
        <div className="flex flex-wrap items-center gap-3 border-b border-slate-200 px-4 py-3 dark:border-slate-800">
          <div>
            <h2 className="font-semibold">Knowledge base</h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {documents.data ? `${documents.data.length} document(s)` : " "}
            </p>
          </div>
          {(documents.data?.length ?? 0) > 5 && (
            <label className="relative ml-auto">
              <span className="sr-only">Filter documents</span>
              <Search className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-slate-400" />
              <input
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
                placeholder="Filter by name"
                className="h-9 w-48 rounded-lg border border-slate-300 bg-white pl-8 pr-3 text-sm focus:border-brand-500 focus:outline-none dark:border-slate-700 dark:bg-slate-950"
              />
            </label>
          )}
        </div>

        {documents.isPending ? (
          <div className="p-4">
            <Spinner label="Loading documents…" />
          </div>
        ) : documents.error ? (
          <div className="p-4">
            <Alert tone="error">
              {documents.error instanceof ApiError ? documents.error.message : "Could not load documents."}
            </Alert>
          </div>
        ) : documents.data.length === 0 ? (
          <p className="px-4 py-10 text-center text-sm text-slate-500 dark:text-slate-400">
            No documents yet. Upload one to get started.
          </p>
        ) : (
          <ul className="divide-y divide-slate-100 dark:divide-slate-800">
            {visible.map((document) => (
              <DocumentRow key={document.doc_id} document={document} />
            ))}
            {visible.length === 0 && (
              <li className="px-4 py-6 text-center text-sm text-slate-500">No matching documents.</li>
            )}
          </ul>
        )}
      </Card>
    </div>
  );
}
