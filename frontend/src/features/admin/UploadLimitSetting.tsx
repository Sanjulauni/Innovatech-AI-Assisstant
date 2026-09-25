import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { Alert, Button } from "../../components/ui";
import { ApiError, getUploadLimit, saveUploadLimit } from "../../lib/api";
import type { UploadLimit } from "../../lib/types";
import { UPLOAD_LIMIT_KEY } from "./session";

interface LimitFormProps {
  limit: UploadLimit;
  onSaved: () => void;
  onCancel: () => void;
}

function LimitForm({ limit, onSaved, onCancel }: LimitFormProps) {
  const queryClient = useQueryClient();
  const [value, setValue] = useState(String(limit.max_upload_size_mb));
  const mb = Number(value);
  const valid = Number.isInteger(mb) && mb >= 1 && mb <= limit.max_allowed_mb;

  const mutation = useMutation({
    mutationFn: saveUploadLimit,
    onSuccess: (result) => {
      queryClient.setQueryData(UPLOAD_LIMIT_KEY, result);
      onSaved();
    },
  });

  return (
    <form
      className="mt-2"
      onSubmit={(event) => {
        event.preventDefault();
        if (valid) mutation.mutate(mb);
      }}
    >
      <label htmlFor="upload-limit" className="text-xs font-medium">
        Max file size (MB)
      </label>
      <div className="mt-1 flex flex-wrap items-center gap-2">
        <input
          id="upload-limit"
          type="number"
          inputMode="numeric"
          min={1}
          max={limit.max_allowed_mb}
          step={1}
          value={value}
          onChange={(event) => {
            setValue(event.target.value);
            mutation.reset();
          }}
          aria-invalid={!valid}
          className="h-8 w-24 rounded-lg border border-slate-300 bg-white px-2 text-sm focus:border-brand-500 focus:outline-none dark:border-slate-700 dark:bg-slate-950"
        />
        <Button type="submit" size="sm" disabled={!valid} loading={mutation.isPending}>
          Save
        </Button>
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
      <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
        Between 1 and {limit.max_allowed_mb} MB. The server default is {limit.default_mb} MB.
      </p>
      {mutation.error && (
        <div className="mt-2">
          <Alert tone="error">
            {mutation.error instanceof ApiError ? mutation.error.message : "Could not save the limit."}
          </Alert>
        </div>
      )}
    </form>
  );
}

/** Shows the largest file the server accepts and lets the admin change it. */
export function UploadLimitSetting() {
  const query = useQuery({ queryKey: UPLOAD_LIMIT_KEY, queryFn: getUploadLimit });
  const [editing, setEditing] = useState(false);
  const [saved, setSaved] = useState(false);

  if (query.isPending) return null;
  if (query.error) {
    return (
      <p className="text-xs text-red-600 dark:text-red-400">
        {query.error instanceof ApiError ? query.error.message : "Could not load the upload limit."}
      </p>
    );
  }

  if (editing) {
    return (
      <LimitForm
        limit={query.data}
        onSaved={() => {
          setEditing(false);
          setSaved(true);
        }}
        onCancel={() => setEditing(false)}
      />
    );
  }

  return (
    <div>
      <div className="flex items-center gap-2 text-sm">
        <span className="text-slate-600 dark:text-slate-300">
          Max file size: <strong>{query.data.max_upload_size_mb} MB</strong>
        </span>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            setSaved(false);
            setEditing(true);
          }}
        >
          Change
        </Button>
      </div>
      {saved && (
        <p role="status" className="text-xs text-emerald-700 dark:text-emerald-400">
          Saved. Applies to the next upload.
        </p>
      )}
    </div>
  );
}
