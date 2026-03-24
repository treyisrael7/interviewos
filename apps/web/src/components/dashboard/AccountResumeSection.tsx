"use client";

import { useState } from "react";
import { ApiError } from "@/lib/api";
import { formatQueryError } from "@/lib/query-error";
import {
  useUserResume,
  useUploadUserResumeMutation,
  useDeleteUserResumeMutation,
} from "@/hooks/use-user-resume";

export function AccountResumeSection() {
  const [expanded, setExpanded] = useState(false);
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { data, isPending: loading, isError, error: queryError } = useUserResume();
  const hasResume = data?.has_resume ?? false;
  const filename = data?.filename ?? null;

  const uploadMutation = useUploadUserResumeMutation();
  const deleteMutation = useDeleteUserResumeMutation();

  const displayError =
    error ??
    (isError && queryError ? formatQueryError(queryError) : null);

  const handleAddResumeFile = () => {
    const file = resumeFile;
    if (!file || file.type !== "application/pdf") return;
    setError(null);
    uploadMutation.mutate(file, {
      onSuccess: () => {
        setResumeFile(null);
      },
      onError: (e) => {
        setError(
          e instanceof ApiError
            ? String(e.detail || e.message)
            : "Failed to upload resume"
        );
      },
    });
  };

  const handleDelete = () => {
    if (
      !confirm(
        "Remove your account resume? It will no longer be used for interview feedback."
      )
    )
      return;
    setError(null);
    deleteMutation.mutate(undefined, {
      onError: (e) => {
        setError(
          e instanceof ApiError
            ? String(e.detail || e.message)
            : "Failed to delete resume"
        );
      },
    });
  };

  return (
    <div className="rounded-lg border border-white/20 bg-white/10">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between px-3 py-2 text-left text-xs font-medium text-zenodrift-text transition-colors hover:bg-white/15"
        aria-expanded={expanded}
      >
        <span>Account Resume (applies to all job descriptions)</span>
        {hasResume && (
          <span className="rounded-full bg-white/30 px-2 py-0.5 text-xs text-zenodrift-text-muted">
            {filename ?? "Added"}
          </span>
        )}
        <svg
          className={`h-3.5 w-3.5 transition-transform ${expanded ? "rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {expanded && (
        <div className="space-y-2 border-t border-white/20 px-3 py-3">
          {displayError && (
            <p className="text-xs text-red-600" role="alert">
              {displayError}
            </p>
          )}
          {loading ? (
            <p className="text-xs text-zenodrift-text-muted">Loading…</p>
          ) : (
            <>
              <div className="flex gap-2">
                <input
                  type="file"
                  accept="application/pdf"
                  onChange={(e) => setResumeFile(e.target.files?.[0] ?? null)}
                  className="hidden"
                  id="account-resume-file"
                />
                <label
                  htmlFor="account-resume-file"
                  className="cursor-pointer rounded-lg border border-white/40 bg-white/60 px-2.5 py-1.5 text-xs text-zenodrift-text hover:bg-white/80"
                >
                  Upload PDF
                </label>
                {resumeFile && (
                  <button
                    onClick={handleAddResumeFile}
                    disabled={uploadMutation.isPending}
                    className="rounded-lg bg-white/60 px-2.5 py-1.5 text-xs font-medium text-zenodrift-accent hover:bg-white/80 disabled:opacity-50"
                  >
                    {uploadMutation.isPending ? "Uploading…" : "Add"}
                  </button>
                )}
              </div>
              {hasResume && (
                <button
                  onClick={handleDelete}
                  disabled={deleteMutation.isPending}
                  className="text-xs font-medium text-red-600 hover:text-red-700 disabled:opacity-50"
                >
                  {deleteMutation.isPending ? "Removing…" : "Remove resume"}
                </button>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
