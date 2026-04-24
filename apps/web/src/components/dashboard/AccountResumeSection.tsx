"use client";

import Link from "next/link";
import { useState } from "react";
import { ApiError } from "@/lib/api";
import { formatQueryError } from "@/lib/query-error";
import { LoadingRow } from "@/components/ui/loading";
import {
  useUserResume,
  useUploadUserResumeMutation,
  useDeleteUserResumeMutation,
} from "@/hooks/use-user-resume";

export function AccountResumeSection() {
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { data, isPending: loading, isError, error: queryError } = useUserResume();
  const hasResume = data?.has_resume ?? false;
  const filename = data?.filename ?? null;

  const uploadMutation = useUploadUserResumeMutation();
  const deleteMutation = useDeleteUserResumeMutation();

  const displayError =
    error ?? (isError && queryError ? formatQueryError(queryError) : null);

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
        "Remove your profile resume? You can upload again anytime. Ask (on job pages), Analyze fit, and interviews will stop using it until you do."
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
    <section
      className="relative overflow-hidden rounded-2xl border border-orange-400/35 bg-gradient-to-br from-orange-50/90 via-white to-amber-50/50 p-5 shadow-[0_1px_0_rgba(255,255,255,0.8)_inset] ring-1 ring-orange-200/40 sm:p-6"
      aria-labelledby="profile-resume-heading"
    >
      <div className="pointer-events-none absolute -right-8 -top-8 h-32 w-32 rounded-full bg-orange-200/30 blur-2xl" aria-hidden />
      <div className="relative flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-orange-500 text-white shadow-sm"
              aria-hidden
            >
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </span>
            <div>
              <h2
                id="profile-resume-heading"
                className="text-base font-semibold tracking-tight text-zenodrift-text-strong"
              >
                Your profile resume
              </h2>
              <p className="text-xs text-zenodrift-text-muted">
                One PDF for your account. We use it for every job you open.
              </p>
            </div>
          </div>
          <p className="max-w-xl text-sm leading-relaxed text-zenodrift-text">
            We use it for <strong className="font-semibold text-zenodrift-text-strong">Analyze fit</strong>,{" "}
            <strong className="font-semibold text-zenodrift-text-strong">Ask</strong> when you&apos;re on a job
            page, and <strong className="font-semibold text-zenodrift-text-strong">mock interviews</strong>.
            Upload a new file anytime; it simply replaces the old one.
          </p>
          {hasResume && (
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <span className="inline-flex items-center rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-semibold text-emerald-800">
                Active
              </span>
              <span className="truncate text-sm font-medium text-zenodrift-text-strong" title={filename ?? undefined}>
                {filename ?? "Resume on file"}
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="relative mt-4 rounded-xl border border-orange-200/60 bg-white/50 p-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-zenodrift-text-muted">
          Make your resume stronger
        </h3>
        <p className="mt-1 text-xs text-zenodrift-text-muted">
          Chat about wording, impact, layout, or what&apos;s missing. Everything here is based on your
          profile resume.
        </p>
        {hasResume ? (
          <Link
            href="/resume/coach"
            className="mt-3 inline-flex items-center justify-center rounded-xl bg-neutral-900 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-neutral-800 focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2"
          >
            Open resume coach
          </Link>
        ) : (
          <p className="mt-3 text-sm text-zenodrift-text-muted">
            Upload a resume below and this unlocks.
          </p>
        )}
      </div>

      <div className="relative mt-5 space-y-3 border-t border-orange-200/50 pt-5">
        {displayError && (
          <p className="text-sm text-red-600" role="alert">
            {displayError}
          </p>
        )}
        {loading ? (
          <LoadingRow message="Checking your resume…" />
        ) : (
          <>
            <p className="text-xs leading-relaxed text-zenodrift-text-muted">
              PDF only, usually one or two pages (five max). We may ask you to re-upload if the file
              doesn&apos;t look like your own resume.
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <input
                type="file"
                accept="application/pdf"
                onChange={(e) => setResumeFile(e.target.files?.[0] ?? null)}
                className="hidden"
                id="account-resume-file"
              />
              <label
                htmlFor="account-resume-file"
                className="cursor-pointer rounded-xl border border-orange-300/80 bg-white px-4 py-2.5 text-sm font-medium text-zenodrift-text-strong shadow-sm transition-colors hover:border-orange-400 hover:bg-orange-50/80"
              >
                {hasResume ? "Replace PDF" : "Upload PDF"}
              </label>
              {resumeFile && (
                <button
                  type="button"
                  onClick={handleAddResumeFile}
                  disabled={uploadMutation.isPending}
                  className="rounded-xl bg-orange-500 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-orange-600 disabled:opacity-50"
                >
                  {uploadMutation.isPending ? "Saving…" : "Save"}
                </button>
              )}
              {resumeFile && (
                <span className="max-w-[200px] truncate text-xs text-zenodrift-text-muted" title={resumeFile.name}>
                  {resumeFile.name}
                </span>
              )}
            </div>
            {hasResume && (
              <button
                type="button"
                onClick={handleDelete}
                disabled={deleteMutation.isPending}
                className="text-sm font-medium text-red-600 hover:text-red-700 disabled:opacity-50"
              >
                {deleteMutation.isPending ? "Removing…" : "Remove profile resume"}
              </button>
            )}
          </>
        )}
      </div>
    </section>
  );
}
