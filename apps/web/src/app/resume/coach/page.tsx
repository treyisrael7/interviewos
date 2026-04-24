"use client";

import { useState } from "react";
import Link from "next/link";
import { useMutation } from "@tanstack/react-query";
import { GradientShell } from "@/components/GradientShell";
import { ResumeCoachAnswerDisplay } from "@/components/ResumeCoachAnswerDisplay";
import {
  ApiError,
  askProfileResumeCoach,
  parseResumeCoachAnswer,
  type AskResponse,
} from "@/lib/api";
import { LoadingRow } from "@/components/ui/loading";
import { useUserResume } from "@/hooks/use-user-resume";

export default function ResumeCoachPage() {
  const [question, setQuestion] = useState("");
  const [answerData, setAnswerData] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { data: resume, isPending: resumeLoading } = useUserResume();
  const hasResume = Boolean(resume?.has_resume);

  const mutation = useMutation({
    mutationFn: (q: string) => askProfileResumeCoach(q),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || mutation.isPending) return;
    setError(null);
    setAnswerData(null);
    mutation.mutate(q, {
      onSuccess: (res) => setAnswerData(res),
      onError: (err) => {
        setError(
          err instanceof ApiError
            ? String(err.detail || err.message)
            : "Request failed"
        );
      },
    });
  };

  const structured = answerData ? parseResumeCoachAnswer(answerData.answer) : null;

  return (
    <GradientShell>
      <div className="mx-auto w-full max-w-3xl px-4 py-10 sm:py-14">
        <nav className="mb-8">
          <Link
            href="/dashboard"
            className="text-sm font-medium text-zenodrift-accent hover:text-zenodrift-accent-hover"
          >
            ← Dashboard
          </Link>
        </nav>

        <h1 className="text-2xl font-bold tracking-tight text-zenodrift-text-strong sm:text-3xl">
          Improve your resume
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-zenodrift-text-muted">
          Ask us anything about your profile resume: stronger bullets, a clearer summary, what to cut,
          how to phrase a win, and whatever else would help you feel more confident sending it out.
        </p>

        {resumeLoading ? (
          <div className="mt-8">
            <LoadingRow message="Checking your resume…" />
          </div>
        ) : !hasResume ? (
          <div
            className="mt-8 rounded-2xl border border-amber-200/80 bg-amber-50/80 px-5 py-4 text-sm text-amber-950"
            role="status"
          >
            <p className="font-medium">No profile resume yet</p>
            <p className="mt-1 text-amber-900/90">
              Pop over to the dashboard and upload a PDF, then come back. We&apos;ll be here.
            </p>
            <Link
              href="/dashboard"
              className="mt-3 inline-block text-sm font-semibold text-zenodrift-accent hover:text-zenodrift-accent-hover"
            >
              Go to dashboard
            </Link>
          </div>
        ) : (
          <>
            <form onSubmit={handleSubmit} className="mt-8 space-y-4">
              <label htmlFor="resume-coach-q" className="sr-only">
                Your question
              </label>
              <textarea
                id="resume-coach-q"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                rows={3}
                disabled={mutation.isPending}
                placeholder="e.g. How can I make my impact bullets stronger? Is my education section too long?"
                className="w-full rounded-xl border border-slate-200 bg-white/90 px-4 py-3 text-zenodrift-text-strong shadow-sm ring-1 ring-slate-200/60 placeholder-zenodrift-text-muted focus:border-orange-400 focus:outline-none focus:ring-2 focus:ring-orange-400/20 disabled:opacity-70"
              />
              <button
                type="submit"
                disabled={mutation.isPending || !question.trim()}
                className="rounded-xl bg-neutral-900 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-neutral-800 focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2 disabled:opacity-50"
              >
                {mutation.isPending ? "Thinking…" : "Ask"}
              </button>
            </form>

            {error && (
              <div
                className="mt-6 rounded-xl bg-red-50/80 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200/50"
                role="alert"
              >
                {error}
              </div>
            )}

            {answerData && (
              <section className="mt-10 border-t border-slate-200/80 pt-10">
                <h2 className="mb-4 text-xs font-semibold uppercase tracking-wider text-zenodrift-text-muted">
                  Answer
                </h2>
                {structured ? (
                  <ResumeCoachAnswerDisplay data={structured} />
                ) : (
                  <div className="prose prose-slate max-w-none whitespace-pre-wrap text-sm text-zenodrift-text">
                    {answerData.answer}
                  </div>
                )}
                {answerData.citations.length > 0 && (
                  <div className="mt-8 border-t border-slate-200/80 pt-6">
                    <h3 className="mb-2 text-sm font-semibold text-zenodrift-text">
                      Resume snippets we used
                    </h3>
                    <ul className="space-y-2">
                      {answerData.citations.map((c) => (
                        <li
                          key={c.chunk_id}
                          className="rounded-lg bg-slate-50/80 px-3 py-2 text-sm text-zenodrift-text"
                        >
                          <span className="font-mono text-xs text-orange-600">[{c.chunk_id}]</span>{" "}
                          Page {c.page_number}: {c.snippet}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </section>
            )}
          </>
        )}
      </div>
    </GradientShell>
  );
}
