"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, type RoleProfile, type CompetencyWithCoverage } from "@/lib/api";
import {
  useGenerateInterviewMutation,
} from "@/hooks/use-interview-setup";

type Difficulty = "junior" | "mid" | "senior";

const DOMAIN_LABELS: Record<string, string> = {
  technical: "Technical",
  finance: "Finance",
  healthcare_social_work: "Healthcare & Social Work",
  sales_marketing: "Sales & Marketing",
  operations: "Operations",
  education: "Education",
  general_business: "General Business",
};

const SENIORITY_LABELS: Record<string, string> = {
  entry: "Entry",
  mid: "Mid",
  senior: "Senior",
};

interface InterviewSetupPanelProps {
  documentId: string;
  roleProfile?: RoleProfile | null;
  competencies?: CompetencyWithCoverage[];
  coveragePracticed?: number;
  coverageTotal?: number;
}

function formatGenerateInterviewError(e: unknown): string {
  let msg = "Failed to start interview";
  if (e instanceof ApiError && e.detail != null) {
    const d = e.detail;
    msg =
      typeof d === "string"
        ? d
        : Array.isArray(d)
          ? d.map((x: { msg?: string }) => x?.msg ?? JSON.stringify(x)).join(", ")
          : typeof d === "object" && d !== null && "detail" in d
            ? String((d as { detail?: unknown }).detail)
            : JSON.stringify(d);
  } else if (e instanceof Error) {
    if (e.message === "Failed to fetch") {
      msg =
        "Could not reach the API. Ensure the API is running (e.g. docker compose up api or cd apps/api && uvicorn app.main:app).";
    } else if (e.name === "AbortError" || e.message.includes("aborted")) {
      msg =
        "Interview generation timed out (took longer than 2 minutes). Try again or use fewer questions.";
    } else {
      msg = e.message;
    }
  }
  return msg;
}

export function InterviewSetupPanel({
  documentId,
  roleProfile,
  competencies = [],
  coveragePracticed = 0,
  coverageTotal = 0,
}: InterviewSetupPanelProps) {
  const router = useRouter();
  const [difficulty, setDifficulty] = useState<Difficulty>("junior");
  const [numQuestions, setNumQuestions] = useState(6);
  const [genError, setGenError] = useState<string | null>(null);
  const generateMutation = useGenerateInterviewMutation(documentId);

  const roleTitle = roleProfile?.roleTitleGuess?.trim() || "Role";
  const detectedDomain = roleProfile?.domain || "general_business";
  const detectedSeniority = roleProfile?.seniority || "entry";
  const focusAreas = roleProfile?.focusAreas ?? [];
  const chips = competencies.length > 0
    ? competencies.slice(0, 8).map((c) => ({ id: c.id, label: c.label, attempts: c.attempts_count, avgScore: c.avg_score }))
    : focusAreas.slice(0, 8).map((a) => ({ id: a, label: a, attempts: 0, avgScore: null as number | null }));

  const handleStartInterview = async () => {
    setGenError(null);
    try {
      const res = await generateMutation.mutateAsync({
        difficulty,
        numQuestions,
      });
      router.push(`/interview/session/${res.session_id}`);
    } catch (e) {
      setGenError(formatGenerateInterviewError(e));
    }
  };

  return (
    <div className="dashboard-card space-y-8 px-6 py-8">
      {/* Detected Role Profile */}
      <div className="rounded-xl border border-white/30 bg-white/40 px-4 py-4 backdrop-blur-sm">
        <p className="mb-1 text-xs font-medium uppercase tracking-wider text-zenodrift-text-muted">
          Detected role
        </p>
        <p className="text-base font-semibold text-zenodrift-text-strong">
          Role: {roleTitle} (Domain: {DOMAIN_LABELS[detectedDomain] ?? detectedDomain}, Level:{" "}
          {SENIORITY_LABELS[detectedSeniority] ?? detectedSeniority})
        </p>
        {coverageTotal > 0 && (
          <p className="mt-1 text-xs text-zenodrift-text-muted">
            Coverage: {coveragePracticed}/{coverageTotal}
          </p>
        )}
        {chips.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {chips.map((chip) => (
              <span
                key={chip.id}
                className="rounded-md bg-white/60 px-2.5 py-1 text-xs font-medium text-zenodrift-text"
                title={chip.attempts > 0 ? `Attempts: ${chip.attempts}, avg: ${chip.avgScore ?? "n/a"}` : undefined}
              >
                {chip.label}
              </span>
            ))}
          </div>
        )}
      </div>

      <div>
        <label className="mb-2 block text-sm font-medium text-zenodrift-text-muted">
          Difficulty
        </label>
        <select
          value={difficulty}
          onChange={(e) => setDifficulty(e.target.value as Difficulty)}
          className="w-full rounded-xl border border-white/60 bg-white/90 px-4 py-3 text-zenodrift-text shadow-md transition-colors focus:border-zenodrift-accent focus:outline-none focus:ring-2 focus:ring-zenodrift-accent/35 focus:ring-offset-2"
        >
          <option value="junior">Junior</option>
          <option value="mid">Mid</option>
          <option value="senior">Senior</option>
        </select>
      </div>

      <div>
        <label className="mb-2 block text-sm font-medium text-zenodrift-text-muted">
          Number of questions
        </label>
        <input
          type="number"
          min={1}
          max={10}
          value={numQuestions}
          onChange={(e) => setNumQuestions(Math.min(10, Math.max(1, Number(e.target.value) || 1)))}
          className="w-full rounded-xl border border-white/60 bg-white/90 px-4 py-3 text-zenodrift-text shadow-md transition-colors focus:border-zenodrift-accent focus:outline-none focus:ring-2 focus:ring-zenodrift-accent/35 focus:ring-offset-2"
        />
      </div>

      {genError && (
        <p className="text-sm text-red-600" role="alert">
          {genError}
        </p>
      )}
      <button
        onClick={handleStartInterview}
        disabled={generateMutation.isPending}
        className="w-full rounded-xl bg-gradient-to-r from-orange-500 to-orange-600 py-4 text-lg font-semibold text-white shadow-lg transition-all duration-200 hover:-translate-y-0.5 hover:shadow-xl focus:outline-none focus:ring-2 focus:ring-zenodrift-accent focus:ring-offset-2 disabled:opacity-50 disabled:hover:translate-y-0 disabled:hover:shadow-lg"
      >
        {generateMutation.isPending ? "Starting…" : "Start Interview"}
      </button>
    </div>
  );
}
