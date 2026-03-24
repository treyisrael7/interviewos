"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, type RoleProfile, type CompetencyWithCoverage } from "@/lib/api";
import {
  useDocumentSources,
  useAddInterviewSourceMutation,
  useGenerateInterviewMutation,
} from "@/hooks/use-interview-setup";

type Difficulty = "junior" | "mid" | "senior";
type QuestionMixPreset = "balanced" | "behavioral_heavy" | "scenario_heavy";
type DomainOverride = string;
type SeniorityOverride = "entry" | "mid" | "senior";

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

const DOMAIN_OPTIONS: { value: string; label: string }[] = [
  { value: "technical", label: "Technical" },
  { value: "finance", label: "Finance" },
  { value: "healthcare_social_work", label: "Healthcare & Social Work" },
  { value: "sales_marketing", label: "Sales & Marketing" },
  { value: "operations", label: "Operations" },
  { value: "education", label: "Education" },
  { value: "general_business", label: "General Business" },
];

const QUESTION_MIX_PRESETS: { value: QuestionMixPreset; label: string }[] = [
  { value: "balanced", label: "Balanced" },
  { value: "behavioral_heavy", label: "Behavioral-heavy" },
  { value: "scenario_heavy", label: "Scenario-heavy" },
];

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
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [domainOverride, setDomainOverride] = useState<DomainOverride | "">("");
  const [seniorityOverride, setSeniorityOverride] = useState<SeniorityOverride | "">("");
  const [questionMixPreset, setQuestionMixPreset] = useState<QuestionMixPreset | "">("");
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [companyPaste, setCompanyPaste] = useState("");
  const [companyUrl, setCompanyUrl] = useState("");
  const [notesText, setNotesText] = useState("");
  const [sourceError, setSourceError] = useState<string | null>(null);
  const { data: sources = [] } = useDocumentSources(documentId);
  const addSourceMutation = useAddInterviewSourceMutation(documentId);
  const generateMutation = useGenerateInterviewMutation(documentId);

  const roleTitle = roleProfile?.roleTitleGuess?.trim() || "Role";
  const detectedDomain = roleProfile?.domain || "general_business";
  const detectedSeniority = roleProfile?.seniority || "entry";
  const focusAreas = roleProfile?.focusAreas ?? [];
  const chips = competencies.length > 0
    ? competencies.slice(0, 8).map((c) => ({ id: c.id, label: c.label, attempts: c.attempts_count, avgScore: c.avg_score }))
    : focusAreas.slice(0, 8).map((a) => ({ id: a, label: a, attempts: 0, avgScore: null as number | null }));

  const handleAddCompanyPaste = () => {
    if (!companyPaste.trim()) return;
    setSourceError(null);
    addSourceMutation.mutate(
      {
        kind: "text",
        sourceType: "company",
        content: companyPaste.trim(),
        title: "Company",
      },
      {
        onSuccess: () => setCompanyPaste(""),
        onError: (e) =>
          setSourceError(
            e instanceof ApiError
              ? String(e.detail || e.message)
              : "Failed to add company info"
          ),
      }
    );
  };

  const handleAddCompanyUrl = () => {
    const url = companyUrl.trim();
    if (!url || !url.startsWith("http")) return;
    setSourceError(null);
    addSourceMutation.mutate(
      { kind: "url", url },
      {
        onSuccess: () => setCompanyUrl(""),
        onError: (e) =>
          setSourceError(
            e instanceof ApiError
              ? String(e.detail || e.message)
              : "Failed to fetch URL"
          ),
      }
    );
  };

  const handleAddNotes = () => {
    if (!notesText.trim()) return;
    setSourceError(null);
    addSourceMutation.mutate(
      {
        kind: "text",
        sourceType: "notes",
        content: notesText.trim(),
        title: "Notes",
      },
      {
        onSuccess: () => setNotesText(""),
        onError: (e) =>
          setSourceError(
            e instanceof ApiError
              ? String(e.detail || e.message)
              : "Failed to add notes"
          ),
      }
    );
  };

  const handleStartInterview = async () => {
    setGenError(null);
    try {
      const res = await generateMutation.mutateAsync({
        difficulty,
        numQuestions,
        overrides: {
          domain_override: domainOverride || undefined,
          seniority_override: seniorityOverride || undefined,
          question_mix_preset: questionMixPreset || undefined,
        },
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
                title={chip.attempts > 0 ? `Attempts: ${chip.attempts}, Avg: ${chip.avgScore ?? "—"}` : undefined}
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

      {/* Add sources (optional) - collapsible */}
      <div className="rounded-xl border border-white/25 bg-white/20">
        <button
          type="button"
          onClick={() => setSourcesOpen(!sourcesOpen)}
          className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-zenodrift-text-strong transition-colors hover:bg-white/20"
          aria-expanded={sourcesOpen}
        >
          <span>Add sources (optional)</span>
          {sources.length > 0 && (
            <span className="rounded-full bg-white/40 px-2 py-0.5 text-xs text-zenodrift-text-muted">
              {sources.length} added
            </span>
          )}
          <svg
            className={`h-4 w-4 transition-transform ${sourcesOpen ? "rotate-180" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {sourcesOpen && (
          <div className="space-y-4 border-t border-white/25 px-4 py-4">
            {sourceError && (
              <p className="text-xs text-red-600" role="alert">
                {sourceError}
              </p>
            )}

            {/* Company */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-zenodrift-text-muted">Company values / about</label>
              <div className="flex gap-2 mb-2">
                <input
                  type="url"
                  value={companyUrl}
                  onChange={(e) => setCompanyUrl(e.target.value)}
                  placeholder="Or paste URL…"
                  className="flex-1 rounded-lg border border-white/50 bg-white/80 px-3 py-2 text-sm text-zenodrift-text placeholder-neutral-400 focus:border-zenodrift-accent focus:outline-none focus:ring-2 focus:ring-zenodrift-accent/30"
                />
                <button
                  onClick={handleAddCompanyUrl}
                  disabled={
                    !companyUrl.trim() ||
                    !companyUrl.startsWith("http") ||
                    addSourceMutation.isPending
                  }
                  className="shrink-0 rounded-lg bg-white/80 px-3 py-2 text-xs font-medium text-zenodrift-accent hover:bg-white disabled:opacity-50"
                >
                  {addSourceMutation.isPending ? "Fetching…" : "Add"}
                </button>
              </div>
              <div className="flex gap-2">
                <textarea
                  value={companyPaste}
                  onChange={(e) => setCompanyPaste(e.target.value)}
                  placeholder="Or paste company/about text…"
                  rows={2}
                  className="flex-1 rounded-lg border border-white/50 bg-white/80 px-3 py-2 text-sm text-zenodrift-text placeholder-neutral-400 focus:border-zenodrift-accent focus:outline-none focus:ring-2 focus:ring-zenodrift-accent/30"
                />
                <button
                  onClick={handleAddCompanyPaste}
                  disabled={!companyPaste.trim() || addSourceMutation.isPending}
                  className="shrink-0 self-end rounded-lg bg-white/80 px-3 py-2 text-xs font-medium text-zenodrift-accent hover:bg-white disabled:opacity-50"
                >
                  {addSourceMutation.isPending ? "…" : "Add"}
                </button>
              </div>
            </div>

            {/* Notes */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-zenodrift-text-muted">Notes</label>
              <div className="flex gap-2">
                <textarea
                  value={notesText}
                  onChange={(e) => setNotesText(e.target.value)}
                  placeholder="Add notes…"
                  rows={2}
                  className="flex-1 rounded-lg border border-white/50 bg-white/80 px-3 py-2 text-sm text-zenodrift-text placeholder-neutral-400 focus:border-zenodrift-accent focus:outline-none focus:ring-2 focus:ring-zenodrift-accent/30"
                />
                <button
                  onClick={handleAddNotes}
                  disabled={!notesText.trim() || addSourceMutation.isPending}
                  className="shrink-0 self-end rounded-lg bg-white/80 px-3 py-2 text-xs font-medium text-zenodrift-accent hover:bg-white disabled:opacity-50"
                >
                  {addSourceMutation.isPending ? "…" : "Add"}
                </button>
              </div>
            </div>

            {sources.length > 0 && (
              <p className="text-xs text-zenodrift-text-muted">
                {sources.map((s) => `${s.title} (${s.source_type})`).join(" • ")} — used for tailored feedback
              </p>
            )}
          </div>
        )}
      </div>

      {/* Advanced collapsible */}
      <div className="rounded-xl border border-white/25 bg-white/20">
        <button
          type="button"
          onClick={() => setAdvancedOpen(!advancedOpen)}
          className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-zenodrift-text-strong transition-colors hover:bg-white/20"
          aria-expanded={advancedOpen}
        >
          <span>Advanced</span>
          <svg
            className={`h-4 w-4 transition-transform ${advancedOpen ? "rotate-180" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {advancedOpen && (
          <div className="space-y-4 border-t border-white/25 px-4 py-4">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-zenodrift-text-muted">
                Domain override
              </label>
              <select
                value={domainOverride}
                onChange={(e) => setDomainOverride(e.target.value as DomainOverride)}
                className="w-full rounded-lg border border-white/50 bg-white/80 px-3 py-2.5 text-sm text-zenodrift-text focus:border-zenodrift-accent focus:outline-none focus:ring-2 focus:ring-zenodrift-accent/30"
              >
                <option value="">Use detected</option>
                {DOMAIN_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-zenodrift-text-muted">
                Seniority override
              </label>
              <select
                value={seniorityOverride}
                onChange={(e) => setSeniorityOverride(e.target.value as SeniorityOverride)}
                className="w-full rounded-lg border border-white/50 bg-white/80 px-3 py-2.5 text-sm text-zenodrift-text focus:border-zenodrift-accent focus:outline-none focus:ring-2 focus:ring-zenodrift-accent/30"
              >
                <option value="">Use detected</option>
                <option value="entry">Entry</option>
                <option value="mid">Mid</option>
                <option value="senior">Senior</option>
              </select>
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-zenodrift-text-muted">
                Question mix preset
              </label>
              <select
                value={questionMixPreset}
                onChange={(e) => setQuestionMixPreset(e.target.value as QuestionMixPreset)}
                className="w-full rounded-lg border border-white/50 bg-white/80 px-3 py-2.5 text-sm text-zenodrift-text focus:border-zenodrift-accent focus:outline-none focus:ring-2 focus:ring-zenodrift-accent/30"
              >
                <option value="">Use detected</option>
                {QUESTION_MIX_PRESETS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}
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
