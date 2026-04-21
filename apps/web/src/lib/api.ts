/**
 * API client for InterviewOS backend.
 * Identity comes only from the Clerk session (Bearer token); never from client-supplied user ids.
 */

import { getAuthToken, hasAuthTokenProvider } from "./auth";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

/** Thrown when a request needs a signed-in user but no Clerk token is available. */
export class AuthRequiredError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AuthRequiredError";
  }
}

/**
 * JSON headers including Authorization: Bearer <token>.
 * @throws AuthRequiredError if Clerk is not wired or the user has no session token.
 */
export async function getAuthHeaders(): Promise<HeadersInit> {
  if (!hasAuthTokenProvider()) {
    throw new AuthRequiredError(
      "API authentication is not initialized. Ensure the app is wrapped with ClerkAuthProvider."
    );
  }
  const token = await getAuthToken();
  const trimmed = token?.trim() ?? "";
  if (!trimmed) {
    throw new AuthRequiredError("You must be signed in to use this feature.");
  }
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${trimmed}`,
  };
}

export interface RoleProfile {
  domain: string;
  seniority: string;
  roleTitleGuess: string;
  focusAreas: string[];
  questionMix: { behavioral: number; roleSpecific: number; scenario: number };
}

export interface CompetencyWithCoverage {
  id: string;
  label: string;
  description?: string | null;
  attempts_count: number;
  avg_score: number | null;
}

export interface DocumentSummary {
  id: string;
  filename: string;
  status: "pending" | "uploaded" | "processing" | "ready" | "failed";
  page_count: number | null;
  error_message: string | null;
  created_at: string;
  doc_domain?: string;
  role_profile?: RoleProfile | null;
  competencies?: CompetencyWithCoverage[];
  coverage_practiced?: number;
  coverage_total?: number;
}

export interface PresignResponse {
  document_id: string;
  s3_key: string;
  upload_url: string;
  method: string;
}

export interface AskResponse {
  answer: string;
  citations: Array<{
    chunk_id: string;
    page_number: number;
    snippet: string;
  }>;
}

/** Parsed body when /ask returns structured JSON (see generate_grounded_answer). */
export interface AskFitMatch {
  requirement: string;
  candidate_experience: string;
  alignment_notes: string;
}

export interface AskFitGap {
  requirement: string;
  reason: string;
}

export interface AskStructuredAnswer {
  key_job_requirements: string[];
  matches: AskFitMatch[];
  gaps: AskFitGap[];
  fit_score: number;
  reasoning: string;
}

/**
 * If ``answer`` is JSON from the grounded Q&A endpoint, return a typed object for UI.
 * Otherwise returns null (show plain text fallback).
 */
export function parseAskStructuredAnswer(answer: string): AskStructuredAnswer | null {
  const trimmed = answer?.trim() ?? "";
  if (!trimmed.startsWith("{")) return null;
  try {
    const raw = JSON.parse(trimmed) as unknown;
    if (!raw || typeof raw !== "object") return null;
    const o = raw as Record<string, unknown>;
    const reqs = o.key_job_requirements;
    const matches = o.matches;
    const gaps = o.gaps;
    const fit = o.fit_score;
    const reasoning = o.reasoning;
    if (!Array.isArray(reqs) || !Array.isArray(matches) || !Array.isArray(gaps))
      return null;
    const fitNum = typeof fit === "number" ? fit : Number(fit);
    if (!Number.isFinite(fitNum) || typeof reasoning !== "string") return null;
    const key_job_requirements = reqs.filter((x): x is string => typeof x === "string");
    const matchList: AskFitMatch[] = [];
    for (const x of matches) {
      if (!x || typeof x !== "object") continue;
      const m = x as Record<string, unknown>;
      if (typeof m.requirement !== "string" || typeof m.candidate_experience !== "string")
        continue;
      matchList.push({
        requirement: m.requirement,
        candidate_experience: m.candidate_experience,
        alignment_notes:
          typeof m.alignment_notes === "string" ? m.alignment_notes : "",
      });
    }
    const gapList: AskFitGap[] = [];
    for (const x of gaps) {
      if (!x || typeof x !== "object") continue;
      const g = x as Record<string, unknown>;
      if (typeof g.requirement !== "string" || typeof g.reason !== "string") continue;
      gapList.push({ requirement: g.requirement, reason: g.reason });
    }
    const fit_score = Math.min(100, Math.max(0, Math.round(fitNum)));
    return {
      key_job_requirements,
      matches: matchList,
      gaps: gapList,
      fit_score,
      reasoning,
    };
  } catch {
    return null;
  }
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: unknown
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  const text = await res.text();
  let json: unknown = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    // ignore
  }
  if (!res.ok) {
    const detail =
      typeof json === "object" && json && "detail" in json
        ? (json as { detail?: unknown }).detail
        : text;
    throw new ApiError(
      res.statusText || "Request failed",
      res.status,
      detail
    );
  }
  return json as T;
}

export async function listDocuments(): Promise<DocumentSummary[]> {
  const res = await fetch(`${API_BASE}/documents`, {
    headers: await getAuthHeaders(),
  });
  return handleResponse<DocumentSummary[]>(res);
}

export async function getDocument(id: string): Promise<DocumentSummary> {
  const res = await fetch(`${API_BASE}/documents/${id}`, {
    headers: await getAuthHeaders(),
  });
  return handleResponse<DocumentSummary>(res);
}

export async function deleteDocument(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${id}`, {
    method: "DELETE",
    headers: await getAuthHeaders(),
  });
  return handleResponse<void>(res);
}

export async function deleteAllDocuments(): Promise<{ count: number }> {
  const res = await fetch(`${API_BASE}/documents`, {
    method: "DELETE",
    headers: await getAuthHeaders(),
  });
  return handleResponse<{ count: number }>(res);
}

export async function presign(
  filename: string,
  fileSizeBytes: number
): Promise<PresignResponse> {
  const res = await fetch(`${API_BASE}/documents/presign`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify({
      filename,
      content_type: "application/pdf",
      file_size_bytes: fileSizeBytes,
    }),
  });
  return handleResponse<PresignResponse>(res);
}

export async function uploadToPresignedUrl(
  uploadUrl: string,
  file: File
): Promise<void> {
  const res = await fetch(uploadUrl, {
    method: "PUT",
    headers: { "Content-Type": "application/pdf" },
    body: file,
  });
  if (!res.ok) {
    throw new ApiError("Upload failed", res.status, await res.text());
  }
}

export async function confirmUpload(
  documentId: string,
  s3Key: string
): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/documents/confirm`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify({ document_id: documentId, s3_key: s3Key }),
  });
  return handleResponse<{ status: string }>(res);
}

export async function ingestDocument(documentId: string): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/documents/${documentId}/ingest`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify({}),
  });
  return handleResponse<{ status: string }>(res);
}

// --- Interview Kit Sources ---

export interface SourceSummary {
  id: string;
  source_type: string;
  title: string;
}

export async function listSources(documentId: string): Promise<SourceSummary[]> {
  const res = await fetch(`${API_BASE}/documents/${documentId}/sources`, {
    headers: await getAuthHeaders(),
  });
  return handleResponse<SourceSummary[]>(res);
}

export async function addTextSource(
  documentId: string,
  sourceType: "resume" | "company" | "notes",
  content: string,
  title?: string
): Promise<{ source_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/documents/${documentId}/sources/add-text`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify({
      source_type: sourceType,
      title: title || "",
      content,
    }),
  });
  return handleResponse<{ source_id: string; status: string }>(res);
}

export async function presignResume(
  documentId: string,
  filename: string,
  fileSizeBytes: number
): Promise<{ s3_key: string; upload_url: string; method: string }> {
  const res = await fetch(
    `${API_BASE}/documents/${documentId}/sources/presign-resume`,
    {
      method: "POST",
      headers: await getAuthHeaders(),
      body: JSON.stringify({ filename, file_size_bytes: fileSizeBytes }),
    }
  );
  return handleResponse<{ s3_key: string; upload_url: string; method: string }>(res);
}

export async function ingestResume(
  documentId: string,
  s3Key: string
): Promise<{ source_id: string; status: string }> {
  const res = await fetch(
    `${API_BASE}/documents/${documentId}/sources/ingest-resume`,
    {
      method: "POST",
      headers: await getAuthHeaders(),
      body: JSON.stringify({ s3_key: s3Key }),
    }
  );
  return handleResponse<{ source_id: string; status: string }>(res);
}

// --- User Resume (account-level, applies to all job descriptions) ---

export interface UserResumeStatus {
  has_resume: boolean;
  filename?: string | null;
  /** Present when ``has_resume``; use with /ask ``additional_document_ids``. */
  document_id?: string | null;
}

export async function getUserResume(): Promise<UserResumeStatus> {
  const res = await fetch(`${API_BASE}/user/resume`, {
    headers: await getAuthHeaders(),
  });
  return handleResponse<UserResumeStatus>(res);
}

export async function presignUserResume(
  filename: string,
  fileSizeBytes: number
): Promise<{ s3_key: string; upload_url: string; method: string }> {
  const res = await fetch(`${API_BASE}/user/resume/presign`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify({ filename, file_size_bytes: fileSizeBytes }),
  });
  return handleResponse<{ s3_key: string; upload_url: string; method: string }>(res);
}

export async function confirmUserResume(
  s3Key: string
): Promise<{ source_id: string; status: string; document_id: string }> {
  const res = await fetch(`${API_BASE}/user/resume`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify({ s3_key: s3Key }),
  });
  return handleResponse<{ source_id: string; status: string; document_id: string }>(res);
}

export async function deleteUserResume(): Promise<{ status: string; message?: string }> {
  const res = await fetch(`${API_BASE}/user/resume`, {
    method: "DELETE",
    headers: await getAuthHeaders(),
  });
  return handleResponse<{ status: string; message?: string }>(res);
}

/** POST /user/resume/ask: resume improvement coaching from profile resume text. */
export async function askProfileResumeCoach(question: string): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/user/resume/ask`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify({ question }),
  });
  return handleResponse<AskResponse>(res);
}

export interface ResumeCoachEdit {
  focus: string;
  observation: string;
  suggestion: string;
}

export interface ResumeCoachStructuredAnswer {
  coaching_reply: string;
  prioritized_edits: ResumeCoachEdit[];
  strengths_to_keep: string[];
  reasoning: string;
}

export function parseResumeCoachAnswer(answer: string): ResumeCoachStructuredAnswer | null {
  const trimmed = answer?.trim() ?? "";
  if (!trimmed.startsWith("{")) return null;
  try {
    const raw = JSON.parse(trimmed) as unknown;
    if (!raw || typeof raw !== "object") return null;
    const o = raw as Record<string, unknown>;
    const reply = o.coaching_reply;
    const edits = o.prioritized_edits;
    const strengths = o.strengths_to_keep;
    const reasoning = o.reasoning;
    if (typeof reply !== "string" || typeof reasoning !== "string") return null;
    if (!Array.isArray(edits) || !Array.isArray(strengths)) return null;
    const prioritized_edits: ResumeCoachEdit[] = [];
    for (const x of edits) {
      if (!x || typeof x !== "object") continue;
      const e = x as Record<string, unknown>;
      if (
        typeof e.focus !== "string" ||
        typeof e.observation !== "string" ||
        typeof e.suggestion !== "string"
      )
        continue;
      prioritized_edits.push({
        focus: e.focus,
        observation: e.observation,
        suggestion: e.suggestion,
      });
    }
    const strengths_to_keep = strengths.filter((x): x is string => typeof x === "string");
    return {
      coaching_reply: reply,
      prioritized_edits,
      strengths_to_keep,
      reasoning,
    };
  } catch {
    return null;
  }
}

export async function addSourceFromUrl(
  documentId: string,
  url: string,
  title?: string
): Promise<{ source_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/documents/${documentId}/sources/add-from-url`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify({
      url,
      title: title || "Company / About",
    }),
  });
  return handleResponse<{ source_id: string; status: string }>(res);
}

export async function ask(
  documentId: string,
  question: string,
  options?: { resumeDocumentId?: string | null }
): Promise<AskResponse> {
  const body: Record<string, unknown> = { document_id: documentId, question };
  const rid = options?.resumeDocumentId?.trim();
  if (rid) {
    body.additional_document_ids = [rid];
  }
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse<AskResponse>(res);
}

// --- Analyze fit (JD + account resume) ---

export interface AnalyzeFitMatch {
  requirement: string;
  resume_evidence: string;
  confidence: number;
  importance: string;
}

export interface AnalyzeFitGap {
  requirement: string;
  reason: string;
  importance: string;
}

export interface AnalyzeFitRecommendation {
  gap: string;
  suggestion: string;
  missing_keywords: string[];
  bullet_rewrite: string;
  example_resume_line: string;
}

export interface AnalyzeFitResult {
  matches: AnalyzeFitMatch[];
  gaps: AnalyzeFitGap[];
  fit_score: number;
  summary: string;
  recommendations: AnalyzeFitRecommendation[];
}

export interface AnalyzeFitLatestPayload {
  has_analysis: boolean;
  analysis: AnalyzeFitResult | null;
  created_at: string | null;
  cache_hit_default_question: boolean;
}

export async function getAnalyzeFitLatest(
  jobDescriptionId: string,
  resumeId: string
): Promise<AnalyzeFitLatestPayload> {
  const params = new URLSearchParams({
    job_description_id: jobDescriptionId,
    resume_id: resumeId,
  });
  const res = await fetch(`${API_BASE}/analyze-fit/latest?${params}`, {
    headers: await getAuthHeaders(),
  });
  return handleResponse<AnalyzeFitLatestPayload>(res);
}

export async function analyzeFit(params: {
  jobDescriptionId: string;
  resumeId: string;
  question?: string | null;
}): Promise<AnalyzeFitResult> {
  const body: Record<string, unknown> = {
    job_description_id: params.jobDescriptionId,
    resume_id: params.resumeId,
  };
  const q = params.question?.trim();
  if (q) body.question = q;

  const res = await fetch(`${API_BASE}/analyze-fit`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse<AnalyzeFitResult>(res);
}

// --- Role-specific study plan (JD only) ---

export interface StudyPlanDay {
  day: number;
  theme: string;
  topics: string[];
  drills: string[];
  mock_target: string;
}

export interface StudyPlanResult {
  title: string;
  role_title: string;
  duration_days: number;
  summary: string;
  daily_plan: StudyPlanDay[];
}

export async function generateStudyPlan(params: {
  documentId: string;
  days: number;
  focus?: string | null;
}): Promise<StudyPlanResult> {
  const body: Record<string, unknown> = { days: params.days };
  const focus = params.focus?.trim();
  if (focus) body.focus = focus;

  const res = await fetch(`${API_BASE}/documents/${params.documentId}/study-plan`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse<StudyPlanResult>(res);
}

// --- Interview Prep API ---

/** Per JD dimension: 0–10 score and why that score was assigned. */
export interface RubricScoreItem {
  name: string;
  score: number;
  reasoning: string;
}

/** Stored evaluation snapshot from interview_answers.evaluation_json (score, strengths, gaps, citations). */
export type InterviewEvaluationJson = Record<string, unknown> & {
  evaluation_mode?: "lite" | "full";
  score?: number;
  summary?: string;
  /** Why this score vs rubric (strengths/gaps). */
  score_reasoning?: string;
  strengths?: unknown[];
  gaps?: unknown[];
  citations?: unknown[];
  improved_answer?: string;
  /** Per-dimension scores (0–10) with reasoning when JD rubric dimensions exist. */
  rubric_scores?: RubricScoreItem[];
};

export interface InterviewQuestion {
  id: string;
  type: string;
  focus_area?: string;
  competency_id?: string | null;
  competency_label?: string | null;
  question: string;
  key_topics: string[];
  evidence: Array<{ chunk_id: string; page_number: number; snippet: string }>;
  rubric_bullets: string[];
  last_answer_id?: string | null;
  evaluation_json?: InterviewEvaluationJson | null;
}

export interface InterviewGenerateResponse {
  session_id: string;
  questions: InterviewQuestion[];
}

export interface EvidenceUsedItem {
  quote: string;
  sourceId: string;
  sourceType?: string;
  sourceTitle?: string;
  page?: number;
  chunkId?: string;
}

export interface CitationItem {
  chunkId: string;
  page?: number | null;
  sourceTitle?: string;
  sourceType?: string;
}

export interface CitedItem {
  text: string;
  citations: CitationItem[];
}

export interface StrengthEvalItem {
  text: string;
  /** Quote from the answer supporting this strength. */
  evidence: string;
  /** Verbatim phrase from the candidate's answer (for UI emphasis). */
  highlight?: string;
  /** Why this strength matters for the role (JD/rubric). */
  impact?: string;
}

export interface GapEvalItem {
  text: string;
  /** What is absent or weak vs rubric/JD (omitted on older stored evaluations). */
  missing?: string;
  expected: string;
  /** How the answer does or does not match the job requirement. */
  jd_alignment?: string;
  /** Concrete phrasing the candidate should say instead. */
  improvement?: string;
}

export interface EvaluationCitation {
  chunk_id: string;
  page_number: number;
  text: string;
}

/** Monthly evaluation quota (UTC) after this evaluate call. */
export interface EvaluationUsage {
  plan: string;
  evaluations_used_this_month: number;
  evaluation_limit: number;
}

export interface InterviewEvaluateResponse {
  answer_id: string;
  /** lite: score + short feedback; full: strengths, gaps, improved answer, etc. */
  evaluation_mode: "lite" | "full";
  usage: EvaluationUsage;
  /** Rubric aggregate (0–100). */
  score: number;
  /** Model-reported score (0–10). */
  llm_score: number;
  /** Short explanation of why the score was given. */
  summary: string;
  /** 1–2 sentences tying the score to rubric expectations via strengths and gaps. */
  score_reasoning: string;
  strengths: StrengthEvalItem[];
  gaps: GapEvalItem[];
  citations: EvaluationCitation[];
  strengths_cited?: CitedItem[];
  gaps_cited?: CitedItem[];
  /** Full rewrite targeting 9–10/10: same idea, more depth, tools/metrics when relevant. */
  improved_answer: string;
  follow_up_questions: string[];
  suggested_followup?: string | null;
  evidence_used: EvidenceUsedItem[];
  /** Per-dimension scores (0–10); reasoning explains each score. */
  rubric_scores: RubricScoreItem[];
  evaluation_json: InterviewEvaluationJson;
}

export interface InterviewSessionSummary {
  id: string;
  document_id: string;
  mode: string;
  difficulty: string;
  created_at: string;
  question_count: number;
}

/** Session detail as returned by the API (user_id is server-owned, not sent by the client). */
export interface InterviewSessionDetail {
  id: string;
  user_id: string;
  document_id: string;
  mode: string;
  difficulty: string;
  created_at: string;
  questions: InterviewQuestion[];
  role_profile?: RoleProfile | null;
  /** Present after at least one evaluated answer; drives adaptive question generation. */
  performance_profile?: Record<string, number> | null;
  /** Human-readable focus derived from performance_profile (server-computed). */
  adaptive_focus_label?: string | null;
  /** JD evaluation dimensions from document ingestion (job_description docs). */
  rubric_json?: { name: string; description: string }[] | null;
}

export interface InterviewGenerateOverrides {
  domain_override?: string;
  seniority_override?: "entry" | "mid" | "senior";
  question_mix_preset?: "balanced" | "behavioral_heavy" | "scenario_heavy";
}

export async function generateInterview(
  documentId: string,
  difficulty: "junior" | "mid" | "senior",
  numQuestions: number,
  overrides?: InterviewGenerateOverrides
): Promise<InterviewGenerateResponse> {
  const body: Record<string, unknown> = {
    document_id: documentId,
    difficulty,
    num_questions: numQuestions,
  };
  if (overrides?.domain_override) body.domain_override = overrides.domain_override;
  if (overrides?.seniority_override) body.seniority_override = overrides.seniority_override;
  if (overrides?.question_mix_preset) body.question_mix_preset = overrides.question_mix_preset;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120_000);
  try {
    const res = await fetch(`${API_BASE}/interview/generate`, {
      method: "POST",
      headers: await getAuthHeaders(),
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    return handleResponse<InterviewGenerateResponse>(res);
  } finally {
    clearTimeout(timeoutId);
  }
}

/** Chunk ids cited or used as evidence in an evaluation response (for retrieval feedback). */
export function collectEvalChunkIds(res: InterviewEvaluateResponse): string[] {
  const ids: string[] = [];
  const seen = new Set<string>();
  const take = (s: string | null | undefined) => {
    const t = (s ?? "").trim();
    if (!t || seen.has(t)) return;
    seen.add(t);
    ids.push(t);
  };
  for (const c of res.citations ?? []) take(c.chunk_id);
  for (const e of res.evidence_used ?? []) take(e.chunkId);
  for (const item of res.strengths_cited ?? []) {
    for (const cit of item.citations ?? []) take(cit.chunkId);
  }
  for (const item of res.gaps_cited ?? []) {
    for (const cit of item.citations ?? []) take(cit.chunkId);
  }
  return ids;
}

export interface InterviewRetrievalFeedbackResponse {
  id: string;
  updated: boolean;
}

export async function submitInterviewRetrievalFeedback(
  documentId: string,
  answerId: string,
  options?: { reason?: string; retrieval_chunk_ids?: string[] }
): Promise<InterviewRetrievalFeedbackResponse> {
  const body: Record<string, unknown> = {
    document_id: documentId,
    answer_id: answerId,
  };
  if (options?.reason?.trim()) body.reason = options.reason.trim();
  if (options?.retrieval_chunk_ids?.length) {
    body.retrieval_chunk_ids = options.retrieval_chunk_ids;
  }
  const res = await fetch(`${API_BASE}/interview/retrieval-feedback`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse<InterviewRetrievalFeedbackResponse>(res);
}

export async function evaluateAnswer(
  documentId: string,
  questionId: string,
  answerText: string,
  options?: { mode?: "lite" | "full" }
): Promise<InterviewEvaluateResponse> {
  const body: Record<string, unknown> = {
    document_id: documentId,
    question_id: questionId,
    answer_text: answerText,
  };
  if (options?.mode) {
    body.mode = options.mode;
  }
  const res = await fetch(`${API_BASE}/interview/evaluate`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse<InterviewEvaluateResponse>(res);
}

export async function listInterviewSessions(): Promise<InterviewSessionSummary[]> {
  const res = await fetch(`${API_BASE}/interview/sessions`, {
    headers: await getAuthHeaders(),
  });
  return handleResponse<InterviewSessionSummary[]>(res);
}

export async function getInterviewSession(
  sessionId: string
): Promise<InterviewSessionDetail> {
  const res = await fetch(`${API_BASE}/interview/sessions/${sessionId}`, {
    headers: await getAuthHeaders(),
  });
  return handleResponse<InterviewSessionDetail>(res);
}

export interface InterviewAnalyticsCompetencyStat {
  competency_id: string | null;
  competency_label: string;
  average_score: number;
  answer_count: number;
}

export interface InterviewAnalyticsTrendPoint {
  at: string;
  score: number;
  session_id: string;
  question_id: string;
}

export interface InterviewAnalyticsRecentSession {
  id: string;
  document_id: string;
  created_at: string;
  difficulty: string;
  question_count: number;
  answer_count: number;
  average_score: number | null;
}

export interface InterviewAnalyticsOverview {
  total_session_count: number;
  total_answer_count: number;
  overall_average_score: number | null;
  score_trend: InterviewAnalyticsTrendPoint[];
  strongest_competencies: InterviewAnalyticsCompetencyStat[];
  weakest_competencies: InterviewAnalyticsCompetencyStat[];
  recent_sessions: InterviewAnalyticsRecentSession[];
  last_session_vs_prior_percent_change: number | null;
  focus_area_hint: string | null;
}

export async function getInterviewAnalyticsOverview(): Promise<InterviewAnalyticsOverview> {
  const res = await fetch(`${API_BASE}/interview/analytics/overview`, {
    headers: await getAuthHeaders(),
  });
  return handleResponse<InterviewAnalyticsOverview>(res);
}
