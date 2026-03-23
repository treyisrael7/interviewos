/**
 * API client for InterviewOS backend.
 * When Clerk is used: passes Bearer token, backend gets user from JWT.
 * When demo mode: passes user_id + x-demo-key.
 */

import { getAuthToken } from "./auth";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
const DEMO_KEY = process.env.NEXT_PUBLIC_DEMO_KEY || "";
const DEMO_USER_ID = "11111111-1111-1111-1111-111111111111";

export async function getAuthHeaders(): Promise<HeadersInit> {
  const h: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const token = await getAuthToken();
  if (token) {
    h["Authorization"] = `Bearer ${token}`;
  } else if (DEMO_KEY) {
    h["x-demo-key"] = DEMO_KEY;
  }
  return h;
}

/** Include user_id in params only when using demo mode (no Clerk token) */
async function withUserId<T extends Record<string, unknown>>(params: T): Promise<T & { user_id?: string }> {
  const token = await getAuthToken();
  if (token) return params;
  return { ...params, user_id: DEMO_USER_ID };
}

/** Build query string; omit user_id when using Clerk */
async function queryWithUserId(params: Record<string, string>): Promise<string> {
  const token = await getAuthToken();
  const p = { ...params };
  if (!token) p.user_id = DEMO_USER_ID;
  return new URLSearchParams(p).toString();
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
  const q = await queryWithUserId({});
  const res = await fetch(
    `${API_BASE}/documents?${q}`,
    { headers: await getAuthHeaders() }
  );
  return handleResponse<DocumentSummary[]>(res);
}

export async function getDocument(id: string): Promise<DocumentSummary> {
  const q = await queryWithUserId({});
  const res = await fetch(
    `${API_BASE}/documents/${id}?${q}`,
    { headers: await getAuthHeaders() }
  );
  return handleResponse<DocumentSummary>(res);
}

export async function deleteDocument(id: string): Promise<void> {
  const q = await queryWithUserId({});
  const res = await fetch(
    `${API_BASE}/documents/${id}?${q}`,
    {
      method: "DELETE",
      headers: await getAuthHeaders(),
    }
  );
  return handleResponse<void>(res);
}

export async function deleteAllDocuments(): Promise<{ count: number }> {
  const q = await queryWithUserId({});
  const res = await fetch(`${API_BASE}/documents?${q}`, {
    method: "DELETE",
    headers: await getAuthHeaders(),
  });
  return handleResponse<{ count: number }>(res);
}

export async function presign(
  filename: string,
  fileSizeBytes: number
): Promise<PresignResponse> {
  const body = await withUserId({
    filename,
    content_type: "application/pdf",
    file_size_bytes: fileSizeBytes,
  });
  const res = await fetch(`${API_BASE}/documents/presign`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify(body),
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
  const body = await withUserId({ document_id: documentId, s3_key: s3Key });
  const res = await fetch(`${API_BASE}/documents/confirm`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse<{ status: string }>(res);
}

export async function ingestDocument(documentId: string): Promise<{ status: string }> {
  const body = await withUserId({});
  const res = await fetch(`${API_BASE}/documents/${documentId}/ingest`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify(body),
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
  const q = await queryWithUserId({});
  const res = await fetch(
    `${API_BASE}/documents/${documentId}/sources?${q}`,
    { headers: await getAuthHeaders() }
  );
  return handleResponse<SourceSummary[]>(res);
}

export async function addTextSource(
  documentId: string,
  sourceType: "resume" | "company" | "notes",
  content: string,
  title?: string
): Promise<{ source_id: string; status: string }> {
  const body = await withUserId({
    source_type: sourceType,
    title: title || "",
    content,
  });
  const res = await fetch(`${API_BASE}/documents/${documentId}/sources/add-text`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse<{ source_id: string; status: string }>(res);
}

export async function presignResume(documentId: string, filename: string, fileSizeBytes: number): Promise<{ s3_key: string; upload_url: string; method: string }> {
  const body = await withUserId({ filename, file_size_bytes: fileSizeBytes });
  const res = await fetch(`${API_BASE}/documents/${documentId}/sources/presign-resume`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse<{ s3_key: string; upload_url: string; method: string }>(res);
}

export async function ingestResume(documentId: string, s3Key: string): Promise<{ source_id: string; status: string }> {
  const body = await withUserId({ s3_key: s3Key });
  const res = await fetch(`${API_BASE}/documents/${documentId}/sources/ingest-resume`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse<{ source_id: string; status: string }>(res);
}

// --- User Resume (account-level, applies to all job descriptions) ---

export interface UserResumeStatus {
  has_resume: boolean;
  filename?: string | null;
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

export async function confirmUserResume(s3Key: string): Promise<{ source_id: string; status: string; document_id: string }> {
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

export async function addSourceFromUrl(
  documentId: string,
  url: string,
  title?: string
): Promise<{ source_id: string; status: string }> {
  const body = await withUserId({
    url,
    title: title || "Company / About",
  });
  const res = await fetch(`${API_BASE}/documents/${documentId}/sources/add-from-url`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse<{ source_id: string; status: string }>(res);
}

export async function ask(
  documentId: string,
  question: string
): Promise<AskResponse> {
  const body = await withUserId({ document_id: documentId, question });
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse<AskResponse>(res);
}

// --- Interview Prep API ---

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

export interface InterviewEvaluateResponse {
  answer_id: string;
  score: number;
  strengths: string[];
  gaps: string[];
  strengths_cited?: CitedItem[];
  gaps_cited?: CitedItem[];
  improved_answer: string;
  follow_up_questions: string[];
  suggested_followup?: string | null;
  evidence_used: EvidenceUsedItem[];
}

export interface InterviewSessionSummary {
  id: string;
  document_id: string;
  mode: string;
  difficulty: string;
  created_at: string;
  question_count: number;
}

export interface InterviewSessionDetail {
  id: string;
  user_id: string;
  document_id: string;
  mode: string;
  difficulty: string;
  created_at: string;
  questions: InterviewQuestion[];
  role_profile?: RoleProfile | null;
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
  const base: Record<string, unknown> = {
    document_id: documentId,
    difficulty,
    num_questions: numQuestions,
  };
  if (overrides?.domain_override) base.domain_override = overrides.domain_override;
  if (overrides?.seniority_override) base.seniority_override = overrides.seniority_override;
  if (overrides?.question_mix_preset) base.question_mix_preset = overrides.question_mix_preset;
  const body = await withUserId(base);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120_000); // 2 min for LLM
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

export async function evaluateAnswer(
  documentId: string,
  questionId: string,
  answerText: string
): Promise<InterviewEvaluateResponse> {
  const body = await withUserId({
    document_id: documentId,
    question_id: questionId,
    answer_text: answerText,
  });
  const res = await fetch(`${API_BASE}/interview/evaluate`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse<InterviewEvaluateResponse>(res);
}

export async function listInterviewSessions(): Promise<InterviewSessionSummary[]> {
  const q = await queryWithUserId({});
  const res = await fetch(
    `${API_BASE}/interview/sessions?${q}`,
    { headers: await getAuthHeaders() }
  );
  return handleResponse<InterviewSessionSummary[]>(res);
}

export async function getInterviewSession(
  sessionId: string
): Promise<InterviewSessionDetail> {
  const q = await queryWithUserId({});
  const res = await fetch(
    `${API_BASE}/interview/sessions/${sessionId}?${q}`,
    { headers: await getAuthHeaders() }
  );
  return handleResponse<InterviewSessionDetail>(res);
}

