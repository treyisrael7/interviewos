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
  question: string
): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify({ document_id: documentId, question }),
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

export async function evaluateAnswer(
  documentId: string,
  questionId: string,
  answerText: string
): Promise<InterviewEvaluateResponse> {
  const res = await fetch(`${API_BASE}/interview/evaluate`, {
    method: "POST",
    headers: await getAuthHeaders(),
    body: JSON.stringify({
      document_id: documentId,
      question_id: questionId,
      answer_text: answerText,
    }),
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
