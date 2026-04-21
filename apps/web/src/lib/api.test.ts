import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { setAuthTokenProvider } from "./auth";

describe("api", () => {
  beforeEach(() => {
    setAuthTokenProvider(() => Promise.resolve("test-jwt-token"));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("getAuthHeaders", () => {
    it("includes Bearer when token is present", async () => {
      setAuthTokenProvider(() => Promise.resolve("jwt-xyz"));
      const { getAuthHeaders } = await import("./api");
      const headers = await getAuthHeaders();
      const h = headers as Record<string, string>;
      expect(h["Authorization"]).toBe("Bearer jwt-xyz");
      expect(h["Content-Type"]).toBe("application/json");
    });

    it("throws AuthRequiredError when provider returns null", async () => {
      setAuthTokenProvider(() => Promise.resolve(null));
      const { getAuthHeaders, AuthRequiredError } = await import("./api");
      await expect(getAuthHeaders()).rejects.toThrow(AuthRequiredError);
      await expect(getAuthHeaders()).rejects.toThrow(/signed in/i);
    });

    it("throws AuthRequiredError when provider returns only whitespace", async () => {
      setAuthTokenProvider(() => Promise.resolve("   \t  "));
      const { getAuthHeaders, AuthRequiredError } = await import("./api");
      await expect(getAuthHeaders()).rejects.toThrow(AuthRequiredError);
    });
  });

  describe("listDocuments", () => {
    it("calls /documents without user_id query", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(JSON.stringify([])),
      });
      vi.stubGlobal("fetch", mockFetch);

      const { listDocuments } = await import("./api");
      await listDocuments();

      expect(mockFetch).toHaveBeenCalledTimes(1);
      const [url, init] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/documents");
      expect(url).not.toContain("user_id");
      expect((init?.headers as Record<string, string>)["Authorization"]).toMatch(/^Bearer /);
    });

    it("returns parsed documents on success", async () => {
      const docs = [{ id: "1", filename: "a.pdf", status: "ready" }];
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(docs)),
      });
      vi.stubGlobal("fetch", mockFetch);

      const { listDocuments } = await import("./api");
      const result = await listDocuments();

      expect(result).toEqual(docs);
    });
  });

  describe("analyzeFit", () => {
    it("POSTs job_description_id and resume_id", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              matches: [],
              gaps: [],
              fit_score: 50,
              summary: "ok",
              recommendations: [],
            })
          ),
      });
      vi.stubGlobal("fetch", mockFetch);

      const { analyzeFit } = await import("./api");
      await analyzeFit({
        jobDescriptionId: "jd-uuid",
        resumeId: "rs-uuid",
      });

      const [, init] = mockFetch.mock.calls[0];
      const body = JSON.parse((init as RequestInit).body as string);
      expect(body).toEqual({
        job_description_id: "jd-uuid",
        resume_id: "rs-uuid",
      });
    });

    it("includes question when provided", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              matches: [],
              gaps: [],
              fit_score: 0,
              summary: "",
              recommendations: [],
            })
          ),
      });
      vi.stubGlobal("fetch", mockFetch);

      const { analyzeFit } = await import("./api");
      await analyzeFit({
        jobDescriptionId: "a",
        resumeId: "b",
        question: "  FP&A focus  ",
      });

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.question).toBe("FP&A focus");
    });
  });

  describe("generateStudyPlan", () => {
    it("POSTs days and optional focus to /documents/:id/study-plan", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              title: "10-Day Interview Study Plan",
              role_title: "Backend Engineer",
              duration_days: 10,
              summary: "Plan summary",
              daily_plan: [],
            })
          ),
      });
      vi.stubGlobal("fetch", mockFetch);

      const { generateStudyPlan } = await import("./api");
      await generateStudyPlan({
        documentId: "doc-123",
        days: 10,
        focus: "  system design  ",
      });

      const [url, init] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8000/documents/doc-123/study-plan");
      const body = JSON.parse((init as RequestInit).body as string);
      expect(body).toEqual({ days: 10, focus: "system design" });
    });
  });

  describe("ask", () => {
    it("sends document_id and question only (no user_id)", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ answer: "Yes", citations: [] })),
      });
      vi.stubGlobal("fetch", mockFetch);

      const { ask } = await import("./api");
      await ask("doc-123", "What is the salary?");

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.user_id).toBeUndefined();
      expect(body.document_id).toBe("doc-123");
      expect(body.question).toBe("What is the salary?");
      expect(body.additional_document_ids).toBeUndefined();
    });

    it("parseAskStructuredAnswer returns typed object for valid JSON", async () => {
      const { parseAskStructuredAnswer } = await import("./api");
      const json = JSON.stringify({
        key_job_requirements: ["Python"],
        matches: [
          {
            requirement: "Python",
            candidate_experience: "5 years",
            alignment_notes: "Strong",
          },
        ],
        gaps: [{ requirement: "Rust", reason: "Not listed" }],
        fit_score: 72,
        reasoning: "Good Python fit.",
      });
      const p = parseAskStructuredAnswer(json);
      expect(p).not.toBeNull();
      expect(p!.fit_score).toBe(72);
      expect(p!.matches).toHaveLength(1);
      expect(p!.gaps).toHaveLength(1);
      expect(p!.key_job_requirements).toEqual(["Python"]);
    });

    it("parseAskStructuredAnswer coerces missing alignment_notes to empty string", async () => {
      const { parseAskStructuredAnswer } = await import("./api");
      const json = JSON.stringify({
        key_job_requirements: [],
        matches: [{ requirement: "A", candidate_experience: "B" }],
        gaps: [],
        fit_score: 50,
        reasoning: "x",
      });
      const p = parseAskStructuredAnswer(json);
      expect(p?.matches[0].alignment_notes).toBe("");
    });

    it("parseAskStructuredAnswer returns null for non-JSON", async () => {
      const { parseAskStructuredAnswer } = await import("./api");
      expect(parseAskStructuredAnswer("plain text")).toBeNull();
      expect(parseAskStructuredAnswer("{")).toBeNull();
    });

    it("askProfileResumeCoach POSTs question to /user/resume/ask", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        text: () =>
          Promise.resolve(JSON.stringify({ answer: "{}", citations: [] })),
      });
      vi.stubGlobal("fetch", mockFetch);

      const { askProfileResumeCoach } = await import("./api");
      await askProfileResumeCoach("How can I improve my bullets?");

      expect(mockFetch.mock.calls[0][0]).toContain("/user/resume/ask");
      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body).toEqual({ question: "How can I improve my bullets?" });
    });

    it("parseResumeCoachAnswer returns typed object for valid JSON", async () => {
      const { parseResumeCoachAnswer } = await import("./api");
      const json = JSON.stringify({
        coaching_reply: "Add metrics.",
        prioritized_edits: [
          {
            focus: "Experience",
            observation: "Vague verbs",
            suggestion: "Use numbers",
          },
        ],
        strengths_to_keep: ["Clear education"],
        reasoning: "Based on page 1.",
      });
      const p = parseResumeCoachAnswer(json);
      expect(p).not.toBeNull();
      expect(p!.coaching_reply).toBe("Add metrics.");
      expect(p!.prioritized_edits).toHaveLength(1);
      expect(p!.strengths_to_keep).toEqual(["Clear education"]);
    });

    it("includes additional_document_ids when resumeDocumentId is set", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ answer: "{}", citations: [] })),
      });
      vi.stubGlobal("fetch", mockFetch);

      const { ask } = await import("./api");
      await ask("doc-jd", "Fit?", { resumeDocumentId: "doc-resume-uuid" });

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.additional_document_ids).toEqual(["doc-resume-uuid"]);
    });
  });

  describe("ApiError", () => {
    it("throws ApiError with status and detail on non-OK response", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        statusText: "Not Found",
        text: () => Promise.resolve(JSON.stringify({ detail: "Document not found" })),
      });
      vi.stubGlobal("fetch", mockFetch);

      const { getDocument } = await import("./api");

      try {
        await getDocument("missing-id");
        expect.fail("Should have thrown");
      } catch (e) {
        expect((e as Error).name).toBe("ApiError");
        expect((e as { status: number; detail?: unknown }).status).toBe(404);
        expect((e as { status: number; detail?: unknown }).detail).toBe("Document not found");
      }
    });

    it("preserves detail from JSON detail field", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 401,
        statusText: "Unauthorized",
        text: () => Promise.resolve(JSON.stringify({ detail: "Authentication required" })),
      });
      vi.stubGlobal("fetch", mockFetch);

      const { listDocuments } = await import("./api");

      try {
        await listDocuments();
        expect.fail("Should have thrown");
      } catch (e) {
        expect((e as Error).name).toBe("ApiError");
        expect((e as { detail?: unknown }).detail).toBe("Authentication required");
      }
    });
  });

  describe("presign", () => {
    it("sends filename and size without user_id", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              document_id: "d1",
              s3_key: "k1",
              upload_url: "/upload",
              method: "PUT",
            })
          ),
      });
      vi.stubGlobal("fetch", mockFetch);

      const { presign } = await import("./api");
      await presign("test.pdf", 1024);

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.user_id).toBeUndefined();
      expect(body.filename).toBe("test.pdf");
      expect(body.file_size_bytes).toBe(1024);
    });
  });
});
