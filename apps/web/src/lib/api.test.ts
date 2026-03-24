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
