import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getAuthToken, hasAuthTokenProvider, setAuthTokenProvider } from "./auth";

describe("auth", () => {
  beforeEach(() => {
    setAuthTokenProvider(() => Promise.resolve(null));
  });

  afterEach(() => {
    setAuthTokenProvider(() => Promise.resolve(null));
  });

  it("hasAuthTokenProvider is false before any setAuthTokenProvider (fresh module)", async () => {
    vi.resetModules();
    const { hasAuthTokenProvider: hasProv, setAuthTokenProvider: setProv } = await import("./auth");
    expect(hasProv()).toBe(false);
    setProv(() => Promise.resolve(null));
    expect(hasProv()).toBe(true);
  });

  it("getAuthToken returns null when provider returns null", async () => {
    setAuthTokenProvider(() => Promise.resolve(null));
    expect(await getAuthToken()).toBeNull();
  });

  it("getAuthToken returns token when provider returns token", async () => {
    setAuthTokenProvider(() => Promise.resolve("jwt-token-123"));
    expect(await getAuthToken()).toBe("jwt-token-123");
  });

});
