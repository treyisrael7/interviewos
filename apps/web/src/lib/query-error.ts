import { ApiError, AuthRequiredError } from "@/lib/api";

export function formatQueryError(error: unknown): string {
  if (error instanceof AuthRequiredError) return error.message;
  if (error instanceof ApiError) {
    return String(error.detail ?? error.message);
  }
  if (error instanceof Error) {
    if (error.message === "Failed to fetch") {
      return "Could not reach the API. Ensure the API is running.";
    }
    return error.message;
  }
  return "Something went wrong";
}

/** Dashboard / documents list: friendlier copy for common 401 misconfiguration. */
export function formatDocumentsListError(error: unknown): string {
  if (error instanceof ApiError && error.status === 401) {
    const detail = String(error.detail || "").toLowerCase();
    if (detail && !detail.includes("authentication required")) {
      return String(error.detail);
    }
    return "Session not recognized by API. Add CLERK_JWKS_URL to your API environment (see .env.example).";
  }
  return formatQueryError(error);
}
