/**
 * Auth token provider for API client.
 * Clerk populates this when the user is signed in (see ClerkAuthProvider).
 */

let _getToken: (() => Promise<string | null>) | null = null;

export function setAuthTokenProvider(getToken: () => Promise<string | null>) {
  _getToken = getToken;
}

export function hasAuthTokenProvider(): boolean {
  return _getToken !== null;
}

export async function getAuthToken(): Promise<string | null> {
  if (!_getToken) return null;
  return _getToken();
}
