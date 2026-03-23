import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";
import { NextFetchEvent, NextRequest, NextResponse } from "next/server";

const isProtectedRoute = createRouteMatcher([
  "/dashboard(.*)",
  "/documents(.*)",
  "/interview(.*)",
]);

/**
 * When Clerk is configured (NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY), use Clerk auth.
 * Otherwise fall back to Basic Auth if BASIC_AUTH_USER/BASIC_AUTH_PASSWORD are set.
 */
function basicAuthMiddleware(request: NextRequest) {
  const user = process.env.BASIC_AUTH_USER;
  const password = process.env.BASIC_AUTH_PASSWORD;
  if (!user || !password) return NextResponse.next();

  // Public paths: no Basic auth required (landing, sign-in, sign-up)
  const path = (request.nextUrl.pathname || "/").replace(/\/$/, "") || "/";
  if (path === "/" || path.startsWith("/sign-in") || path.startsWith("/sign-up")) {
    return NextResponse.next();
  }

  const authHeader = request.headers.get("authorization");
  if (!authHeader?.startsWith("Basic ")) {
    return new NextResponse("Authentication required", {
      status: 401,
      headers: { "WWW-Authenticate": 'Basic realm="InterviewOS"' },
    });
  }
  try {
    const [u, p] = Buffer.from(authHeader.slice(6), "base64").toString("utf-8").split(":", 2);
    if (u !== user || p !== password) return new NextResponse("Invalid credentials", { status: 401 });
  } catch {
    return new NextResponse("Invalid credentials", { status: 401 });
  }
  return NextResponse.next();
}

const clerkAuthMiddleware = clerkMiddleware(async (auth, req) => {
  if (isProtectedRoute(req)) await auth.protect();
});

function isPublicPath(request: NextRequest): boolean {
  const pathname = request.nextUrl?.pathname ?? new URL(request.url).pathname ?? "/";
  const path = pathname.replace(/\/$/, "") || "/";
  if (path === "" || path === "/") return true;
  return path.startsWith("/sign-in") || path.startsWith("/sign-up");
}

export default function middleware(
  request: NextRequest,
  event: NextFetchEvent
) {
  // Always allow public paths through first - no auth required
  if (isPublicPath(request)) {
    return NextResponse.next();
  }

  if (process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY) {
    return clerkAuthMiddleware(request, event);
  }
  return basicAuthMiddleware(request);
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
};
