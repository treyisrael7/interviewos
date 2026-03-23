"use client";

import Link from "next/link";
import { UserButton, useUser } from "@clerk/nextjs";

export function AppChrome() {
  const { isSignedIn } = useUser();

  return (
    <>
      <div className="fixed left-0 right-0 top-0 z-40 flex items-center justify-end px-6 py-4">
        <div className="flex items-center gap-2">
            {isSignedIn ? (
              <UserButton
                appearance={{
                  elements: {
                    avatarBox: "h-8 w-8",
                  },
                }}
              />
            ) : (
              <>
                <Link
                  href="/sign-in"
                  className="rounded-lg border border-white/40 bg-white/30 px-4 py-2 text-sm font-medium text-zenodrift-text-strong backdrop-blur-sm hover:bg-white/50 transition-colors"
                >
                  Sign in
                </Link>
                <Link
                  href="/sign-up"
                  className="rounded-lg bg-zenodrift-accent/90 px-4 py-2 text-sm font-medium text-white backdrop-blur-sm hover:bg-zenodrift-accent transition-colors"
                >
                  Sign up
                </Link>
              </>
            )}
          </div>
        </div>
    </>
  );
}
