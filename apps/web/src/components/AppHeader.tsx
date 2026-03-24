"use client";

import { useState } from "react";
import Link from "next/link";
import { LibraryModal } from "./LibraryModal";

export function AppHeader() {
  const [libraryOpen, setLibraryOpen] = useState(false);

  return (
    <>
      <header className="fixed left-0 right-0 top-0 z-40 border-b border-white/30 bg-white/55 backdrop-blur-xl shadow-zenodrift-soft">
        <div className="mx-auto flex h-14 max-w-[1200px] items-center justify-between px-4 sm:px-6 lg:px-8">
          <Link
            href="/dashboard"
            className="text-lg font-semibold text-zenodrift-text-strong transition-colors hover:text-zenodrift-text"
          >
            InterviewOS
          </Link>
          <div className="flex items-center gap-3 sm:gap-4">
            <Link
              href="/dashboard/analytics"
              className="text-sm font-medium text-zenodrift-text-muted transition-colors hover:text-zenodrift-accent"
            >
              Analytics
            </Link>
            <button
              type="button"
              onClick={() => setLibraryOpen(true)}
              className="flex items-center gap-2 rounded-full border border-neutral-200 bg-white px-4 py-2 text-sm font-medium text-zenodrift-text shadow-zenodrift-soft transition-all duration-200 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-zenodrift-accent/25 focus:ring-offset-2 focus:ring-offset-transparent"
            >
              <svg
                className="h-4 w-4 text-zenodrift-text-muted"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                />
              </svg>
              Library
            </button>
          </div>
        </div>
      </header>
      <LibraryModal isOpen={libraryOpen} onClose={() => setLibraryOpen(false)} />
    </>
  );
}
