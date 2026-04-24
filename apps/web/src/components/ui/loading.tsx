"use client";

import type { ReactNode } from "react";
import { motion, useReducedMotion } from "framer-motion";

const SPINNER_SIZE = {
  sm: "h-6 w-6 border-2",
  md: "h-8 w-8 border-2",
  lg: "h-10 w-10 border-2",
} as const;

const SPINNER_VARIANT = {
  default: "border-neutral-200 border-t-zenodrift-accent",
  light: "border-white/55 border-t-zenodrift-accent",
  warm: "border-amber-300/90 border-t-zenodrift-accent",
  modal: "border-slate-200 border-t-orange-500",
} as const;

export type LoadingSpinnerSize = keyof typeof SPINNER_SIZE;
export type LoadingSpinnerVariant = keyof typeof SPINNER_VARIANT;

type LoadingSpinnerProps = {
  size?: LoadingSpinnerSize;
  variant?: LoadingSpinnerVariant;
  className?: string;
  /** Accessible name; defaults to "Loading" */
  label?: string;
  /** When true, hides role/label (use when nearby text describes state). */
  decorative?: boolean;
};

/**
 * Smooth continuous rotation (linear). Respects prefers-reduced-motion with a soft pulse instead.
 */
export function LoadingSpinner({
  size = "md",
  variant = "default",
  className = "",
  label = "Loading",
  decorative = false,
}: LoadingSpinnerProps) {
  const reduceMotion = useReducedMotion();
  const dim = SPINNER_SIZE[size];
  const colors = SPINNER_VARIANT[variant];
  const a11y = decorative
    ? ({ "aria-hidden": true } as const)
    : ({ role: "status" as const, "aria-label": label });

  if (reduceMotion) {
    return (
      <div
        {...a11y}
        className={`rounded-full border-2 border-dashed opacity-70 ${dim} ${colors} ${className}`}
      />
    );
  }

  return (
    <motion.div
      {...a11y}
      className={`rounded-full border-2 ${dim} ${colors} ${className}`}
      animate={{ rotate: 360 }}
      transition={{ duration: 0.78, repeat: Infinity, ease: "linear" }}
    />
  );
}

type FadeInProps = {
  children: ReactNode;
  className?: string;
  delay?: number;
};

/** Gentle entrance for loading shells and content blocks. */
export function FadeIn({ children, className = "", delay = 0 }: FadeInProps) {
  const reduceMotion = useReducedMotion();
  if (reduceMotion) {
    return <div className={className}>{children}</div>;
  }
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.38,
        delay,
        ease: [0.22, 1, 0.36, 1],
      }}
    >
      {children}
    </motion.div>
  );
}

type LoadingCenterProps = {
  /** Spinner size */
  size?: LoadingSpinnerSize;
  variant?: LoadingSpinnerVariant;
  className?: string;
  message?: string;
  label?: string;
};

/** Vertically centered loading block with optional caption. */
export function LoadingCenter({
  size = "lg",
  variant = "default",
  className = "",
  message,
  label,
}: LoadingCenterProps) {
  return (
    <FadeIn className={className}>
      <div className="flex flex-col items-center justify-center gap-4 py-4">
        <LoadingSpinner size={size} variant={variant} label={label ?? message ?? "Loading"} />
        {message ? (
          <motion.p
            className="text-sm text-zenodrift-text-muted"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.08, duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
          >
            {message}
          </motion.p>
        ) : null}
      </div>
    </FadeIn>
  );
}

/** Inline spinner + label for compact spaces (forms, sidebars). */
export function LoadingRow({
  message,
  size = "sm",
  variant = "default",
  className = "",
}: {
  message: string;
  size?: LoadingSpinnerSize;
  variant?: LoadingSpinnerVariant;
  className?: string;
}) {
  return (
    <FadeIn className={`flex items-center gap-3 ${className}`}>
      <LoadingSpinner size={size} variant={variant} label={message} />
      <span className="text-sm text-zenodrift-text-muted">{message}</span>
    </FadeIn>
  );
}

/** Shimmer-style placeholders for the job list on the dashboard. */
export function DocumentListSkeleton({ className = "" }: { className?: string }) {
  const reduceMotion = useReducedMotion();
  return (
    <FadeIn className={`divide-y divide-neutral-100/90 py-1 ${className}`}>
      {[0, 1, 2].map((i) => (
        <div key={i} className="flex flex-col gap-3 py-4 first:pt-1 sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0 flex-1 space-y-2.5">
            <motion.div
              className="h-4 max-w-[min(280px,55vw)] rounded-md bg-neutral-200/75"
              animate={
                reduceMotion
                  ? { opacity: 0.65 }
                  : { opacity: [0.45, 0.85, 0.45] }
              }
              transition={
                reduceMotion
                  ? {}
                  : {
                      duration: 1.45,
                      repeat: Infinity,
                      ease: "easeInOut",
                      delay: i * 0.12,
                    }
              }
            />
            <motion.div
              className="h-3 w-28 rounded-md bg-neutral-200/55"
              animate={
                reduceMotion
                  ? { opacity: 0.55 }
                  : { opacity: [0.4, 0.75, 0.4] }
              }
              transition={
                reduceMotion
                  ? {}
                  : {
                      duration: 1.45,
                      repeat: Infinity,
                      ease: "easeInOut",
                      delay: i * 0.12 + 0.08,
                    }
              }
            />
          </div>
          <motion.div
            className="h-9 w-[5.5rem] shrink-0 self-start rounded-lg bg-neutral-200/50 sm:self-center"
            animate={
              reduceMotion ? { opacity: 0.55 } : { opacity: [0.42, 0.72, 0.42] }
            }
            transition={
              reduceMotion
                ? {}
                : {
                    duration: 1.45,
                    repeat: Infinity,
                    ease: "easeInOut",
                    delay: i * 0.12 + 0.04,
                  }
            }
          />
        </div>
      ))}
    </FadeIn>
  );
}

/** Placeholder layout for analytics while the overview query resolves. */
export function AnalyticsOverviewSkeleton({ className = "" }: { className?: string }) {
  const reduceMotion = useReducedMotion();
  const pulse = (delay: number) =>
    reduceMotion
      ? { opacity: 0.65 }
      : { opacity: [0.45, 0.82, 0.45] };
  const pulseTransition = (delay: number) =>
    reduceMotion
      ? {}
      : { duration: 1.5, repeat: Infinity, ease: "easeInOut" as const, delay };

  return (
    <FadeIn className={`mx-auto w-full max-w-[1160px] space-y-6 ${className}`}>
      <div className="grid gap-4 sm:grid-cols-2">
        <motion.div
          className="dashboard-card h-[220px] rounded-zenodrift-panel bg-white/50 p-4"
          animate={pulse(0)}
          transition={pulseTransition(0)}
        >
          <div className="mb-4 h-3 w-24 rounded bg-neutral-200/70" />
          <div className="h-[160px] w-full rounded-xl bg-neutral-200/40" />
        </motion.div>
        <motion.div
          className="dashboard-card h-[220px] rounded-zenodrift-panel bg-white/50 p-4"
          animate={pulse(0.1)}
          transition={pulseTransition(0.1)}
        >
          <div className="mb-4 h-3 w-28 rounded bg-neutral-200/70" />
          <div className="h-[160px] w-full rounded-xl bg-neutral-200/40" />
        </motion.div>
      </div>
      <motion.div
        className="dashboard-card h-32 rounded-zenodrift-panel bg-white/50 px-5 py-4"
        animate={pulse(0.15)}
        transition={pulseTransition(0.15)}
      >
        <div className="h-3 w-36 rounded bg-neutral-200/70" />
        <div className="mt-4 h-3 w-full max-w-lg rounded bg-neutral-200/45" />
        <div className="mt-2 h-3 w-[80%] max-w-md rounded bg-neutral-200/40" />
      </motion.div>
    </FadeIn>
  );
}
