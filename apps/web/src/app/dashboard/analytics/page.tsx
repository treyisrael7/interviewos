"use client";

import Link from "next/link";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { GradientShell } from "@/components/GradientShell";
import { useInterviewAnalyticsOverview } from "@/hooks/use-interview-analytics";
import { formatQueryError } from "@/lib/query-error";
import { Lightbulb, TrendingDown, TrendingUp } from "lucide-react";

const DIFFICULTY_LABELS: Record<string, string> = {
  junior: "Junior",
  mid: "Mid",
  senior: "Senior",
};

function formatSessionDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return iso;
  }
}

function truncateLabel(s: string, max = 26): string {
  const t = s.trim();
  return t.length <= max ? t : `${t.slice(0, max - 1)}…`;
}

function buildInsights(data: {
  last_session_vs_prior_percent_change: number | null;
  focus_area_hint: string | null;
}): { icon: "up" | "down" | "flat" | "light"; text: string }[] {
  const items: { icon: "up" | "down" | "flat" | "light"; text: string }[] = [];
  const pct = data.last_session_vs_prior_percent_change;

  if (pct != null) {
    if (pct > 0.5) {
      items.push({
        icon: "up",
        text: `You improved ${pct}% compared to your previous practice session.`,
      });
    } else if (pct < -0.5) {
      items.push({
        icon: "down",
        text: `Your last session averaged ${Math.abs(pct)}% lower than the one before—normal variance; steady reps add up.`,
      });
    } else {
      items.push({
        icon: "flat",
        text: "You are holding steady session over session.",
      });
    }
  }

  if (data.focus_area_hint) {
    items.push({
      icon: "light",
      text: `Focus on ${data.focus_area_hint} questions in your next reps.`,
    });
  }

  return items;
}

function InsightIcon({ kind }: { kind: "up" | "down" | "flat" | "light" }) {
  const cls = "mt-0.5 h-5 w-5 shrink-0";
  if (kind === "up") {
    return <TrendingUp className={`${cls} text-emerald-600`} aria-hidden />;
  }
  if (kind === "down") {
    return <TrendingDown className={`${cls} text-amber-700`} aria-hidden />;
  }
  if (kind === "flat") {
    return (
      <span
        className={`${cls} flex items-center justify-center text-zenodrift-text-muted`}
        aria-hidden
      >
        —
      </span>
    );
  }
  return <Lightbulb className={`${cls} text-zenodrift-accent`} aria-hidden />;
}

export default function InterviewAnalyticsPage() {
  const { data, isPending, isError, error } = useInterviewAnalyticsOverview();

  const queryError = isError && error ? formatQueryError(error) : null;

  return (
    <GradientShell>
      <div className="mx-auto w-full max-w-[1160px] space-y-8 pb-10 pt-6 sm:pt-10">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <Link
              href="/dashboard"
              className="text-sm font-medium text-zenodrift-text-muted transition-colors hover:text-zenodrift-accent"
            >
              ← Dashboard
            </Link>
            <h1 className="mt-3 text-3xl font-bold tracking-tight text-zenodrift-text-strong sm:text-4xl">
              Practice analytics
            </h1>
            <p className="mt-2 max-w-xl text-sm leading-relaxed text-zenodrift-text-muted sm:text-base">
              Track scores over time, see where you shine, and spot the next skill to sharpen.
            </p>
          </div>
        </div>

        {queryError && (
          <div
            className="dashboard-card border-red-200/60 bg-red-50/70 px-5 py-4 text-sm text-red-800"
            role="alert"
          >
            {queryError}
          </div>
        )}

        {isPending && !queryError && (
          <div className="flex justify-center py-20">
            <div
              className="h-10 w-10 animate-spin rounded-full border-2 border-neutral-200 border-t-zenodrift-accent"
              aria-label="Loading analytics"
            />
          </div>
        )}

        {!isPending && data && (
          <>
            {data.total_answer_count === 0 ? (
              <section className="dashboard-card px-8 py-12 text-center">
                <p className="text-lg font-medium text-zenodrift-text-strong">
                  No scored answers yet
                </p>
                <p className="mx-auto mt-2 max-w-md text-sm text-zenodrift-text-muted">
                  Complete a few interview answers to unlock your trend line, competency
                  breakdown, and session history.
                </p>
                <Link
                  href="/dashboard"
                  className="mt-6 inline-flex rounded-full bg-gradient-to-r from-orange-500 to-orange-600 px-6 py-2.5 text-sm font-semibold text-white shadow-md transition hover:-translate-y-0.5 hover:shadow-lg"
                >
                  Go to dashboard
                </Link>
              </section>
            ) : (
              <>
                <section className="grid gap-4 sm:grid-cols-3">
                  <div className="dashboard-card px-6 py-5">
                    <p className="text-xs font-medium uppercase tracking-wider text-zenodrift-text-muted">
                      Average score
                    </p>
                    <p className="mt-2 text-3xl font-bold tabular-nums text-zenodrift-text-strong">
                      {data.overall_average_score != null
                        ? `${data.overall_average_score}`
                        : "—"}
                      {data.overall_average_score != null && (
                        <span className="ml-1 text-lg font-semibold text-zenodrift-text-muted">
                          / 100
                        </span>
                      )}
                    </p>
                    <p className="mt-1 text-xs text-zenodrift-text-muted">
                      Across {data.total_answer_count} scored answer
                      {data.total_answer_count !== 1 ? "s" : ""}
                    </p>
                  </div>
                  <div className="dashboard-card px-6 py-5">
                    <p className="text-xs font-medium uppercase tracking-wider text-zenodrift-text-muted">
                      Answers logged
                    </p>
                    <p className="mt-2 text-3xl font-bold tabular-nums text-zenodrift-text-strong">
                      {data.total_answer_count}
                    </p>
                    <p className="mt-1 text-xs text-zenodrift-text-muted">
                      Each point builds your trend line
                    </p>
                  </div>
                  <div className="dashboard-card px-6 py-5">
                    <p className="text-xs font-medium uppercase tracking-wider text-zenodrift-text-muted">
                      Practice sessions
                    </p>
                    <p className="mt-2 text-3xl font-bold tabular-nums text-zenodrift-text-strong">
                      {data.total_session_count}
                    </p>
                    <p className="mt-1 text-xs text-zenodrift-text-muted">
                      Interview runs started from your JDs
                    </p>
                  </div>
                </section>

                {buildInsights(data).length > 0 && (
                  <section className="dashboard-card px-6 py-5">
                    <h2 className="text-xs font-medium uppercase tracking-wider text-zenodrift-text-muted">
                      Insights
                    </h2>
                    <ul className="mt-4 space-y-3">
                      {buildInsights(data).map((item, i) => (
                        <li key={i} className="flex gap-3 text-sm leading-relaxed text-zenodrift-text">
                          <InsightIcon kind={item.icon} />
                          <span>{item.text}</span>
                        </li>
                      ))}
                    </ul>
                  </section>
                )}

                <section className="dashboard-card px-6 py-6">
                  <h2 className="text-sm font-semibold text-zenodrift-text-strong">
                    Score over time
                  </h2>
                  <p className="mt-1 text-xs text-zenodrift-text-muted">
                    Chronological order of every scored answer
                  </p>
                  <div className="mt-4 h-[280px] w-full min-w-0">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={data.score_trend.map((p, i) => ({
                          n: i + 1,
                          score: p.score,
                          at: formatSessionDate(p.at),
                        }))}
                        margin={{ top: 8, right: 12, left: -8, bottom: 8 }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="rgba(66, 63, 58, 0.12)"
                          vertical={false}
                        />
                        <XAxis
                          dataKey="n"
                          name="Answer #"
                          tick={{ fill: "#6b6560", fontSize: 11 }}
                          tickLine={false}
                          axisLine={{ stroke: "rgba(66, 63, 58, 0.15)" }}
                        />
                        <YAxis
                          domain={[0, 100]}
                          tick={{ fill: "#6b6560", fontSize: 11 }}
                          tickLine={false}
                          axisLine={{ stroke: "rgba(66, 63, 58, 0.15)" }}
                          width={36}
                        />
                        <Tooltip
                          contentStyle={{
                            borderRadius: "12px",
                            border: "1px solid rgba(255,255,255,0.5)",
                            boxShadow: "0 4px 24px rgb(0 0 0 / 0.08)",
                          }}
                          labelFormatter={(_, payload) => {
                            const p = payload?.[0]?.payload as
                              | { n: number; at: string }
                              | undefined;
                            return p ? `Answer ${p.n} · ${p.at}` : "";
                          }}
                          formatter={(value: number) => [`${value}`, "Score"]}
                        />
                        <Line
                          type="monotone"
                          dataKey="score"
                          stroke="#ea580c"
                          strokeWidth={2.5}
                          dot={{ fill: "#ea580c", strokeWidth: 0, r: 4 }}
                          activeDot={{ r: 6, fill: "#c2410c" }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </section>

                {(data.strongest_competencies.length > 0 ||
                  data.weakest_competencies.length > 0) && (
                  <section className="grid gap-6 lg:grid-cols-2">
                    <div className="dashboard-card px-6 py-6">
                      <h2 className="text-sm font-semibold text-zenodrift-text-strong">
                        Strengths
                      </h2>
                      <p className="mt-1 text-xs text-zenodrift-text-muted">
                        Highest average scores by competency
                      </p>
                      <div
                        className="mt-4 w-full min-w-0"
                        style={{
                          height: Math.max(180, data.strongest_competencies.length * 40),
                        }}
                      >
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            layout="vertical"
                            data={data.strongest_competencies.map((c) => ({
                              name: truncateLabel(c.competency_label),
                              full: c.competency_label,
                              score: c.average_score,
                            }))}
                            margin={{ left: 4, right: 16, top: 4, bottom: 4 }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              horizontal={false}
                              stroke="rgba(66, 63, 58, 0.08)"
                            />
                            <XAxis
                              type="number"
                              domain={[0, 100]}
                              tick={{ fill: "#6b6560", fontSize: 11 }}
                            />
                            <YAxis
                              type="category"
                              dataKey="name"
                              width={118}
                              tick={{ fill: "#6b6560", fontSize: 11 }}
                            />
                            <Tooltip
                              formatter={(value: number) => [`${value}`, "Avg score"]}
                              labelFormatter={(_, payload) => {
                                const row = payload?.[0]?.payload as
                                  | { full: string }
                                  | undefined;
                                return row?.full ?? "";
                              }}
                              contentStyle={{
                                borderRadius: "12px",
                                border: "1px solid rgba(255,255,255,0.5)",
                              }}
                            />
                            <Bar
                              dataKey="score"
                              fill="#059669"
                              radius={[0, 6, 6, 0]}
                              barSize={18}
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                    <div className="dashboard-card px-6 py-6">
                      <h2 className="text-sm font-semibold text-zenodrift-text-strong">
                        Growth areas
                      </h2>
                      <p className="mt-1 text-xs text-zenodrift-text-muted">
                        Competencies with the most room to improve
                      </p>
                      <div
                        className="mt-4 w-full min-w-0"
                        style={{
                          height: Math.max(180, data.weakest_competencies.length * 40),
                        }}
                      >
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            layout="vertical"
                            data={data.weakest_competencies.map((c) => ({
                              name: truncateLabel(c.competency_label),
                              full: c.competency_label,
                              score: c.average_score,
                            }))}
                            margin={{ left: 4, right: 16, top: 4, bottom: 4 }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              horizontal={false}
                              stroke="rgba(66, 63, 58, 0.08)"
                            />
                            <XAxis
                              type="number"
                              domain={[0, 100]}
                              tick={{ fill: "#6b6560", fontSize: 11 }}
                            />
                            <YAxis
                              type="category"
                              dataKey="name"
                              width={118}
                              tick={{ fill: "#6b6560", fontSize: 11 }}
                            />
                            <Tooltip
                              formatter={(value: number) => [`${value}`, "Avg score"]}
                              labelFormatter={(_, payload) => {
                                const row = payload?.[0]?.payload as
                                  | { full: string }
                                  | undefined;
                                return row?.full ?? "";
                              }}
                              contentStyle={{
                                borderRadius: "12px",
                                border: "1px solid rgba(255,255,255,0.5)",
                              }}
                            />
                            <Bar
                              dataKey="score"
                              fill="#ea580c"
                              radius={[0, 6, 6, 0]}
                              barSize={18}
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </section>
                )}

                <section className="dashboard-card px-6 py-6">
                  <h2 className="text-sm font-semibold text-zenodrift-text-strong">
                    Recent sessions
                  </h2>
                  <p className="mt-1 text-xs text-zenodrift-text-muted">
                    Newest first · jump back in anytime
                  </p>
                  {data.recent_sessions.length === 0 ? (
                    <p className="mt-6 text-sm text-zenodrift-text-muted">
                      No sessions yet.
                    </p>
                  ) : (
                    <ul className="mt-4 divide-y divide-neutral-100/80">
                      {data.recent_sessions.map((s) => (
                        <li
                          key={s.id}
                          className="flex flex-col gap-2 py-4 first:pt-0 last:pb-0 sm:flex-row sm:items-center sm:justify-between"
                        >
                          <div>
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="font-medium text-zenodrift-text-strong">
                                {formatSessionDate(s.created_at)}
                              </span>
                              <span className="rounded-full bg-white/80 px-2.5 py-0.5 text-xs font-medium text-zenodrift-text ring-1 ring-black/5">
                                {DIFFICULTY_LABELS[s.difficulty] ?? s.difficulty}
                              </span>
                            </div>
                            <p className="mt-1 text-xs text-zenodrift-text-muted">
                              {s.question_count} question
                              {s.question_count !== 1 ? "s" : ""}
                              {s.answer_count > 0 && (
                                <>
                                  {" "}
                                  · {s.answer_count} scored answer
                                  {s.answer_count !== 1 ? "s" : ""}
                                  {s.average_score != null && (
                                    <> · avg {s.average_score}</>
                                  )}
                                </>
                              )}
                            </p>
                          </div>
                          <Link
                            href={`/interview/session/${s.id}`}
                            className="shrink-0 text-sm font-semibold text-zenodrift-accent transition-colors hover:text-zenodrift-accent-hover"
                          >
                            Open session →
                          </Link>
                        </li>
                      ))}
                    </ul>
                  )}
                </section>
              </>
            )}
          </>
        )}
      </div>
    </GradientShell>
  );
}
