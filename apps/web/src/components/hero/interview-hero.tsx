"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, Bot, FileText, ShieldCheck, Sparkles } from "lucide-react";
import AIInterviewer from "@/components/ui/ai-interviewer";
import AINetwork from "@/components/background/ai-network";

const badgeItems = [
  "Job description-grounded",
  "Evidence-cited feedback",
  "Resume-aware practice",
];

const featureItems = [
  {
    icon: FileText,
    label: "Tailored to each role",
  },
  {
    icon: ShieldCheck,
    label: "Structured, evidence-backed feedback",
  },
  {
    icon: Bot,
    label: "AI interviewer that feels live",
  },
];

export default function InterviewHero() {
  return (
    <section className="relative min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.72),_transparent_34%),linear-gradient(135deg,_#ff7a45_0%,_#ff9b6b_32%,_#ffd3b8_72%,_#fff2e8_100%)]">
      <motion.div
        className="absolute inset-0"
        animate={{
          backgroundPosition: [
            "0% 50%, 0% 0%",
            "100% 50%, 100% 100%",
            "0% 50%, 0% 0%",
          ],
        }}
        transition={{ duration: 18, repeat: Infinity, ease: "linear" }}
        style={{
          backgroundImage:
            "radial-gradient(circle at 20% 20%, rgba(255,255,255,0.4), transparent 26%), radial-gradient(circle at 80% 30%, rgba(255,209,174,0.42), transparent 30%), radial-gradient(circle at 50% 80%, rgba(255,146,74,0.22), transparent 28%)",
          backgroundSize: "140% 140%",
        }}
      />
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.12)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.12)_1px,transparent_1px)] bg-[size:36px_36px] opacity-30 [mask-image:radial-gradient(circle_at_center,black,transparent_82%)]" />
      <AINetwork />

      <div className="relative z-10 mx-auto flex min-h-screen max-w-7xl items-center px-6 py-16 sm:px-8 lg:px-12">
        <div className="grid w-full grid-cols-1 items-center gap-16 lg:grid-cols-[1.05fr_0.95fr]">
          <div>
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45, ease: "easeOut" }}
              className="inline-flex items-center gap-2 rounded-full border border-white/60 bg-white/65 px-4 py-2 text-sm font-medium text-slate-700 shadow-[0_18px_50px_rgba(15,23,42,0.08)] backdrop-blur-xl"
            >
              <Sparkles className="h-4 w-4 text-orange-500" />
              AI interview practice for serious candidates
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.55, delay: 0.05, ease: "easeOut" }}
              className="mt-6 max-w-3xl text-[clamp(3.3rem,8vw,6.2rem)] font-semibold leading-[0.94] tracking-[-0.07em] text-slate-950"
            >
              InterviewOS
              <span className="mt-4 block max-w-2xl text-[clamp(1.1rem,2.3vw,1.7rem)] font-medium leading-[1.6] tracking-[-0.035em] text-slate-700">
                Practice with an AI interviewer that adapts to your resume, the job description, and the signals hiring teams actually care about.
              </span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.55, delay: 0.12, ease: "easeOut" }}
              className="mt-8 max-w-2xl text-lg leading-9 text-slate-600"
            >
              Upload a JD, add your resume, and run realistic mock interviews with targeted technical and behavioral prompts, live follow-ups, and grounded feedback you can actually use.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.55, delay: 0.18, ease: "easeOut" }}
              className="mt-10 flex flex-col gap-4 sm:flex-row"
            >
              <Link
                href="/sign-up"
                className="group inline-flex items-center justify-center gap-2 rounded-2xl bg-slate-950 px-6 py-3.5 text-sm font-semibold text-white shadow-[0_18px_45px_rgba(15,23,42,0.22)] transition duration-200 hover:-translate-y-0.5 hover:bg-slate-900"
              >
                Sign up free
                <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
              </Link>
              <Link
                href="/sign-in"
                className="inline-flex items-center justify-center rounded-2xl border border-white/70 bg-white/55 px-6 py-3.5 text-sm font-semibold text-slate-800 shadow-[0_18px_45px_rgba(15,23,42,0.08)] backdrop-blur-xl transition duration-200 hover:-translate-y-0.5 hover:bg-white/75"
              >
                Sign in
              </Link>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.55, delay: 0.25, ease: "easeOut" }}
              className="mt-10 flex flex-wrap gap-3"
            >
              {badgeItems.map((item) => (
                <span
                  key={item}
                  className="rounded-full border border-white/60 bg-white/55 px-4 py-2 text-sm font-medium text-slate-700 backdrop-blur-xl"
                >
                  {item}
                </span>
              ))}
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.55, delay: 0.32, ease: "easeOut" }}
              className="mt-10 grid gap-3 sm:max-w-xl sm:grid-cols-2"
            >
              {featureItems.map(({ icon: Icon, label }) => (
                <div
                  key={label}
                  className="rounded-3xl border border-white/55 bg-white/50 px-4 py-4 text-sm font-medium text-slate-700 shadow-[0_18px_40px_rgba(15,23,42,0.08)] backdrop-blur-xl"
                >
                  <Icon className="mb-3 h-5 w-5 text-orange-500" />
                  {label}
                </div>
              ))}
            </motion.div>
          </div>

          <motion.div
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.15, ease: "easeOut" }}
            className="relative flex items-center justify-center"
          >
            <div className="absolute h-[24rem] w-[24rem] rounded-full bg-white/25 blur-3xl" />
            <AIInterviewer />
          </motion.div>
        </div>
      </div>
    </section>
  );
}
