"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";

const QUESTIONS = [
  "Walk me through a time you improved a slow React interface.",
  "How would you explain `useEffect` to a mid-level frontend engineer?",
  "Tell me about a project where you balanced speed with code quality.",
  "What signals tell you a component should be split into smaller pieces?",
];

function randomBetween(min: number, max: number) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export default function AIInterviewer() {
  const [questionIndex, setQuestionIndex] = useState(0);
  const [isBlinking, setIsBlinking] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [showQuestion, setShowQuestion] = useState(true);

  useEffect(() => {
    if (isThinking) {
      return;
    }

    let isCancelled = false;
    let blinkTimer: number | undefined;
    let reopenTimer: number | undefined;
    let secondBlinkTimer: number | undefined;

    const queueBlink = () => {
      blinkTimer = window.setTimeout(() => {
        if (isCancelled) {
          return;
        }

        const blinkDuration = randomBetween(120, 180);
        const shouldDoubleBlink = Math.random() < 0.18;

        setIsBlinking(true);

        reopenTimer = window.setTimeout(() => {
          if (isCancelled) {
            return;
          }

          setIsBlinking(false);

          if (shouldDoubleBlink) {
            secondBlinkTimer = window.setTimeout(() => {
              if (isCancelled) {
                return;
              }

              setIsBlinking(true);

              reopenTimer = window.setTimeout(() => {
                if (isCancelled) {
                  return;
                }

                setIsBlinking(false);
                queueBlink();
              }, randomBetween(120, 150));
            }, randomBetween(120, 220));
            return;
          }

          queueBlink();
        }, blinkDuration);
      }, randomBetween(2400, 6200));
    };

    queueBlink();

    return () => {
      isCancelled = true;
      window.clearTimeout(blinkTimer);
      window.clearTimeout(reopenTimer);
      window.clearTimeout(secondBlinkTimer);
    };
  }, [isThinking]);

  useEffect(() => {
    let isCancelled = false;
    let thinkingTimer: number | undefined;
    let thinkingEndTimer: number | undefined;
    let questionSwapTimer: number | undefined;

    const queueThinking = () => {
      thinkingTimer = window.setTimeout(() => {
        if (isCancelled) {
          return;
        }

        setIsBlinking(false);
        setIsThinking(true);
        setShowQuestion(false);

        thinkingEndTimer = window.setTimeout(() => {
          if (isCancelled) {
            return;
          }

          setIsThinking(false);
        }, 1560);

        questionSwapTimer = window.setTimeout(() => {
          if (isCancelled) {
            return;
          }

          setQuestionIndex((current) => (current + 1) % QUESTIONS.length);
          setShowQuestion(true);
          queueThinking();
        }, 1760);
      }, randomBetween(6200, 8600));
    };

    queueThinking();

    return () => {
      isCancelled = true;
      window.clearTimeout(thinkingTimer);
      window.clearTimeout(thinkingEndTimer);
      window.clearTimeout(questionSwapTimer);
    };
  }, []);

  return (
    <div className="relative flex w-full max-w-[38rem] items-center justify-center px-2 py-8 sm:py-12 lg:justify-end">
      <div className="pointer-events-none absolute inset-x-4 top-10 h-[20rem] rounded-full bg-orange-400/18 blur-3xl" />

      <div className="relative h-[35rem] w-full max-w-[36rem]">
        <div className="absolute left-0 top-0 z-20 w-[19rem] sm:w-[21rem]">
          <AnimatePresence mode="wait">
            {showQuestion && (
              <motion.div
                key={questionIndex}
                initial={{ opacity: 0, x: -18, y: 10, scale: 0.99 }}
                animate={{ opacity: 1, x: 0, y: 0, scale: 1 }}
                exit={{ opacity: 0, x: 16, y: -6, scale: 0.985 }}
                transition={{ duration: 0.36, ease: [0.22, 1, 0.36, 1] }}
                className="relative rounded-[28px] border border-white/75 bg-white/90 px-5 py-4 text-left text-sm font-medium leading-7 text-slate-800 shadow-[0_24px_60px_rgba(15,23,42,0.12)] backdrop-blur-xl"
              >
                <span className="mb-3 inline-flex rounded-full bg-orange-100 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-orange-700">
                  Interview question
                </span>
                <p>{QUESTIONS[questionIndex]}</p>
                <div className="absolute left-[74%] top-full h-8 w-8 -translate-x-1/2 -translate-y-4 rotate-45 rounded-[6px] border-b border-r border-white/75 bg-white/90" />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="absolute bottom-0 right-0 flex h-[29rem] w-[26rem] items-end justify-center sm:h-[30rem] sm:w-[27rem]">
          <div
            className={`relative z-10 will-change-transform ${isThinking ? "" : "animate-ai-head-float"}`}
          >
            <motion.div
              animate={
                isThinking
                  ? {
                      x: [0, -0.9, 0.9, -0.7, 0.7, 0],
                      y: [0, -0.5, -1.5, -0.5, 0],
                      rotate: [-0.22, 0.18, -0.14, 0.1, -0.08, 0],
                    }
                  : { x: 0, y: 0, rotate: 0 }
              }
              transition={
                isThinking
                  ? {
                      duration: 0.52,
                      repeat: Infinity,
                      ease: "easeInOut",
                    }
                  : {
                      duration: 0.2,
                      ease: "easeOut",
                    }
              }
            >
            <svg
              viewBox="0 0 320 190"
              className="h-[25.5rem] w-[25.5rem] drop-shadow-[0_28px_40px_rgba(15,23,42,0.18)] sm:h-[26.75rem] sm:w-[26.75rem]"
              aria-label="AI interviewer robot head"
              role="img"
            >
              <defs>
                <linearGradient id="helmetShell" x1="0%" x2="100%" y1="0%" y2="100%">
                  <stop offset="0%" stopColor="#fffaf5" />
                  <stop offset="58%" stopColor="#f3ede6" />
                  <stop offset="100%" stopColor="#d9e1e7" />
                </linearGradient>
                <linearGradient id="sidePanel" x1="0%" x2="100%" y1="0%" y2="100%">
                  <stop offset="0%" stopColor="#e7edf2" />
                  <stop offset="100%" stopColor="#ccd5dd" />
                </linearGradient>
              </defs>

              <g transform="translate(0 18)">
                <g transform="rotate(-4 160 73)">
                  <circle cx="112" cy="14" r="4.5" fill="#d4dde4" />
                  <circle cx="205" cy="12" r="4.5" fill="#d4dde4" />
                  <path d="M112 18v21" fill="none" stroke="#d4dde4" strokeWidth="4" strokeLinecap="round" />
                  <path d="M205 16v23" fill="none" stroke="#d4dde4" strokeWidth="4" strokeLinecap="round" />
                  <path
                    d="M94 44c0-31 24-56 54-56h29c31 0 54 25 54 56v45c0 10-7 16-16 16H109c-9 0-15-6-15-16z"
                    fill="url(#helmetShell)"
                  />
                  <rect x="88" y="53" width="17" height="39" rx="7" fill="url(#sidePanel)" />
                  <rect x="211" y="48" width="24" height="48" rx="8" fill="url(#sidePanel)" />
                  <path
                    d="M101 95c10 11 24 16 44 16h35c18 0 32-5 43-16"
                    fill="none"
                    stroke="rgba(148,163,184,0.28)"
                    strokeWidth="6"
                    strokeLinecap="round"
                  />
                  <foreignObject x="110" y="43" width="101" height="49">
                    <div className="relative h-full w-full overflow-hidden rounded-[18px] bg-[#2d3142]">
                      <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.11)_0%,rgba(255,255,255,0.025)_35%,rgba(0,0,0,0.12)_100%)]" />
                      <div className="absolute left-3 right-3 top-[4px] h-[5px] rounded-full bg-[linear-gradient(180deg,rgba(255,255,255,0.24),rgba(255,255,255,0.05))]" />
                      <motion.div
                        className="absolute -top-4 bottom-[-16px] w-8 rotate-[18deg]"
                        style={{
                          background:
                            "linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.18) 45%, rgba(255,255,255,0.24) 50%, rgba(255,255,255,0.18) 55%, rgba(255,255,255,0) 100%)",
                        }}
                        initial={{ x: -34, opacity: 0 }}
                        animate={{ x: 132, opacity: [0, 0.1, 0.22, 0.1, 0] }}
                        transition={{
                          duration: 2.8,
                          repeat: Infinity,
                          ease: "easeInOut",
                        }}
                      />
                      <div className="absolute left-0 right-0 top-0 h-full bg-[linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0)_40%)]" />
                      <div className="absolute inset-x-0 bottom-0 h-5 bg-[linear-gradient(180deg,rgba(255,255,255,0),rgba(0,0,0,0.12))]" />

                      <AnimatePresence mode="wait" initial={false}>
                        {isThinking ? (
                          <motion.div
                            key="thinking"
                            initial={{ opacity: 0, y: 8, scale: 0.9 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: -8, scale: 0.94 }}
                            transition={{ duration: 0.18, ease: [0.2, 0.8, 0.2, 1] }}
                            className="absolute inset-0 flex items-center justify-center"
                          >
                            <div className="rounded-md border border-[#fde047]/20 bg-[#facc15]/[0.06] px-[5px] py-[4px] font-mono text-[8.5px] font-semibold tracking-[0.08em] text-[#fde047] shadow-[0_0_14px_rgba(253,224,71,0.22)]">
                              GENERATING
                            </div>
                          </motion.div>
                        ) : (
                          <motion.div
                            key="eyes"
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 12 }}
                            transition={{ duration: 0.16, ease: [0.2, 0.8, 0.2, 1] }}
                            className="absolute inset-0"
                          >
                            <motion.div
                              className="absolute left-[25px] top-[15px] h-[16px] w-[8px] origin-center bg-[#fde047] shadow-[0_0_8px_rgba(253,224,71,0.28)]"
                              animate={
                                isBlinking
                                  ? { y: 3, scaleX: 1.2, scaleY: 0.16, borderRadius: 999 }
                                  : { y: 0, scaleX: 1, scaleY: 1, borderRadius: 3 }
                              }
                              transition={{ duration: 0.14, ease: [0.32, 0, 0.18, 1] }}
                            />
                            <motion.div
                              className="absolute left-[57px] top-[15px] h-[16px] w-[8px] origin-center bg-[#fde047] shadow-[0_0_8px_rgba(253,224,71,0.28)]"
                              animate={
                                isBlinking
                                  ? { y: 3, scaleX: 1.2, scaleY: 0.16, borderRadius: 999 }
                                  : { y: 0, scaleX: 1, scaleY: 1, borderRadius: 3 }
                              }
                              transition={{ duration: 0.14, ease: [0.32, 0, 0.18, 1] }}
                            />
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </foreignObject>
                  <path
                    d="M114 40c4-7 10-10 18-10h63c8 0 14 3 18 10"
                    fill="none"
                    stroke="#f8fafc"
                    strokeWidth="4"
                    strokeLinecap="round"
                    opacity="0.35"
                  />
                </g>
              </g>
            </svg>
            </motion.div>
          </div>
        </div>
      </div>
      <style jsx>{`
        @keyframes ai-head-float {
          0% {
            transform: translate3d(0, 0, 0) rotate(-0.3deg);
          }
          50% {
            transform: translate3d(0.25px, -9px, 0) rotate(0.24deg);
          }
          100% {
            transform: translate3d(0, 0, 0) rotate(-0.3deg);
          }
        }

        .animate-ai-head-float {
          animation: ai-head-float 8.6s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}
