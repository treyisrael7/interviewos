"use client";

import { motion } from "framer-motion";

const cards = [
  { label: "System Design", top: "14%", left: "2%", delay: 0.1 },
  { label: "React Hooks", top: "28%", left: "14%", delay: 0.4 },
  { label: "SQL Queries", top: "60%", left: "8%", delay: 0.9 },
  { label: "Behavioral", top: "18%", left: "74%", delay: 0.6 },
  { label: "REST APIs", top: "62%", left: "78%", delay: 1.1 },
];

export default function FloatingCards() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      {cards.map((card, index) => (
        <motion.div
          key={card.label}
          className="absolute hidden rounded-2xl border border-white/55 bg-white/70 px-4 py-2 text-sm font-medium text-slate-700 shadow-[0_20px_60px_rgba(15,23,42,0.08)] backdrop-blur-xl md:block"
          style={{
            top: card.top,
            left: card.left,
          }}
          animate={{
            y: [0, -18, 0],
            rotate: [0, index % 2 === 0 ? 2 : -2, 0],
          }}
          transition={{
            duration: 6 + index,
            repeat: Infinity,
            delay: card.delay,
            ease: "easeInOut",
          }}
        >
          {card.label}
        </motion.div>
      ))}
    </div>
  );
}
