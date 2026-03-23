"use client";

import { motion } from "framer-motion";

const nodes = [
  { x: 8, y: 18 },
  { x: 20, y: 34 },
  { x: 32, y: 16 },
  { x: 42, y: 42 },
  { x: 54, y: 22 },
  { x: 68, y: 36 },
  { x: 82, y: 20 },
  { x: 90, y: 44 },
  { x: 12, y: 70 },
  { x: 28, y: 84 },
  { x: 46, y: 72 },
  { x: 62, y: 88 },
  { x: 78, y: 74 },
  { x: 88, y: 62 },
];

const edges = [
  [0, 1],
  [1, 2],
  [2, 4],
  [1, 3],
  [3, 5],
  [4, 5],
  [5, 6],
  [6, 7],
  [1, 8],
  [8, 9],
  [9, 10],
  [10, 11],
  [10, 12],
  [12, 13],
  [5, 10],
  [7, 13],
];

export default function AINetwork() {
  return (
    <div className="absolute inset-0 overflow-hidden opacity-70">
      <svg
        className="absolute inset-0 h-full w-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        {edges.map(([from, to], index) => (
          <motion.line
            key={`${from}-${to}`}
            x1={nodes[from].x}
            y1={nodes[from].y}
            x2={nodes[to].x}
            y2={nodes[to].y}
            stroke="rgba(255,255,255,0.18)"
            strokeWidth="0.18"
            initial={{ opacity: 0.15 }}
            animate={{ opacity: [0.12, 0.32, 0.12] }}
            transition={{
              duration: 4.2,
              repeat: Infinity,
              delay: index * 0.12,
              ease: "easeInOut",
            }}
          />
        ))}
      </svg>

      {nodes.map((node, index) => (
        <motion.div
          key={`${node.x}-${node.y}`}
          className="absolute h-2.5 w-2.5 rounded-full bg-white"
          style={{ top: `${node.y}%`, left: `${node.x}%` }}
          initial={{ opacity: 0.2, scale: 0.8 }}
          animate={{
            opacity: [0.2, 0.8, 0.2],
            scale: [0.9, 1.15, 0.9],
          }}
          transition={{
            duration: 3.4,
            repeat: Infinity,
            delay: index * 0.18,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}
