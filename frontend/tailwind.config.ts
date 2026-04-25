import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ["var(--font-display)", "Georgia", "serif"],
        mono: ["var(--font-mono)", "'Courier New'", "monospace"],
      },
      colors: {
        paper: "#F5F2EC",
        ink: "#0F0F0F",
        accent: "#B91C1C",
        muted: "#6B6560",
        "paper-dark": "#EAE6DF",
      },
      boxShadow: {
        hard: "3px 3px 0px 0px #0F0F0F",
        "hard-lg": "5px 5px 0px 0px #0F0F0F",
        "hard-accent": "3px 3px 0px 0px #B91C1C",
      },
      keyframes: {
        blink: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0" },
        },
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        blink: "blink 1s step-end infinite",
        "fade-up": "fade-up 0.4s ease-out forwards",
      },
    },
  },
  plugins: [],
};

export default config;
