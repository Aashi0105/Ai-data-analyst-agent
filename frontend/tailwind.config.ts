import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        cream: "#FAF9F5",
        surface: "#FFFFFF",
        charcoal: {
          DEFAULT: "#1C1917",
          muted: "#57534E",
          subtle: "#78716C",
        },
        editorial: {
          border: "#E7E5E0",
          hover: "#F5F4F0",
          accent: "#2D5A43",
          sage: "#4A7C59",
          plum: "#4A3F55",
          ochre: "#9C6615",
        }
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        serif: ["var(--font-serif)", "Georgia", "Cambria", "serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
