import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Data Analyst | Natural Language Tabular Intelligence",
  description: "Turn CSV datasets into verified insights, statistics, and visualizations using plain conversational English.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-cream text-charcoal flex flex-col font-sans selection:bg-editorial-hover selection:text-charcoal">
        {children}
      </body>
    </html>
  );
}
