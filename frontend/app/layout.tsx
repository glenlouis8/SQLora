import type { Metadata } from "next";
import { Playfair_Display, Fira_Code } from "next/font/google";
import "./globals.css";

const playfair = Playfair_Display({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["700", "900"],
});

const firaCode = Fira_Code({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["300", "400", "500", "600"],
});

export const metadata: Metadata = {
  title: "SQL.GEN — Text-to-SQL",
  description: "Llama 3.2 3B fine-tuned on sql-create-context with QLoRA. Generate SQL from natural language.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${playfair.variable} ${firaCode.variable}`}>
      <body className="font-mono bg-paper text-ink">{children}</body>
    </html>
  );
}
