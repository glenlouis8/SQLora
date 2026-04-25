"use client";

import { useState, useRef } from "react";

const API_URL = "https://rockyglen--sql-qlora-sqlmodel-api.modal.run";

const EXAMPLES = [
  {
    question: "What are the top 5 customers by total order value?",
    schema:
      "CREATE TABLE customers (id INT, name TEXT, email TEXT);\nCREATE TABLE orders (id INT, customer_id INT, amount DECIMAL, order_date DATE);",
  },
  {
    question: "List all customers who have never placed an order.",
    schema:
      "CREATE TABLE customers (id INT, name TEXT);\nCREATE TABLE orders (id INT, customer_id INT);",
  },
  {
    question: "What is the average salary by department?",
    schema: "CREATE TABLE employees (id INT, name TEXT, department TEXT, salary DECIMAL);",
  },
];

type Status = "idle" | "loading" | "success" | "error";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [schema, setSchema] = useState("");
  const [sql, setSql] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [copied, setCopied] = useState(false);
  const outputRef = useRef<HTMLDivElement>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim() || !schema.trim()) return;

    setStatus("loading");
    setErrorMsg("");
    setSql("");

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim(), schema: schema.trim() }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();

      if (data.error) {
        setErrorMsg(data.error);
        setStatus("error");
      } else {
        setSql(data.sql);
        setStatus("success");
        setTimeout(() => outputRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to reach API.");
      setStatus("error");
    }
  }

  function loadExample(i: number) {
    setQuestion(EXAMPLES[i].question);
    setSchema(EXAMPLES[i].schema);
    setSql("");
    setStatus("idle");
    setErrorMsg("");
  }

  async function handleCopy() {
    await navigator.clipboard.writeText(sql);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <main className="min-h-screen bg-paper">
      {/* Noise texture overlay */}
      <div
        className="fixed inset-0 pointer-events-none opacity-[0.03]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        }}
      />

      <div className="relative max-w-3xl mx-auto px-6 py-12 md:py-20">

        {/* Header */}
        <header className="mb-14">
          <div className="flex items-baseline gap-4 mb-3">
            <h1 className="font-display text-6xl md:text-7xl text-ink leading-none tracking-tight">
              SQL.GEN
            </h1>
            <span className="font-mono text-xs text-muted border border-muted px-2 py-1 mb-1 self-end">
              v1.0
            </span>
          </div>
          <div className="border-t border-ink pt-4 flex flex-wrap gap-x-6 gap-y-1">
            <span className="font-mono text-xs text-muted">
              Model · Llama 3.2 3B Instruct
            </span>
            <span className="font-mono text-xs text-muted">
              Method · QLoRA (r=16)
            </span>
            <span className="font-mono text-xs text-muted">
              Dataset · sql-create-context
            </span>
            <span className="font-mono text-xs text-muted">
              ROUGE-L · 0.9856
            </span>
          </div>
        </header>

        {/* Examples */}
        <div className="mb-8">
          <p className="font-mono text-xs uppercase tracking-widest text-muted mb-3">
            Examples
          </p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex, i) => (
              <button
                key={i}
                onClick={() => loadExample(i)}
                className="font-mono text-xs px-3 py-1.5 border border-ink hover:bg-ink hover:text-paper transition-colors shadow-hard"
              >
                {i + 1}. {ex.question.slice(0, 32)}…
              </button>
            ))}
          </div>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Question */}
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-muted mb-2">
              Natural Language Question
            </label>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. What are the top 5 customers by total order value?"
              rows={3}
              className="w-full border border-ink bg-paper font-mono text-sm text-ink p-4 focus:outline-none focus:shadow-hard-lg shadow-hard transition-shadow placeholder:text-muted/40"
            />
          </div>

          {/* Schema */}
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-muted mb-2">
              SQL Schema
            </label>
            <textarea
              value={schema}
              onChange={(e) => setSchema(e.target.value)}
              placeholder={"CREATE TABLE customers (id INT, name TEXT, email TEXT);\nCREATE TABLE orders (id INT, customer_id INT, amount DECIMAL);"}
              rows={7}
              className="w-full border border-ink bg-paper font-mono text-sm text-ink p-4 focus:outline-none focus:shadow-hard-lg shadow-hard transition-shadow placeholder:text-muted/40 scrollbar-thin"
            />
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={status === "loading" || !question.trim() || !schema.trim()}
            className="w-full font-mono text-sm uppercase tracking-widest py-4 bg-ink text-paper border border-ink hover:bg-accent hover:border-accent transition-colors disabled:opacity-40 disabled:cursor-not-allowed shadow-hard-lg active:translate-x-[3px] active:translate-y-[3px] active:shadow-none"
          >
            {status === "loading" ? (
              <span className="flex items-center justify-center gap-3">
                <span>Generating</span>
                <span className="flex gap-1">
                  {[0, 1, 2].map((i) => (
                    <span
                      key={i}
                      className="inline-block w-1 h-1 bg-paper animate-blink"
                      style={{ animationDelay: `${i * 0.2}s` }}
                    />
                  ))}
                </span>
              </span>
            ) : (
              "Generate SQL →"
            )}
          </button>
        </form>

        {/* Loading hint */}
        {status === "loading" && (
          <p className="mt-4 font-mono text-xs text-muted animate-fade-up">
            Cold start may take 1–2 min on first request. Subsequent requests are fast.
          </p>
        )}

        {/* Error */}
        {status === "error" && (
          <div className="mt-10 border border-accent p-5 shadow-hard-accent animate-fade-up">
            <p className="font-mono text-xs text-accent uppercase tracking-widest mb-2">
              Error
            </p>
            <p className="font-mono text-sm text-ink">{errorMsg}</p>
          </div>
        )}

        {/* SQL Output */}
        {status === "success" && sql && (
          <div ref={outputRef} className="mt-10 animate-fade-up">
            <p className="font-mono text-xs uppercase tracking-widest text-muted mb-2">
              Generated SQL
            </p>
            <div className="border border-ink shadow-hard-lg">
              {/* Code block titlebar */}
              <div className="bg-ink px-4 py-2.5 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-paper/20" />
                  <span className="w-2.5 h-2.5 rounded-full bg-paper/20" />
                  <span className="w-2.5 h-2.5 rounded-full bg-paper/20" />
                  <span className="ml-3 font-mono text-xs text-paper/40 uppercase tracking-widest">
                    output.sql
                  </span>
                </div>
                <button
                  onClick={handleCopy}
                  className="font-mono text-xs text-paper/50 hover:text-paper transition-colors px-2 py-0.5 border border-paper/20 hover:border-paper/50"
                >
                  {copied ? "copied ✓" : "copy"}
                </button>
              </div>
              {/* Code */}
              <pre className="bg-ink p-6 overflow-x-auto scrollbar-thin">
                <code className="font-mono text-sm text-paper leading-relaxed whitespace-pre-wrap break-words">
                  {sql}
                </code>
              </pre>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-20 pt-6 border-t border-ink/20">
          <p className="font-mono text-xs text-muted">
            Served via{" "}
            <a
              href="https://modal.com"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2 hover:text-ink transition-colors"
            >
              Modal
            </a>{" "}
            · Adapter on{" "}
            <a
              href="https://huggingface.co/glen-louis/llama-3.2-3b-sql-qlora"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2 hover:text-ink transition-colors"
            >
              HuggingFace
            </a>
          </p>
        </footer>
      </div>
    </main>
  );
}
