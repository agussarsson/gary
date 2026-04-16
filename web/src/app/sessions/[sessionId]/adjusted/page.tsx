"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";
import { ApplyAdjustmentResponse, applyAdjustment } from "@/lib/api";

export default function AdjustedProgramPage() {
  const params = useParams<{ sessionId: string }>();
  const sessionId = params.sessionId;
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ApplyAdjustmentResponse | null>(null);

  async function onApply() {
    setLoading(true);
    setError(null);
    try {
      const data = await applyAdjustment(sessionId);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to apply adjustments");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <h1>Adjusted Program</h1>
      <p className="muted">Step 4/4: apply workout adjustments for this session.</p>
      <div className="row">
        <button type="button" disabled={loading} onClick={onApply}>
          {loading ? "Applying..." : "Apply Adjustments"}
        </button>
        <Link href={`/sessions/${sessionId}`}>Back to session</Link>
      </div>
      {error && <p className="error">{error}</p>}
      {result && (
        <>
          {result.message && <p>{result.message}</p>}
          <section className="card">
            <h2>Adjustments ({result.adjustments.length})</h2>
            {result.adjustments.length === 0 ? (
              <p className="muted">No adjustments returned.</p>
            ) : (
              <ul className="stack">
                {result.adjustments.map((adj, idx) => (
                  <li key={`${adj.exercise_name}-${adj.field}-${idx}`}>
                    {adj.exercise_name} - {adj.field}
                    {adj.multiplier ? ` x${adj.multiplier}` : ""}
                    {adj.reason ? ` (${adj.reason})` : ""}
                  </li>
                ))}
              </ul>
            )}
          </section>
          <section className="card">
            <h2>Program JSON</h2>
            <pre>{JSON.stringify(result.program_json, null, 2)}</pre>
          </section>
        </>
      )}
    </main>
  );
}
