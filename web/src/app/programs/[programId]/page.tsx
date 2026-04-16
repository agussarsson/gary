"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";
import {
  DeliverProgram,
  createSession,
  getProgram,
} from "@/lib/api";

function ProgramHubContent() {
  const params = useParams<{ programId: string }>();
  const programId = params.programId;
  const router = useRouter();

  const [program, setProgram] = useState<DeliverProgram | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const p = await getProgram(programId);
      setProgram(p);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load program");
    } finally {
      setLoading(false);
    }
  }, [programId]);

  useEffect(() => {
    void load();
  }, [load]);

  async function startDay(dayName: string) {
    setError(null);
    try {
      const session = await createSession({ program_id: programId, day_name: dayName });
      router.push(`/sessions/${session.id}/workout`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start session");
    }
  }

  return (
    <main className="container">
      <nav className="row nav-crumb">
        <Link href="/">Home</Link>
        <span className="muted">/</span>
        <span>{program?.name ?? "Program"}</span>
      </nav>

      <h1>{program?.name ?? "Program"}</h1>
      {loading && <p className="muted">Loading…</p>}
      {error && <p className="error">{error}</p>}

      {program && (
        <>
          <section className="card">
            <h2>Workout days</h2>
            <p className="muted">
              {program.program_json.split} · {program.program_json.days_per_week}{" "}
              days/week · {program.program_json.goal}
            </p>
            <div className="stack">
              {program.program_json.days.map((d) => (
                <div key={d.name} className="card card-nested">
                  <div className="row-between">
                    <div>
                      <strong>{d.name}</strong>
                      <span className="muted"> — {d.focus}</span>
                    </div>
                    <button type="button" onClick={() => void startDay(d.name)}>
                      Start this day
                    </button>
                  </div>
                  <ul className="list-exercises">
                    {d.exercises.map((ex) => (
                      <li key={`${d.name}-${ex.name}`}>
                        {ex.name} — {ex.sets}×{ex.reps}
                        {ex.load != null ? ` @ ${ex.load}` : ""}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
            <p className="row">
              <Link href={`/programs/${programId}/edit`}>Edit program</Link>
            </p>
          </section>
        </>
      )}
    </main>
  );
}

export default function ProgramHubPage() {
  return (
    <Suspense fallback={
      <main className="container">
        <p className="muted">Loading…</p>
      </main>
    }>
      <ProgramHubContent />
    </Suspense>
  );
}
