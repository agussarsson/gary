"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import {
  DeliverProgram,
  DeliverSession,
  DeliverSuggestion,
  completeDay,
  getProgram,
  getSession,
  listProgramSuggestions,
  suggestionsFromNote,
} from "@/lib/api";
import SuggestionReviewModal from "@/app/components/SuggestionReviewModal";

export default function WorkoutRunPage() {
  const params = useParams<{ sessionId: string }>();
  const sessionId = params.sessionId;

  const [session, setSession] = useState<DeliverSession | null>(null);
  const [program, setProgram] = useState<DeliverProgram | null>(null);
  const [started, setStarted] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [note, setNote] = useState("");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [postCompleteMessage, setPostCompleteMessage] = useState<string | null>(null);
  const [modalSuggestions, setModalSuggestions] = useState<DeliverSuggestion[]>([]);
  const [showSuggestionsModal, setShowSuggestionsModal] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await getSession(sessionId);
      setSession(s);
      const p = await getProgram(s.program_id);
      setProgram(p);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load workout");
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    void load();
  }, [load]);

  const dayName = session?.day_name;
  const day =
    program && dayName
      ? program.program_json.days.find((d) => d.name === dayName)
      : undefined;

  async function onComplete() {
    if (!session || !dayName) return;
    setSubmitting(true);
    setError(null);
    try {
      const trimmed = note.trim();
      await completeDay(sessionId, {
        day_name: dayName,
        note: trimmed || undefined,
      });
      if (trimmed.length >= 2) {
        await suggestionsFromNote(sessionId, trimmed);
      }
      const pending = await listProgramSuggestions(session.program_id);
      const forSession = pending.filter((s) => s.session_id === session.id);
      setCompleted(true);
      if (forSession.length > 0) {
        setModalSuggestions(forSession);
        setShowSuggestionsModal(true);
        setPostCompleteMessage(null);
      } else {
        setModalSuggestions([]);
        setShowSuggestionsModal(false);
        setPostCompleteMessage(
          "Workout complete! No review actions needed right now.",
        );
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not complete day");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="container">
      <nav className="row nav-crumb">
        <Link href="/">Home</Link>
        {program && (
          <>
            <span className="muted">/</span>
            <Link href={`/programs/${program.id}`}>Program</Link>
          </>
        )}
      </nav>

      <h1>Workout</h1>
      {loading && <p className="muted">Loading…</p>}
      {error && <p className="error">{error}</p>}

      {!dayName && session && (
        <p className="error">
          This session has no day assigned. Create a new session from the
          program hub with &quot;Start this day&quot;.
        </p>
      )}

      {day && program && (
        <>
          <section className="card">
            <h2>{day.name}</h2>
            <p className="muted">{day.focus}</p>
            <ul className="list-exercises">
              {day.exercises.map((ex) => (
                <li key={ex.name}>
                  {ex.name} — {ex.sets}×{ex.reps}
                  {ex.load != null ? ` @ ${ex.load}` : ""}
                </li>
              ))}
            </ul>
            {!started ? (
              <button type="button" onClick={() => setStarted(true)}>
                Start workout
              </button>
            ) : (
              <p className="muted">Session in progress — mark complete when done.</p>
            )}
          </section>

          {started && !completed && (
            <section className="card">
              <h2>Complete</h2>
              <label>
                Notes (optional — adds reviewable suggestions)
                <textarea
                  rows={4}
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  placeholder="e.g. bench felt heavy on last set"
                />
              </label>
              <button
                type="button"
                disabled={submitting}
                onClick={() => void onComplete()}
              >
                {submitting ? "Saving…" : "Workout complete"}
              </button>
            </section>
          )}

          {completed && (
            <section className="card">
              <h2>Workout complete!</h2>
              {postCompleteMessage && <p className="muted">{postCompleteMessage}</p>}
              {!showSuggestionsModal && (
                <div className="row">
                  <Link href={`/programs/${program.id}`}>Back to program</Link>
                  <button
                    type="button"
                    className="btn-muted"
                    onClick={() => {
                      setStarted(false);
                      setCompleted(false);
                      setNote("");
                      setPostCompleteMessage(null);
                    }}
                  >
                    Start another workout
                  </button>
                </div>
              )}
            </section>
          )}
        </>
      )}
      {showSuggestionsModal && program && (
        <SuggestionReviewModal
          programJson={program.program_json}
          suggestions={modalSuggestions}
          onProgramJsonUpdated={(nextJson) =>
            setProgram((prev) =>
              prev ? { ...prev, program_json: nextJson } : prev,
            )
          }
          onClose={() => {
            setShowSuggestionsModal(false);
            setPostCompleteMessage("Suggestions reviewed and saved.");
          }}
        />
      )}
    </main>
  );
}
