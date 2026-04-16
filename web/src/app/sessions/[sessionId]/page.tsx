"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { DeliverExceptionEvent, DeliverSession, getSession, listSessionExceptions } from "@/lib/api";

export default function SessionPage() {
  const params = useParams<{ sessionId: string }>();
  const sessionId = params.sessionId;

  const [session, setSession] = useState<DeliverSession | null>(null);
  const [events, setEvents] = useState<DeliverExceptionEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [sessionData, eventData] = await Promise.all([
          getSession(sessionId),
          listSessionExceptions(sessionId),
        ]);
        setSession(sessionData);
        setEvents(eventData);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load session");
      } finally {
        setLoading(false);
      }
    }
    void load();
  }, [sessionId]);

  return (
    <main className="container">
      <h1>Session Details</h1>
      {loading && <p className="muted">Loading session…</p>}
      {error && <p className="error">{error}</p>}
      {session && (
        <section className="card">
          <p>Session ID: {session.id}</p>
          <p>Program ID: {session.program_id}</p>
          <p>Date: {session.session_date}</p>
          <p>Completed: {session.completed ? "Yes" : "No"}</p>
        </section>
      )}

      <section className="card">
        <h2>Logged Exceptions</h2>
        {events.length === 0 ? (
          <p className="muted">No exceptions have been logged yet.</p>
        ) : (
          <ul className="stack">
            {events.map((event) => (
              <li key={event.id}>
                {event.exercise_name} - {event.event_type} ({event.severity})
              </li>
            ))}
          </ul>
        )}
      </section>

      <div className="row">
        <Link href={`/sessions/${sessionId}/exceptions/new`}>Log Exception Event</Link>
        <Link href={`/sessions/${sessionId}/adjusted`}>View Adjusted Program</Link>
        <Link href="/">Back to start</Link>
      </div>
    </main>
  );
}
