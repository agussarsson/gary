"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { FormEvent, useState } from "react";
import { BodyArea, EventType, Severity, createException } from "@/lib/api";

const eventTypes: EventType[] = [
  "pain",
  "too_heavy",
  "too_light",
  "time",
  "equipment",
  "form",
  "other",
];
const severities: Severity[] = ["low", "medium", "high"];
const bodyAreas: BodyArea[] = [
  "none",
  "lower_back",
  "shoulder",
  "knee",
  "elbow",
  "wrist",
  "hip",
  "neck",
  "ankle",
  "other",
];

export default function NewExceptionPage() {
  const params = useParams<{ sessionId: string }>();
  const sessionId = params.sessionId;
  const router = useRouter();

  const [exerciseName, setExerciseName] = useState("Bench Press");
  const [eventType, setEventType] = useState<EventType>("too_heavy");
  const [severity, setSeverity] = useState<Severity>("medium");
  const [bodyArea, setBodyArea] = useState<BodyArea>("none");
  const [note, setNote] = useState("");
  const [confidence, setConfidence] = useState("0.8");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await createException({
        session_id: sessionId,
        exercise_name: exerciseName,
        event_type: eventType,
        severity,
        body_area: bodyArea,
        note: note || undefined,
        confidence: confidence ? Number(confidence) : undefined,
        source: "manual",
      });
      router.push(`/sessions/${sessionId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create exception");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <h1>Log Exception Event</h1>
      <form className="card stack" onSubmit={onSubmit}>
        <label>
          Exercise name
          <input value={exerciseName} onChange={(e) => setExerciseName(e.target.value)} />
        </label>
        <label>
          Event type
          <select value={eventType} onChange={(e) => setEventType(e.target.value as EventType)}>
            {eventTypes.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label>
          Severity
          <select value={severity} onChange={(e) => setSeverity(e.target.value as Severity)}>
            {severities.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label>
          Body area
          <select value={bodyArea} onChange={(e) => setBodyArea(e.target.value as BodyArea)}>
            {bodyAreas.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label>
          Note
          <textarea rows={4} value={note} onChange={(e) => setNote(e.target.value)} />
        </label>
        <label>
          Confidence (0-1)
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={confidence}
            onChange={(e) => setConfidence(e.target.value)}
          />
        </label>
        <button type="submit" disabled={loading}>
          {loading ? "Saving..." : "Save Exception"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      <div className="row">
        <Link href={`/sessions/${sessionId}`}>Back to session</Link>
      </div>
    </main>
  );
}
