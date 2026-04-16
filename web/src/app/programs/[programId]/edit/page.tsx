"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ProgramJson, authMe, getProgram, refineProgram, updateProgram } from "@/lib/api";

type ExerciseRow = ProgramJson["days"][0]["exercises"][0];

export default function ProgramEditPage() {
  const params = useParams<{ programId: string }>();
  const programId = params.programId;

  const [name, setName] = useState("");
  const [pj, setPj] = useState<ProgramJson | null>(null);
  const [refineFeedback, setRefineFeedback] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [quota, setQuota] = useState<{ used: number; max: number } | null>(null);
  const [editing, setEditing] = useState<{
    day: number;
    ex: number;
  } | null>(null);
  const [draft, setDraft] = useState<ExerciseRow | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const p = await getProgram(programId);
      setName(p.name);
      setPj(p.program_json);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load program");
    } finally {
      setLoading(false);
    }
  }, [programId]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    async function loadQuota() {
      try {
        const me = await authMe();
        setQuota({ used: me.llm_used_calls, max: me.llm_max_calls });
      } catch {
        setQuota(null);
      }
    }
    void loadQuota();
  }, []);

  function beginEdit(dayIndex: number, exIndex: number) {
    if (!pj) return;
    const ex = pj.days[dayIndex].exercises[exIndex];
    setEditing({ day: dayIndex, ex: exIndex });
    setDraft({ ...ex });
  }

  function cancelEdit() {
    setEditing(null);
    setDraft(null);
  }

  function saveExerciseEdit() {
    if (!pj || !editing || !draft) return;
    const next = structuredClone(pj);
    next.days[editing.day].exercises[editing.ex] = { ...draft };
    setPj(next);
    cancelEdit();
  }

  function deleteExercise(dayIndex: number, exIndex: number) {
    if (!pj) return;
    const next = structuredClone(pj);
    next.days[dayIndex].exercises.splice(exIndex, 1);
    setPj(next);
    cancelEdit();
  }

  async function onSave() {
    if (!pj) return;
    setSaving(true);
    setError(null);
    try {
      const updated = await updateProgram(programId, {
        name: name.trim() || "Program",
        program_json: pj,
      });
      setPj(updated.program_json);
      setName(updated.name);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  async function onRefine() {
    if (!pj || !refineFeedback.trim()) return;
    setSaving(true);
    setError(null);
    try {
      const updated = await refineProgram(programId, {
        program_json: pj,
        feedback: refineFeedback.trim(),
      });
      setPj(updated.program_json);
      setName(updated.name);
      setRefineFeedback("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Refine failed");
      if (e instanceof Error && e.message.toLowerCase().includes("llm call budget exhausted")) {
        try {
          const me = await authMe();
          setQuota({ used: me.llm_used_calls, max: me.llm_max_calls });
        } catch {
          // ignore
        }
      }
    } finally {
      setSaving(false);
    }
  }
  const quotaExhausted = quota != null && quota.used >= quota.max;

  return (
    <main className="container">
      <nav className="row nav-crumb">
        <Link href="/">Home</Link>
        <span className="muted">/</span>
        <Link href={`/programs/${programId}`}>Program</Link>
        <span className="muted">/</span>
        <span>Edit</span>
      </nav>

      <h1>Edit program</h1>
      {loading && <p className="muted">Loading…</p>}
      {error && <p className="error">{error}</p>}

      {pj && (
        <>
          <section className="card">
            <label>
              Name
              <input value={name} onChange={(e) => setName(e.target.value)} />
            </label>
            <div className="row">
              <button
                type="button"
                disabled={saving}
                onClick={() => void onSave()}
              >
                {saving ? "Saving…" : "Save changes"}
              </button>
            </div>
          </section>

          <section className="card">
            <h2>Refine with instructions</h2>
            <p className="muted">
              {quota ? `LLM budget: ${quota.used}/${quota.max}` : "LLM budget unavailable"}
            </p>
            <label>
              Feedback (e.g. swap bench for dumbbells, add a leg day)
              <textarea
                rows={3}
                value={refineFeedback}
                onChange={(e) => setRefineFeedback(e.target.value)}
              />
            </label>
            <button
              type="button"
              disabled={saving || !refineFeedback.trim() || quotaExhausted}
              onClick={() => void onRefine()}
            >
              Apply refinement
            </button>
            {quotaExhausted && (
              <p className="error">Global LLM budget exhausted. Refinement is disabled.</p>
            )}
          </section>

          <section className="card">
            <h2>Days & exercises</h2>
            <div className="stack">
              {pj.days.map((day, di) => (
                <div key={day.name} className="card card-nested">
                  <h3>{day.name}</h3>
                  <p className="muted">{day.focus}</p>
                  <div className="stack">
                    {day.exercises.map((ex, ei) => (
                      <div
                        key={`${day.name}-${ex.name}-${ei}`}
                        className="card-nested"
                      >
                        {editing?.day === di && editing?.ex === ei && draft ? (
                          <div className="stack">
                            <label>
                              Exercise name
                              <input
                                value={draft.name}
                                onChange={(e) =>
                                  setDraft({ ...draft, name: e.target.value })
                                }
                              />
                            </label>
                            <label>
                              Sets
                              <input
                                type="number"
                                min={1}
                                value={draft.sets}
                                onChange={(e) =>
                                  setDraft({
                                    ...draft,
                                    sets: Number(e.target.value),
                                  })
                                }
                              />
                            </label>
                            <label>
                              Reps
                              <input
                                type="number"
                                min={1}
                                value={draft.reps}
                                onChange={(e) =>
                                  setDraft({
                                    ...draft,
                                    reps: Number(e.target.value),
                                  })
                                }
                              />
                            </label>
                            <label>
                              Load (optional)
                              <input
                                type="number"
                                value={draft.load ?? ""}
                                onChange={(e) =>
                                  setDraft({
                                    ...draft,
                                    load:
                                      e.target.value === ""
                                        ? null
                                        : Number(e.target.value),
                                  })
                                }
                              />
                            </label>
                            <label>
                              Progression rule
                              <input
                                value={draft.progression_rule}
                                onChange={(e) =>
                                  setDraft({
                                    ...draft,
                                    progression_rule: e.target.value,
                                  })
                                }
                              />
                            </label>
                            <div className="row">
                              <button type="button" onClick={saveExerciseEdit}>
                                Done
                              </button>
                              <button
                                type="button"
                                className="btn-muted"
                                onClick={cancelEdit}
                              >
                                Cancel
                              </button>
                              <button
                                type="button"
                                className="btn-danger"
                                onClick={() => deleteExercise(di, ei)}
                              >
                                Delete exercise
                              </button>
                            </div>
                          </div>
                        ) : (
                          <>
                            <p>
                              <strong>{ex.name}</strong> — {ex.sets}×{ex.reps}
                              {ex.load != null ? ` @ ${ex.load}` : ""}
                            </p>
                            <p className="muted">{ex.progression_rule}</p>
                            <div className="row">
                              <button
                                type="button"
                                onClick={() => beginEdit(di, ei)}
                              >
                                Modify
                              </button>
                              <button
                                type="button"
                                className="btn-danger"
                                onClick={() => deleteExercise(di, ei)}
                              >
                                Delete
                              </button>
                            </div>
                          </>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </main>
  );
}
