"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { WorkoutStyle, authMe, generateProgram, logout } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [goal, setGoal] = useState("Build strength");
  const [daysPerWeek, setDaysPerWeek] = useState(3);
  const [experience, setExperience] = useState("beginner");
  const [workoutStyle, setWorkoutStyle] = useState<WorkoutStyle>("gym_equipment");
  const [programName, setProgramName] = useState("My Program");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [quota, setQuota] = useState<{ used: number; max: number } | null>(null);

  useEffect(() => {
    async function loadAuth() {
      try {
        const me = await authMe();
        setQuota({ used: me.llm_used_calls, max: me.llm_max_calls });
      } catch {
        setQuota(null);
      }
    }
    void loadAuth();
  }, []);

  async function onGenerate() {
    setError(null);
    setLoading(true);
    try {
      const res = await generateProgram({
        goal,
        days_per_week: daysPerWeek,
        experience_level: experience,
        workout_style: workoutStyle,
        program_name: programName,
      });
      router.push(`/programs/${res.program.id}/edit`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed");
      if (e instanceof Error && e.message.toLowerCase().includes("llm call budget exhausted")) {
        try {
          const me = await authMe();
          setQuota({ used: me.llm_used_calls, max: me.llm_max_calls });
        } catch {
          // ignore quota refresh failures
        }
      }
    } finally {
      setLoading(false);
    }
  }
  const quotaExhausted = quota != null && quota.used >= quota.max;

  return (
    <main className="container">
      <h1>Welcome to Gary</h1>
      <div className="row-between">
        <p className="muted">
          {quota ? `LLM budget: ${quota.used}/${quota.max}` : "LLM budget unavailable"}
        </p>
        <button
          type="button"
          className="btn-muted"
          onClick={async () => {
            await logout();
            router.push("/login");
          }}
        >
          Sign out
        </button>
      </div>
      <p className="muted">
        Generate a structured program (Gemini when configured, otherwise a
        safe template). Then edit days and exercises before you train.
      </p>

      <section className="card">
        <h2>Generate your program</h2>
        <label>
          Program name
          <input
            value={programName}
            onChange={(e) => setProgramName(e.target.value)}
          />
        </label>
        <label>
          Goal
          <input value={goal} onChange={(e) => setGoal(e.target.value)} />
        </label>
        <label>
          Days per week
          <input
            type="number"
            min={1}
            max={7}
            value={daysPerWeek}
            onChange={(e) => setDaysPerWeek(Number(e.target.value))}
          />
        </label>
        <label>
          Experience
          <input
            value={experience}
            onChange={(e) => setExperience(e.target.value)}
          />
        </label>
        <label>
          Workout style
          <select
            value={workoutStyle}
            onChange={(e) => setWorkoutStyle(e.target.value as WorkoutStyle)}
          >
            <option value="gym_equipment">Gym equipment</option>
            <option value="body_weight">Body weight</option>
          </select>
        </label>
        <button type="button" disabled={loading || quotaExhausted} onClick={onGenerate}>
          {loading ? "Generating…" : "Generate program"}
        </button>
        {quotaExhausted && (
          <p className="error">
            Global LLM budget exhausted. Generation/refinement is temporarily disabled.
          </p>
        )}
      </section>

      {error && <p className="error">{error}</p>}

      <p className="muted">
        After generation you are taken to the editor. Your program hub URL is{" "}
        <code>/programs/&lt;id&gt;</code> (copy it from the address bar anytime).
      </p>
    </main>
  );
}
