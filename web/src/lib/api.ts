export type EventType =
  | "pain"
  | "too_heavy"
  | "too_light"
  | "time"
  | "equipment"
  | "form"
  | "other";

export type Severity = "high" | "medium" | "low";
export type BodyArea =
  | "none"
  | "lower_back"
  | "shoulder"
  | "knee"
  | "elbow"
  | "wrist"
  | "hip"
  | "neck"
  | "ankle"
  | "other";
export type Source = "manual" | "rules" | "model" | "llm";
export type WorkoutStyle = "gym_equipment" | "body_weight";

export type ProgramJson = {
  split: string;
  days_per_week: number;
  goal: string;
  experience_level: string;
  notes?: string | null;
  days: {
    name: string;
    focus: string;
    exercises: {
      name: string;
      sets: number;
      reps: number;
      load?: number | null;
      progression_rule: string;
    }[];
  }[];
};

export type DeliverProgram = {
  id: string;
  name: string;
  program_json: ProgramJson;
};

export type DeliverSession = {
  id: string;
  program_id: string;
  session_date: string;
  completed: boolean;
  day_name?: string | null;
};

export type DeliverExceptionEvent = {
  id: string;
  session_id: string;
  exercise_name: string;
  event_type: EventType;
  severity: Severity;
  body_area: BodyArea;
  note?: string | null;
  confidence?: number | null;
  source: Source;
};

export type ApplyAdjustmentResponse = {
  message?: string | null;
  program_id?: string | null;
  program_json: ProgramJson;
  adjustments: {
    exercise_name: string;
    field: string;
    multiplier?: number | null;
    reason?: string | null;
    body_part?: string | null;
  }[];
};

export type GenerateProgramResponse = {
  program: DeliverProgram;
  source: string;
  message?: string | null;
};

export type DeliverSuggestion = {
  id: string;
  program_id: string;
  session_id: string | null;
  suggestion_type: string;
  status: string;
  payload: { adjustments?: ApplyAdjustmentResponse["adjustments"] };
  rationale: string | null;
  day_name: string | null;
  created_at: string;
};

export type AuthMe = {
  authenticated: boolean;
  user: { id: string; email: string } | null;
  llm_used_calls: number;
  llm_max_calls: number;
};

/** Empty → same-origin `/backend-api` (Next rewrites to FastAPI). Set full URL for prod / Docker. */
const envApiBase = process.env.NEXT_PUBLIC_API_BASE_URL?.trim() ?? "";
const API_BASE_URL = envApiBase.replace(/\/$/, "");
const API_PROXY_PREFIX = "/backend-api";

function apiUrl(path: string): string {
  if (API_BASE_URL) return `${API_BASE_URL}${path}`;
  if (typeof window !== "undefined") {
    return `${window.location.origin}${API_PROXY_PREFIX}${path}`;
  }
  return `http://127.0.0.1:8000${path}`;
}

type ErrJson = { error?: { message?: string }; detail?: string; message?: string };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers ?? undefined);
  if (init?.body != null && init.body !== "") {
    if (!headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
  }

  const response = await fetch(apiUrl(path), {
    ...init,
    headers,
    credentials: "include",
    cache: "no-store",
  });

  if (!response.ok) {
    let payload: ErrJson | null = null;
    try {
      payload = (await response.json()) as ErrJson;
    } catch {
      payload = null;
    }
    const message =
      payload?.error?.message ??
      (typeof payload?.detail === "string" ? payload.detail : undefined) ??
      payload?.message ??
      `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export async function generateProgram(input: {
  goal: string;
  days_per_week: number;
  experience_level: string;
  workout_style: WorkoutStyle;
  preferences?: string[];
  program_name?: string;
}): Promise<GenerateProgramResponse> {
  return request<GenerateProgramResponse>("/programs/generate", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function getProgram(programId: string): Promise<DeliverProgram> {
  return request<DeliverProgram>(`/programs/${programId}`);
}

export async function createProgram(input: {
  name: string;
  program_json: ProgramJson;
}): Promise<DeliverProgram> {
  return request<DeliverProgram>("/programs", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function refineProgram(
  programId: string,
  input: { program_json: ProgramJson; feedback: string },
): Promise<DeliverProgram> {
  return request<DeliverProgram>(`/programs/${programId}/refine`, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function updateProgram(
  programId: string,
  input: { name: string; program_json: ProgramJson },
): Promise<DeliverProgram> {
  return request<DeliverProgram>(`/programs/${programId}`, {
    method: "PUT",
    body: JSON.stringify(input),
  });
}

export async function createSession(input: {
  program_id: string;
  session_date?: string;
  day_name?: string;
}): Promise<DeliverSession> {
  return request<DeliverSession>("/sessions", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function getSession(sessionId: string): Promise<DeliverSession> {
  return request<DeliverSession>(`/sessions/${sessionId}`);
}

export async function completeDay(
  sessionId: string,
  input: { day_name: string; note?: string },
): Promise<{ ok: boolean; progression_suggestion_id: string | null }> {
  return request(`/sessions/${sessionId}/complete-day`, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function suggestionsFromNote(
  sessionId: string,
  text: string,
): Promise<DeliverSuggestion> {
  return request<DeliverSuggestion>(
    `/sessions/${sessionId}/suggestions/from-note`,
    {
      method: "POST",
      body: JSON.stringify({ text }),
    },
  );
}

export async function listProgramSuggestions(
  programId: string,
): Promise<DeliverSuggestion[]> {
  return request<DeliverSuggestion[]>(`/programs/${programId}/suggestions`);
}

export async function suggestionDecision(
  suggestionId: string,
  input: {
    decision: "accept" | "decline" | "modify";
    modified_program_json?: ProgramJson;
  },
): Promise<{ ok: boolean; program_json: ProgramJson | null }> {
  return request(`/suggestions/${suggestionId}/decision`, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function listSessionExceptions(
  sessionId: string,
): Promise<DeliverExceptionEvent[]> {
  return request<DeliverExceptionEvent[]>(`/sessions/${sessionId}/exceptions`);
}

export async function createException(input: {
  session_id: string;
  exercise_name: string;
  event_type: EventType;
  severity: Severity;
  body_area?: BodyArea;
  note?: string;
  confidence?: number;
  source?: Source;
}): Promise<DeliverExceptionEvent> {
  return request<DeliverExceptionEvent>("/exceptions", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function applyAdjustment(
  sessionId: string,
): Promise<ApplyAdjustmentResponse> {
  return request<ApplyAdjustmentResponse>(
    `/sessions/${sessionId}/apply-adjustment`,
    { method: "POST" },
  );
}

export async function signup(input: { email: string; password: string }): Promise<AuthMe> {
  return request<AuthMe>("/auth/signup", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function login(input: { email: string; password: string }): Promise<AuthMe> {
  return request<AuthMe>("/auth/login", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function logout(): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>("/auth/logout", { method: "POST" });
}

export async function authMe(): Promise<AuthMe> {
  return request<AuthMe>("/auth/me");
}
