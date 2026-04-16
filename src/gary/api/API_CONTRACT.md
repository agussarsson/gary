# API Contract

Contract for the Gary FastAPI backend and Next.js client.

## Health

- `GET /health` → `{ "status": "ok" }`
- `GET /ready` → database readiness

## Authentication

- `POST /auth/signup`
  - Request: `AuthSignupRequest` (`email`, `password`)
  - Behavior: hard signup cap at 5 users (configurable by `AUTH_MAX_USERS`)
  - Response: `AuthMeResponse` and sets `gary_session` cookie
- `POST /auth/login`
  - Request: `AuthLoginRequest` (`email`, `password`)
  - Response: authenticated payload and sets `gary_session` cookie
- `POST /auth/logout` → `{ ok: true }` and clears session cookie
- `GET /auth/me` → `AuthMeResponse` (`authenticated`, `user`, `llm_used_calls`, `llm_max_calls`)

## Programs

- `POST /programs/generate`
  - Request: `ProgramGenerateRequest` (`goal`, `days_per_week`, `experience_level`, `workout_style` = `gym_equipment` | `body_weight`, optional `preferences`, `program_name`)
  - Auth required
  - Enforced global LLM quota (`LLM_MAX_CALLS`, default 10) when Gemini key is configured
  - Response: `GenerateProgramResponse` (`program`, `source` = `gemini` | `fallback`, optional `message`)
- `POST /programs`
  - Request: `CreateProgram`
  - Response: `DeliverProgram`
- `GET /programs/{program_id}` → `DeliverProgram`
- `PUT /programs/{program_id}`
  - Request: `CreateProgram` (same shape as create: `name`, `program_json`)
  - Response: `DeliverProgram`
- `POST /programs/{program_id}/refine`
  - Request: `ProgramRefineRequest` (`program_json`, `feedback`)
  - Auth required
  - Enforced global LLM quota (`LLM_MAX_CALLS`, default 10) when Gemini key is configured
  - Response: `DeliverProgram`

## Sessions

- `POST /sessions`
  - Request: **CreateSession**: `program_id`, optional `session_date`, optional `day_name` (workout day label from the program)
  - Response: `DeliverSession` (includes optional `day_name`)
- `GET /sessions/{session_id}` → `DeliverSession`

## Workout day & suggestions

- `POST /sessions/{session_id}/complete-day`
  - Request: `CompleteDayRequest`: `day_name`, optional `note`
  - Response: `{ ok, progression_suggestion_id }` — after three recent no-note completions for that day, may create a pending progression suggestion
- `POST /sessions/{session_id}/suggestions/from-note`
  - Request: `WorkoutFeedbackRequest`: `text` (min length 2)
  - Response: `DeliverSuggestion` (pending; does not mutate the program)
- `GET /programs/{program_id}/suggestions` → list of pending `DeliverSuggestion`
- `POST /suggestions/{suggestion_id}/decision`
  - Request: `SuggestionDecisionRequest`: `decision` ∈ `accept` | `decline` | `modify`, optional `modified_program_json` (required for `modify`)
  - Response: `{ ok, program_json }` — `program_json` set when program was updated

## Exception Events

- `POST /exceptions` → `DeliverExceptionEvent`
- `GET /sessions/{session_id}/exceptions` → list of `DeliverExceptionEvent`

## Adjustments

- `POST /sessions/{session_id}/apply-adjustment` → `ApplyAdjustmentResponse`

## Workout Feedback (legacy structured path)

- `POST /sessions/{session_id}/workout-feedback` → `WorkoutFeedbackResponse`

## Contract notes

- `event_type`, `severity`, `body_area`, and `source` are strict enums where applicable.
- `workout_style` is a validated enum in generation requests.
- `Exercise.reps` is an integer (no rep ranges).
- Cookie session auth is required for all non-auth endpoints.
- `program_json` must match `ProgramJSON`.
- Unknown request fields are rejected (`extra="forbid"`).
