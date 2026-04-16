CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS programs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  program_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  program_id UUID NOT NULL REFERENCES programs(id) ON DELETE CASCADE,
  session_date DATE NOT NULL DEFAULT CURRENT_DATE,
  completed BOOLEAN NOT NULL DEFAULT FALSE,
  day_name TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sessions_program_id ON sessions(program_id);

CREATE TABLE IF NOT EXISTS exception_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,

  exercise_name TEXT NOT NULL,

  event_type TEXT NOT NULL CHECK (event_type IN (
    'too_heavy','too_light','pain','time','equipment','form','other'
  )),
  severity TEXT NOT NULL CHECK (severity IN ('low','medium','high')),
  body_area TEXT NOT NULL DEFAULT 'none' CHECK (body_area IN (
    'none','lower_back','shoulder','knee','elbow','wrist','hip','neck','ankle','other'
  )),

  note TEXT,
  confidence REAL CHECK (confidence >= 0 AND confidence <= 1),

  source TEXT NOT NULL DEFAULT 'manual' CHECK (source IN ('manual','rules','model','llm')),

  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_exception_events_session_id ON exception_events(session_id);

CREATE TABLE IF NOT EXISTS adjustments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  program_id UUID NOT NULL REFERENCES programs(id) ON DELETE CASCADE,

  exercise_name TEXT NOT NULL,
  field TEXT NOT NULL DEFAULT 'load' CHECK (field IN ('load','sets','reps','exercise_swap','rest','pain')),

  old_value TEXT,
  new_value TEXT,

  reason TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_adjustments_program_id ON adjustments(program_id);

CREATE TABLE IF NOT EXISTS day_completions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  program_id UUID NOT NULL REFERENCES programs(id) ON DELETE CASCADE,
  session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
  day_name TEXT NOT NULL,
  day_index INT NOT NULL DEFAULT 0,
  note TEXT,
  completed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_day_completions_program_day ON day_completions(program_id, day_name);
CREATE INDEX IF NOT EXISTS idx_day_completions_completed_at ON day_completions(completed_at DESC);

CREATE TABLE IF NOT EXISTS progression_suggestions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  program_id UUID NOT NULL REFERENCES programs(id) ON DELETE CASCADE,
  session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
  suggestion_type TEXT NOT NULL CHECK (suggestion_type IN (
    'note_feedback','no_note_progression','rule_based'
  )),
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
    'pending','accepted','declined','modified'
  )),
  payload JSONB NOT NULL DEFAULT '{}',
  rationale TEXT,
  day_name TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_progression_suggestions_program ON progression_suggestions(program_id, status);
CREATE INDEX IF NOT EXISTS idx_progression_suggestions_pending ON progression_suggestions(status) WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS suggestion_decisions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  suggestion_id UUID NOT NULL REFERENCES progression_suggestions(id) ON DELETE CASCADE,
  decision TEXT NOT NULL CHECK (decision IN ('accept','decline','modify')),
  modified_program_json JSONB,
  decided_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_suggestion_decisions_suggestion ON suggestion_decisions(suggestion_id);

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS auth_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  session_token_hash TEXT NOT NULL UNIQUE,
  expires_at TIMESTAMPTZ NOT NULL,
  revoked_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id ON auth_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_token_hash ON auth_sessions(session_token_hash);

CREATE TABLE IF NOT EXISTS llm_quota (
  id SMALLINT PRIMARY KEY CHECK (id = 1),
  used_calls INT NOT NULL DEFAULT 0,
  max_calls INT NOT NULL DEFAULT 10,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO llm_quota(id, used_calls, max_calls)
VALUES (1, 0, 10)
ON CONFLICT (id) DO NOTHING;
