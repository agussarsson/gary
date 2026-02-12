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
  field TEXT NOT NULL DEFAULT 'load' CHECK (field IN ('load','sets','reps','exercise_swap','rest')),

  old_value TEXT,
  new_value TEXT,

  reason TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_adjustments_program_id ON adjustments(program_id);
