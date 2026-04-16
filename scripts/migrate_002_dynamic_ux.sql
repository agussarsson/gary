-- Run once on existing databases created before dynamic UX tables.
-- Fresh installs: use full src/gary/sql/001_init.sql instead.
--
-- Example (Docker, same as init_db.ps1):
--   Get-Content scripts/migrate_002_dynamic_ux.sql | docker exec -i gary-postgres psql -U gary -d gary_db

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS day_name TEXT;

ALTER TABLE adjustments DROP CONSTRAINT IF EXISTS adjustments_field_check;
ALTER TABLE adjustments ADD CONSTRAINT adjustments_field_check
  CHECK (field IN ('load','sets','reps','exercise_swap','rest','pain'));

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
